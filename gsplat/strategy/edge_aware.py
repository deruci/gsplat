"""Edge-Aware Densification Strategy for 3D Gaussian Splatting.

Faithful implementation of "Improving Densification in 3DGS for High-Fidelity
Rendering" (arXiv 2508.12313), based on the official code at
https://github.com/XiaoBin2001/Improved-GS.

Key improvements over DefaultStrategy:
1. Edge-Aware Score (EAS): PIL FIND_EDGES → normalize → pass as pixel_weights
   to CUDA rasterizer → per-Gaussian importance via atomicAdd(weight * T * alpha)
2. Long-Axis Split (LAS): Multinomial sampling weighted by edge importance,
   split along longest axis with d = 0.45 * scale_max * 3, children at 55%/89.3%
3. Recovery-Aware Pruning: Bottom 20% opacity at iterations 300, 3300, 6300
4. Budget schedule: sqrt ramp N_max * sqrt((iter - start) / (end - start))
5. Multi-step optimizer: every 1 iter (0-15K) → every 5 (15-22.5K) → every 20 (22.5K+)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .default import DefaultStrategy
from .ops import _update_param_with_optimizer, duplicate, remove, reset_opa


def _normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternion [w, x, y, z] to rotation matrix [3, 3]."""
    w, x, y, z = quat.unbind(-1)
    mat = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    return mat


@dataclass
class EdgeAwareStrategy(DefaultStrategy):
    """Edge-Aware Densification Strategy (Improved-GS).

    Extends DefaultStrategy with edge-aware importance scoring, long-axis
    splitting, recovery-aware pruning, and budget-controlled densification.

    Usage:
        Call ``strategy.set_edge_maps(edge_maps)`` before training to provide
        pre-computed edge maps for all training views. The strategy will use
        these during densification steps.

    Args:
        budget: Maximum Gaussian count target. Default: 3_000_000.
        split_distance: Offset ratio for LAS (d = ratio * scale_max * 3).
            Default: 0.45.
        opacity_reduction: Child opacity multiplier after split. Default: 0.6.
        edge_n_cams: Number of cameras to sample for edge score computation.
            Use -1 for all cameras. Default: 10.
        rap_iterations: Iterations for Recovery-Aware Pruning (bottom 20%).
            Default: [300, 3300, 6300].
        rap_prune_ratio: Fraction of lowest-opacity Gaussians to prune at
            RAP iterations. Default: 0.2.
        edge_fallback_iter: After this iteration, fall back to gradient-only
            scoring. Default: 14500.
    """

    # Budget control
    budget: int = 3_000_000

    # Long-Axis Split
    split_distance: float = 0.45
    opacity_reduction: float = 0.6

    # Edge score
    edge_n_cams: int = 10

    # Recovery-Aware Pruning
    rap_iterations: List[int] = field(default_factory=lambda: [300, 3300, 6300])
    rap_prune_ratio: float = 0.2

    # Phase transition
    edge_fallback_iter: int = 14500

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale)
        # Edge score state (set by set_edge_scores or compute_edge_scores)
        state["edge_importance"] = None  # [N] or None
        return state

    def set_edge_scores(self, state: Dict[str, Any], scores: Tensor):
        """Set pre-computed per-Gaussian edge importance scores.

        Called from the training loop after running compute_edge_scores().

        Args:
            state: Strategy state dict.
            scores: [N] tensor of per-Gaussian importance scores.
        """
        state["edge_importance"] = scores

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Extended post-backward with edge-aware grow and recovery-aware pruning."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # Edge-aware grow with budget control
            n_dupli, n_split = self._grow_gs_edge_aware(
                params, optimizers, state, step
            )
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split "
                    f"(edge-aware). Now having {len(params['means'])} GSs."
                )

            # Standard opacity pruning (stop near end of densification)
            if step < self.refine_stop_iter - 100:
                n_prune = self._prune_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(params['means'])} GSs."
                    )

            # Reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0 and state.get("radii") is not None:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        # Reset opacity periodically
        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

        # Recovery-Aware Pruning at specific iterations
        if step in self.rap_iterations:
            self._recovery_aware_prune(params, optimizers, state, step)

    @torch.no_grad()
    def _grow_gs_edge_aware(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        """Edge-aware densification with budget-controlled long-axis split."""
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device
        N = len(params["means"])

        # Gradient threshold
        min_grad = self.grow_grad2d

        # Get importance scores: edge-aware or gradient fallback
        edge_scores = state.get("edge_importance")
        if edge_scores is not None and edge_scores.shape[0] == N and step <= self.edge_fallback_iter:
            # Phase 1: Use edge-aware scores
            scores = edge_scores.clone()
        else:
            # Phase 2: Fall back to gradient-only scoring
            scores = grads.clone()
            # Relax threshold if under budget (official code: / 1.5)
            if N < self.budget and step > self.edge_fallback_iter:
                min_grad = min_grad / 1.5

        # Filter: only consider Gaussians with high enough gradient
        grad_qualifiers = grads >= min_grad
        scores[~grad_qualifiers] = 0.0

        # Budget: sqrt ramp schedule
        start_iter = self.refine_start_iter
        end_iter = self.refine_stop_iter - 500
        rate = (step - start_iter) / max(end_iter - start_iter, 1)
        if rate >= 1.0:
            current_budget = self.budget
        else:
            current_budget = int(math.sqrt(max(rate, 0.0)) * self.budget)

        # How many to add
        all_budget = current_budget - N
        if all_budget <= 0:
            return 0, 0

        # Count qualifying Gaussians
        n_qualifying = (scores > 0).sum().item()
        if n_qualifying == 0:
            return 0, 0

        # Cap budget to qualifying count
        split_budget = min(all_budget, n_qualifying)

        # Multinomial sampling weighted by importance
        # Official code: torch.multinomial(importance, budget, replacement=False)
        sample_weights = scores.float()
        sample_weights[sample_weights <= 0] = 0.0
        if sample_weights.sum() <= 0:
            return 0, 0

        sampled_indices = torch.multinomial(
            sample_weights, min(split_budget, n_qualifying), replacement=False
        )
        split_mask = torch.zeros(N, dtype=torch.bool, device=device)
        split_mask[sampled_indices] = True

        # Separate small (duplicate) vs large (split)
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = split_mask & is_small
        is_split = split_mask & ~is_small

        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()

        # Duplicate small Gaussians
        if n_dupli > 0:
            duplicate(
                params=params, optimizers=optimizers, state=state, mask=is_dupli
            )

        # Extend split mask for new duplicates
        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)]
        )

        # Long-Axis Split for large Gaussians
        if n_split > 0:
            self._long_axis_split(params, optimizers, state, is_split)

        return n_dupli, n_split

    @torch.no_grad()
    def _long_axis_split(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        mask: Tensor,
    ):
        """Long-Axis Split (LAS): split along longest principal axis.

        Geometry from official code (scene/gaussian_model.py:long_axis_split):
        - d = split_distance * scale_max * 3  (offset along principal axis)
        - rate_w = 1 - split_distance = 0.55
        - rate_h = sqrt(1 - split_distance^2) ≈ 0.8929
        - Child principal axis scale = rate_w * parent
        - Child other axes = rate_h * parent
        - Child opacity = opacity_reduction * parent (0.6x)
        """
        device = mask.device
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        if len(sel) == 0:
            return

        scales = torch.exp(params["scales"][sel])  # [M, 3]
        quats = F.normalize(params["quats"][sel], dim=-1)
        rotmats = _normalized_quat_to_rotmat(quats)  # [M, 3, 3]

        # Find longest axis and its scale
        max_values, max_indices = scales.max(dim=-1, keepdim=True)  # [M, 1]

        # Offset samples along principal axis only
        # Official code: mask = zeros_like(stds).scatter(1, max_indices, True)
        #                samples = stds * mask * 3
        #                x1 = samples * rate  (= scale_max * 3 * split_distance)
        axis_mask = torch.zeros_like(scales, dtype=torch.bool).scatter(
            1, max_indices, True
        )
        samples = scales * axis_mask.float() * 3  # [M, 3], non-zero only in max axis

        rate = self.split_distance  # 0.45
        rate_w = 1.0 - rate  # 0.55
        rate_h = math.sqrt(1.0 - rate * rate)  # 0.8929

        x1 = samples * rate  # [M, 3], offset in local space

        # Create symmetric offsets: [+offset, -offset]
        x1_both = torch.cat([x1, -x1], dim=0)  # [2M, 3]

        # Rotate to world space
        rots = rotmats.repeat(2, 1, 1)  # [2M, 3, 3]
        offsets_world = torch.bmm(rots, x1_both.unsqueeze(-1)).squeeze(-1)  # [2M, 3]

        # New positions
        parent_means = params["means"][sel].repeat(2, 1)  # [2M, 3]
        new_xyz = parent_means + offsets_world  # [2M, 3]

        # New scales: shrink principal axis by rate_w, others by rate_h
        # Official: new_scaling = stds.scatter(1, max_indices, max_values * rate_w / rate_h) * rate_h
        # This gives: principal_axis = max_values * rate_w, other_axes = original * rate_h
        adjusted_scales = scales.scatter(
            1, max_indices, max_values * rate_w / rate_h
        )
        new_scales_linear = (adjusted_scales * rate_h).repeat(2, 1)  # [2M, 3]
        new_scales_log = torch.log(new_scales_linear.clamp(min=1e-8))

        # New opacity: reduced
        parent_opacity = torch.sigmoid(params["opacities"][sel])
        child_opacity = parent_opacity * self.opacity_reduction
        new_opacity_logit = torch.logit(child_opacity.clamp(0.01, 0.99)).repeat(
            2, *([1] * (parent_opacity.dim() - 1))
        )

        def param_fn(name: str, p: Tensor) -> Tensor:
            repeats = [2] + [1] * (p.dim() - 1)
            if name == "means":
                p_split = new_xyz
            elif name == "scales":
                p_split = new_scales_log
            elif name == "opacities":
                p_split = new_opacity_logit
            else:
                p_split = p[sel].repeat(repeats)
            p_new = torch.cat([p[rest], p_split])
            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
            return torch.cat([v[rest], v_split])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

        # Update running state tensors
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                repeats = [2] + [1] * (v.dim() - 1)
                v_new = v[sel].repeat(repeats)
                state[k] = torch.cat((v[rest], v_new))

    @torch.no_grad()
    def _recovery_aware_prune(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ):
        """Recovery-Aware Pruning: remove bottom rap_prune_ratio by opacity.

        Official code: only_prune(0.2, percentile=True) at iter 300, 3300, 6300.
        """
        N = len(params["means"])
        opacities = torch.sigmoid(params["opacities"].flatten())
        n_prune = int(N * self.rap_prune_ratio)

        if n_prune <= 0 or n_prune >= N:
            return

        _, bottom_idx = opacities.topk(n_prune, largest=False)
        prune_mask = torch.zeros(N, dtype=torch.bool, device=opacities.device)
        prune_mask[bottom_idx] = True

        n_actual = prune_mask.sum().item()
        if n_actual > 0:
            remove(
                params=params, optimizers=optimizers, state=state, mask=prune_mask
            )
            if self.verbose:
                print(
                    f"Step {step}: RAP pruned {n_actual} GSs (bottom {self.rap_prune_ratio*100:.0f}% opacity). "
                    f"Now having {len(params['means'])} GSs."
                )
