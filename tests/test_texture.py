"""Test texture rasterization (LGTM-inspired).

Run inside Docker with CUDA:
    pip install -e . && python tests/test_texture.py
"""
import torch
import torch.nn.functional as F


def test_texture_forward():
    """Test that texture rasterization produces different output than SH."""
    from gsplat import rasterization

    torch.manual_seed(42)
    N, C = 100, 1
    device = "cuda"

    means = torch.randn(N, 3, device=device) * 2
    means[:, 2] += 5  # push in front of camera
    quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device) * 0.5
    opacities = torch.sigmoid(torch.randn(N, device=device))
    colors = torch.rand(N, 1, 3, device=device)  # SH0
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor([[500, 0, 160], [0, 500, 120], [0, 0, 1]],
                       dtype=torch.float32, device=device).unsqueeze(0)

    W, H = 320, 240
    T = 4  # 4x4 texture

    # Render without texture
    out_sh, alpha_sh, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, W, H,
        sh_degree=0, packed=False,
    )

    # Create texture: random 4x4 per-Gaussian
    # Layout: [N, C*T*T] where C=3 (RGB), stored as [N, 3*16] = [N, 48]
    textures = torch.rand(N, 3 * T * T, device=device, requires_grad=True)

    # Render with texture
    out_tex, alpha_tex, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, W, H,
        sh_degree=0, packed=False,
        textures=textures, texture_size=T,
    )

    print(f"SH output shape: {out_sh.shape}, range: [{out_sh.min():.3f}, {out_sh.max():.3f}]")
    print(f"Tex output shape: {out_tex.shape}, range: [{out_tex.min():.3f}, {out_tex.max():.3f}]")
    print(f"Outputs differ: {not torch.allclose(out_sh, out_tex)}")
    assert not torch.allclose(out_sh, out_tex), "Texture output should differ from SH"
    print("✅ Forward test passed")


def test_texture_backward():
    """Test that gradients flow to texture parameters."""
    from gsplat import rasterization

    torch.manual_seed(42)
    N = 50
    device = "cuda"
    T = 4

    means = torch.randn(N, 3, device=device) * 2
    means[:, 2] += 5
    quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device) * 0.5
    opacities = torch.sigmoid(torch.randn(N, device=device))
    colors = torch.rand(N, 1, 3, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor([[500, 0, 80], [0, 500, 60], [0, 0, 1]],
                       dtype=torch.float32, device=device).unsqueeze(0)

    W, H = 160, 120
    textures = torch.rand(N, 3 * T * T, device=device, requires_grad=True)

    # Forward
    out, alpha, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, W, H,
        sh_degree=0, packed=False,
        textures=textures, texture_size=T,
    )

    # Backward
    target = torch.rand_like(out)
    loss = F.l1_loss(out, target)
    loss.backward()

    print(f"Texture grad: {textures.grad is not None}")
    if textures.grad is not None:
        print(f"  shape: {textures.grad.shape}")
        print(f"  norm: {textures.grad.norm():.6f}")
        print(f"  nonzero: {(textures.grad != 0).sum()} / {textures.grad.numel()}")
    assert textures.grad is not None, "Texture should receive gradients"
    assert textures.grad.norm() > 0, "Texture gradient should be non-zero"
    print("✅ Backward test passed")


if __name__ == "__main__":
    test_texture_forward()
    print()
    test_texture_backward()
    print("\n🎉 All texture tests passed!")
