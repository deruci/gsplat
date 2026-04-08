/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "MacroUtils.h"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_fwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2 *__restrict__ means2d,         // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [I, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [I, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [I, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [I, CDIM]
    const bool *__restrict__ masks,           // [I, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // Texture support (LGTM-inspired): per-Gaussian TxT color grid
    const scalar_t *__restrict__ textures,    // [N, T*T, CDIM] or nullptr
    const uint32_t texture_size,              // T (e.g. 2, 4, 8). 0 = no texture
    scalar_t
        *__restrict__ render_colors, // [I, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas, // [I, image_height, image_width, 1]
    int32_t *__restrict__ last_ids        // [I, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_colors += image_id * image_height * image_width * CDIM;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        if (inside) {
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id * CDIM + k] =
                    backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float alpha = min(MAX_ALPHA, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= TRANSMITTANCE_THRESHOLD) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            if (textures != nullptr && texture_size > 0) {
                // Bilinear texture lookup (LGTM/BBSplat-style)
                // Texture is RGB-only (3 channels). Extra channels (depth) use colors.
                const vec3 xy_opac_t = xy_opacity_batch[t];
                const vec3 con = conic_batch[t];
                float inv_radius = sqrtf(fmaxf(con.x, con.z));
                float radius = 1.0f / fmaxf(inv_radius, 1e-6f);
                float u = fminf(fmaxf((px - xy_opac_t.x) / radius, -1.0f), 1.0f);
                float v = fminf(fmaxf((py - xy_opac_t.y) / radius, -1.0f), 1.0f);
                float tx = ((u + 1.0f) * 0.5f) * (texture_size - 1);
                float ty = ((v + 1.0f) * 0.5f) * (texture_size - 1);
                int ix0 = min(max((int)floorf(tx), 0), (int)texture_size - 1);
                int iy0 = min(max((int)floorf(ty), 0), (int)texture_size - 1);
                int ix1 = min(ix0 + 1, (int)texture_size - 1);
                int iy1 = min(iy0 + 1, (int)texture_size - 1);
                float wx = tx - ix0;
                float wy = ty - iy0;
                int tex_area = texture_size * texture_size;
                int g_tex = packed ? g : (g % N);
                // Texture channels: min(3, CDIM) for RGB, rest from colors
                constexpr uint32_t TEX_CHANNELS = 3;
                int tex_offset = g_tex * tex_area * TEX_CHANNELS;
                // RGB from texture
                for (uint32_t k = 0; k < min(TEX_CHANNELS, CDIM); ++k) {
                    float c00 = textures[tex_offset + k * tex_area + iy0 * texture_size + ix0];
                    float c01 = textures[tex_offset + k * tex_area + iy0 * texture_size + ix1];
                    float c10 = textures[tex_offset + k * tex_area + iy1 * texture_size + ix0];
                    float c11 = textures[tex_offset + k * tex_area + iy1 * texture_size + ix1];
                    float sampled = c00 * (1-wx) * (1-wy) + c01 * wx * (1-wy) +
                                    c10 * (1-wx) * wy + c11 * wx * wy;
                    pix_out[k] += sampled * vis;
                }
                // Extra channels (depth etc.) from colors
                const float *c_ptr = colors + g * CDIM;
                for (uint32_t k = TEX_CHANNELS; k < CDIM; ++k) {
                    pix_out[k] += c_ptr[k] * vis;
                }
            } else {
                const float *c_ptr = colors + g * CDIM;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    pix_out[k] += c_ptr[k] * vis;
                }
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // texture support (LGTM-inspired)
    const at::optional<at::Tensor> textures,    // [N, T*T*channels] or nullopt
    const uint32_t texture_size,                // T (2, 4, 8). 0 = no texture
    // outputs
    at::Tensor renders, // [..., image_height, image_width, channels]
    at::Tensor alphas,  // [..., image_height, image_width]
    at::Tensor last_ids // [..., image_height, image_width]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dgs_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            textures.has_value() ? textures.value().data_ptr<float>()
                                 : nullptr,
            texture_size,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_fwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::optional<at::Tensor> textures,                               \
        uint32_t texture_size,                                                 \
        at::Tensor renders,                                                    \
        at::Tensor alphas,                                                     \
        at::Tensor last_ids                                                    \
    );

GSPLAT_FOR_EACH(__INS__, GSPLAT_NUM_CHANNELS)
#undef __INS__

} // namespace gsplat

#endif
