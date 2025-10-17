#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

__device__ void transpose_buffer(
    float *dst, const uint32_t dst_width,
    const uint32_t num_items, float4 *tmp4
) {
    // Vectorize dst
    float4 *dst4 = reinterpret_cast<float4*>(dst);
    const uint32_t dst_vec_width = dst_width / 4;
    // Vector load entire buffer into registers
    for (uint32_t idx = threadIdx.x; idx < num_items / 4; idx += blockDim.x) {
        const uint32_t i = idx / dst_vec_width;
        const uint32_t j = idx % dst_vec_width;
        tmp4[idx / blockDim.x] = dst4[i * dst_vec_width + j];
    }
    __syncthreads();
    // Scalar store buffer back to SMEM transposed
    const uint32_t dst_height = num_items / dst_width + 1;
    for (uint32_t idx = threadIdx.x; idx < num_items / 4; idx += blockDim.x) {
        const uint32_t i = idx / dst_vec_width;
        const uint32_t j = (idx % dst_vec_width) * 4;
        dst[(j + 0) * dst_height + i] = tmp4[idx / blockDim.x].x;
        dst[(j + 1) * dst_height + i] = tmp4[idx / blockDim.x].y;
        dst[(j + 2) * dst_height + i] = tmp4[idx / blockDim.x].z;
        dst[(j + 3) * dst_height + i] = tmp4[idx / blockDim.x].w;
    }
    __syncthreads();
}
__device__ void load_buffer(
    const float *src, const uint32_t src_width,
    float *dst, const uint32_t dst_width,
    const uint32_t num_items
) {
    const float4 *src4 = reinterpret_cast<const float4*>(src);
    float4 *dst4 = reinterpret_cast<float4*>(dst);
    const uint32_t src_vec_width = src_width / 4;
    const uint32_t dst_vec_width = dst_width / 4;
    for (uint32_t idx = threadIdx.x; idx < num_items / 4; idx += blockDim.x) {
        // Compute 2D index in terms of float4s
        const uint32_t i = idx / dst_vec_width;
        const uint32_t j = idx % dst_vec_width;
        // Vectorized load and store
        float4 val = src4[i * src_vec_width + j];
        dst4[i * dst_vec_width + j] = val;
    }
    __syncthreads();
}
__device__ void load_buffer_async(
    float const *src, const uint32_t src_width, 
    float *dst, const uint32_t dst_width,
    const uint32_t num_items
) {
    for (uint32_t idx = threadIdx.x; idx < num_items / 4; idx += blockDim.x) {
        // Get index to copy
        const uint32_t flat_idx = idx * 4;
        const uint32_t i = flat_idx / dst_width;
        const uint32_t j = flat_idx % dst_width;
        // Copy mem over
        __pipeline_memcpy_async(&dst[i * dst_width + j], &src[i * src_width + j], sizeof(float4), 0);
    }
    __pipeline_commit();
}

// OPTIONAL: Uncomment this block to include your kernel implementation
// from Lab 5 for easy comparison.

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k (Baseline from Lab 5)

#define HAS_LAB_5_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab 5 kernel!

namespace matmul_improved_reduce {

template <
    uint32_t TH, uint32_t TW, uint32_t TD, // SM tile size
    uint32_t AH, uint32_t AW, // A in SMEM tile size
    uint32_t BH, uint32_t BW, // B in SMEM tile size
    uint32_t CH, uint32_t CW  // C in registers tile size
>
__device__ void matmul_tile(
    const uint32_t size_i, const uint32_t size_j, const uint32_t size_k, // Matrix dimensions
    float const *a, float const *b, float *reduce_c, // Matrices in GMEM
    float *local_a, float *local_b, float *local_a_stage, float *local_b_stage // Matrices in SMEM
) {
    // Each thread gets a c_ij block in the tile
    const uint32_t start_i = (threadIdx.x / (TW / CW)) * CH;
    const uint32_t start_j = (threadIdx.x % (TW / CW)) * CW;

    // Keep c_ij's in local registers
    float local_c_ij[CH * CW] = {0.0f};

    // Load compute buffer
    load_buffer(a, size_k, local_a, AW, AH * AW);
    load_buffer(b, size_j, local_b, BW, BH * BW);

    // local_a transpose tmp buffer
    constexpr uint32_t num_threads = (TH * TW) / (CH * CW);
    constexpr uint32_t transpose_size = ((AH * AW / num_threads) == 0 ? 1 : (AH * AW / num_threads)) * 4;
    float tmp_buffer[transpose_size];
    float4 *tmp4 = reinterpret_cast<float4*>(tmp_buffer);

    // Iterate over local buffers
    for (uint32_t idx = 0; idx < TD / AW - 1; ++idx) {
        // Move global buffers
        a += AW;
        b += BH * size_j;

        // Load stage buffer
        load_buffer_async(a, size_k, local_a_stage, AW, AH * AW);
        load_buffer_async(b, size_j, local_b_stage, BW, BH * BW);

        // Transpose local_a
        transpose_buffer(local_a, AW, AH * AW, tmp4);

        // Iterate over a_i, b_k
        #pragma unroll
        for (uint32_t k = 0; k < AW; ++k) {
            const uint32_t a_i_offset = k * (AH + 1) + start_i;
            const uint32_t b_k_offset = k * BW + start_j;
            #pragma unroll
            for (uint32_t j = 0; j < CW; ++j) {
                const float tmp = local_b[b_k_offset + j];
                #pragma unroll
                for (uint32_t i = 0; i < CH; ++i) {
                    local_c_ij[i * CW + j] += local_a[a_i_offset + i] * tmp;
                }
            }
        }

        // Swap double buffers
        __pipeline_wait_prior(0);
        __syncthreads();
        std::swap(local_a, local_a_stage);
        std::swap(local_b, local_b_stage);
    }
    // Process last block
    transpose_buffer(local_a, AW, AH * AW, tmp4);
    #pragma unroll
    for (uint32_t k = 0; k < AW; ++k) {
        const uint32_t a_i_offset = k * (AH + 1) + start_i;
        const uint32_t b_k_offset = k * BW + start_j;
        #pragma unroll
        for (uint32_t j = 0; j < CW; ++j) {
            const float tmp = local_b[b_k_offset + j];
            #pragma unroll
            for (uint32_t i = 0; i < CH; ++i) {
                local_c_ij[i * CW + j] += local_a[a_i_offset + i] * tmp;
            }
        }
    }

    // Write back to main memory at the end
    const uint32_t reduce_j_offset = size_k / TD;
    const uint32_t reduce_size_j = size_j * reduce_j_offset;
    #pragma unroll
    for (uint32_t c_ij_idx = 0; c_ij_idx < CH * CW; ++c_ij_idx) {
        const uint32_t i = start_i + c_ij_idx / CW;
        const uint32_t j = start_j + c_ij_idx % CW;
        reduce_c[i * reduce_size_j + j * reduce_j_offset] = local_c_ij[c_ij_idx];
    }

    // Make sure the whole tile is done before moving on
    __syncthreads();
}

template <
    uint32_t TH, uint32_t TW, uint32_t TD, // SM tile size
    uint32_t AH, uint32_t AW, // A in SMEM tile size
    uint32_t BH, uint32_t BW, // B in SMEM tile size
    uint32_t CH, uint32_t CW  // C in registers tile size
>
__launch_bounds__((TH*TW)/(CH*CW))
__global__ void matmul_improved(
    const int32_t size_i, const int32_t size_j, const int32_t size_k,
    float const *a,  float const *b, float *reduce_c
) {
    // Grid dimensions
    const uint32_t tiles_per_i = size_i / TH;
    const uint32_t tiles_per_j = size_j / TW;
    const uint32_t tiles_per_k = size_k / TD;

    // Setup the block's SRAM
    extern __shared__ float sram[];
    // Split the SRAM into a double buffer
    constexpr uint32_t a_double_buffer_size = (AH + 1) * AW;
    constexpr uint32_t b_double_buffer_size = BH * BW;
    float *local_a = sram;
    float *local_a_stage = local_a + a_double_buffer_size;
    float *local_b = local_a_stage + a_double_buffer_size;
    float *local_b_stage = local_b + b_double_buffer_size;

    // Conversions between c and reduce_c
    const uint32_t reduce_j_offset = size_k / TD;
    const uint32_t reduce_size_j = size_j * reduce_j_offset;

    // Iterate over tiles
    for (uint32_t idx = blockIdx.x; idx < tiles_per_i * tiles_per_j * tiles_per_k; idx += gridDim.x) {
        // Tile indices
        const uint32_t tile_i = idx / (tiles_per_j * tiles_per_k);
        const uint32_t tile_j = (idx % (tiles_per_j * tiles_per_k)) / tiles_per_k;
        const uint32_t tile_k = (idx % (tiles_per_j * tiles_per_k)) % tiles_per_k;

        // Move buffers
        float const *tile_a = a + tile_i * TH * size_k + tile_k * TD;
        float const *tile_b = b + tile_k * TD * size_j + tile_j * TW;
        float *tile_reduce_c = reduce_c + tile_i * TH * reduce_size_j + tile_j * TW * reduce_j_offset + tile_k;

        matmul_tile<TH, TW, TD, AH, AW, BH, BW, CH, CW>(
            size_i, size_j, size_k,
            tile_a, tile_b, tile_reduce_c,
            local_a, local_b,
            local_a_stage, local_b_stage
        );
    }
}

template <uint32_t TD, uint32_t V_SIZE>
__launch_bounds__(1024)
__global__ void matmul_reduce(
    const uint32_t size_i, const uint32_t size_j, const uint32_t size_k,
    float *c, const float *reduce_c
) {
    const uint32_t reduce_j_offset = size_k / TD;
    const uint32_t reduce_size_j   = size_j * reduce_j_offset;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_i * size_j; idx += gridDim.x * blockDim.x) {
        // Load a c_ij array from GMEM
        const uint32_t i = idx / size_j;
        const uint32_t c_j = idx % size_j;
        const uint32_t reduce_c_j = c_j * reduce_j_offset;
        const float *base = &reduce_c[i * reduce_size_j + reduce_c_j];
        float c_ij = 0.0f;
        // Select the vector size
        if constexpr (V_SIZE == 1) {
            #pragma unroll
            for (uint32_t k = 0; k < reduce_j_offset; ++k)
                c_ij += base[k];
        }
        else if constexpr (V_SIZE == 2) {
            const float2 *vptr = reinterpret_cast<const float2*>(base);
            const uint32_t vecCount = reduce_j_offset / 2;
            #pragma unroll
            for (uint32_t vk = 0; vk < vecCount; ++vk) {
                float2 v = vptr[vk];
                c_ij += v.x + v.y;
            }
        }
        else if constexpr (V_SIZE == 4) {
            const float4 *vptr = reinterpret_cast<const float4*>(base);
            const uint32_t vecCount = reduce_j_offset / 4;
            #pragma unroll
            for (uint32_t vk = 0; vk < vecCount; ++vk) {
                float4 v = vptr[vk];
                c_ij += v.x + v.y + v.z + v.w;
            }
        }
        // Write back at the end
        c[i * size_j + c_j] = c_ij;
    }
}

template <uint32_t TD, uint32_t KW>
__launch_bounds__(1024)
__global__ void matmul_reduce_vec(
    const uint32_t size_i, const uint32_t size_j, const uint32_t size_k,
    float *c, const float *reduce_c
) {
    // Vectorize
    const uint32_t size_vj = size_j / 4;
    float4 *c4 = reinterpret_cast<float4*>(c);
    const float4 *reduce_c4 = reinterpret_cast<const float4*>(reduce_c);
    // Loop over vectors of c_ij's
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_i * size_vj; idx += gridDim.x * blockDim.x) {
        const uint32_t vi = idx / size_vj;
        const uint32_t vj = idx % size_vj;
        const uint32_t offset = (vi * size_vj + vj) * KW;
        // Load the c_ij vector into registers
        float4 tmp4[KW];
        #pragma unroll
        for (uint32_t vk = 0; vk < KW; ++vk) {
            tmp4[vk] = reduce_c4[offset + vk];
        }
        // Sum up for each c_ij
        float c_ij[4] = {0.0f};
        float *tmp = reinterpret_cast<float*>(tmp4);
        #pragma unroll
        for (uint32_t v = 0; v < 4; ++v) {
            for (uint32_t k = 0; k < KW; ++k) {
                c_ij[v] += tmp[v * KW + k];
            }
        }
        // Write back at the end
        float4 *c_ij4 = reinterpret_cast<float4*>(c_ij);
        c4[vi * size_vj + vj] = c_ij4[0];
    }
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    // return (size_t)size_i * (size_t)size_j * (size_t)(size_k / 512) * sizeof(float); // Uses the minimum TD
    return 0;
}

template <
    uint32_t B, uint32_t W, uint32_t T, // Kernel dimensions
    uint32_t TH, uint32_t TW, uint32_t TD, uint32_t AH, uint32_t AW, uint32_t BH, uint32_t BW, uint32_t CH, uint32_t CW,  // Work dimensions
    uint32_t V_SIZE = 1 // Reduce dimensions
>
void launch_specialized_kernel(
    const int32_t size_i, const int32_t size_j, const int32_t size_k,
    float const *a, float const *b, float *c, void *workspace) {
    // Set dynamic shared memory size
    constexpr int shmem_size_bytes = ((AH + 1) * AW + BH * BW) * 2 * sizeof(float);
    cudaFuncSetAttribute(
        matmul_improved<TH, TW, TD, AH, AW, BH, BW, CH, CW>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes
    );
    if (TD == size_k) {
        // No need to reduce
        matmul_improved<TH, TW, TD, AH, AW, BH, BW, CH, CW><<<B, W*T, shmem_size_bytes>>>(
            size_i, size_j, size_k, a, b, c
        );
        return;
    } else {
        matmul_improved<TH, TW, TD, AH, AW, BH, BW, CH, CW><<<B, W*T, shmem_size_bytes>>>(
            size_i, size_j, size_k, a, b, (float*)workspace
        );
        matmul_reduce<TD, V_SIZE><<<B, 32*T>>>(
            size_i, size_j, size_k,
            c, (float*)workspace
        );
        // matmul_reduce_vec<TD, V_SIZE><<<B, 32*T>>>(
        //     size_i, size_j, size_k,
        //     c, (float*)workspace
        // );
    }
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    // Thread block dimensions
    constexpr uint32_t B = 48;
    constexpr uint32_t W = 8; // Tuning parameter
    constexpr uint32_t T = 32;

    // Tile dimensions
    constexpr uint32_t CH = 8; // Tuning parameter
    constexpr uint32_t CW = CH;
    constexpr uint32_t TH = W * CH;
    constexpr uint32_t TW = T * CW;
    constexpr uint32_t TD = 3072; // Constant for these problem sizes

    // SMEM dimensions
    constexpr uint32_t AH = TH;
    constexpr uint32_t AW = 1 * 32; // Tuning parameter
    constexpr uint32_t BH = AW;
    constexpr uint32_t BW = TW;

    if (size_i == 3072 || size_i == 2048 || size_i == 1024 || size_i == 512 || size_i == 256) {
        launch_specialized_kernel<
            B, W, T,
            TH, TW, TD, // 64, 256, 3072
            AH, AW, // 64, 32
            BH, BW, // 32, 256
            CH, CW // 8, 8
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 128) {
        launch_specialized_kernel<
            B, W, T,
            TH/2, TW, TD, // 32, 256, 3072
            AH/2, AW, // 32, 32
            BH, BW, // 32, 256
            CH/2, CW // 4, 8
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 64) {
        launch_specialized_kernel<
            B, W, T,
            TH/2, TW/2, TD, // 32, 128, 3072
            AH/2, 2 * AW, // 32, 64
            2 * BH, BW/2, // 64, 128
            CH/2, CW/2 // 4, 4
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 32) { 
        launch_specialized_kernel<
            B, W, T,
            TH/4, TW/2, TD, // 16, 128, 3072
            AH/4, 2 * AW, // 16, 64
            2 * BH, BW/2, // 64, 128
            CH/4, CW/2 // 2, 4
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 16) {
        launch_specialized_kernel<
            B, W, T,
            TH/8, TW/2, TD, // 8, 128, 3072
            AH/8, AW, // 8, 32
            BH, BW/2, // 32, 128
            CH/8, CW/2 // 1, 4
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 1) { 
        launch_specialized_kernel<
            B, W/4, T, // 48, 2, 32
            TH/64, TW/4, TD, // 1, 64, 3072
            AH/64, 2 * AW, // 1, 64
            2 * BH, BW/4, // 64, 64
            CH/8, CW/8 // 1, 1
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else {
        return;
    }
}

} // namespace matmul_improved_reduce

////////////////////////////////////////////////////////////////////////////////
// Tensor Core GPU Implementation

namespace matmul_tensor {

// Helper functions
template <uint32_t LW, uint32_t GW>
__device__ uint32_t local_idx_to_global(uint32_t idx) {
    return (idx / LW) * GW + (idx % LW);
}
template <uint32_t LW>
__device__ uint32_t local_idx_to_global(uint32_t idx, uint32_t GW) {
    return (idx / LW) * GW + (idx % LW);
}

// Functions to rearrange A and B for vector loads
template <uint32_t NW, uint32_t SMEM_TH, uint32_t SMEM_TW>
__device__ void rearrange_a_m16n8k8(float *a) {
    // Thread block info
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;

    // Warp grid dimensions
    constexpr uint32_t wt_per_i = SMEM_TH / 16;
    constexpr uint32_t wt_per_j = SMEM_TW / 8;
    constexpr uint32_t wt_per_w = wt_per_i * wt_per_j / NW;

    // Iterate over warp tiles
    for (uint32_t idx = 0; idx < wt_per_w; ++idx) {
        // Warp tile indices
        const uint32_t warp_idx = warp + idx * NW;
        const uint32_t wt_i = warp_idx / wt_per_j;
        const uint32_t wt_j = warp_idx % wt_per_j;

        // Move buffer
        float *wa = a + wt_i * 16 * SMEM_TW + wt_j * 8;

        // Scalar load
        const uint32_t a_idx_x = local_idx_to_global<8, SMEM_TW>((thread % 4) + (thread / 4) * 8);
        const uint32_t a_idx_y = local_idx_to_global<8, SMEM_TW>((thread % 4) + (thread / 4) * 8 + 64);
        const uint32_t a_idx_z = local_idx_to_global<8, SMEM_TW>((thread % 4) + (thread / 4) * 8 + 4);
        const uint32_t a_idx_w = local_idx_to_global<8, SMEM_TW>((thread % 4) + (thread / 4) * 8 + 68);
        float4 A = {wa[a_idx_x], wa[a_idx_y], wa[a_idx_z], wa[a_idx_w]};
        __syncthreads();

        // Vector store (threads in a warp are synchronized so no explicit sync needed)
        const uint32_t i = thread / (SMEM_TW / 4);
        const uint32_t j = (thread % (SMEM_TW / 4)) * 4;
        wa += i * (SMEM_TW + 1) + j;
        float4 *wa4 = reinterpret_cast<float4*>(wa);
        *wa4 = A;
    }
}
template <uint32_t NW, uint32_t SMEM_TH, uint32_t SMEM_TW>
__device__ void rearrange_b_m16n8k8(float *b) {
    // Thread block info
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;

    // Warp grid dimensions
    constexpr uint32_t wt_per_i = SMEM_TH / 8;
    constexpr uint32_t wt_per_j = SMEM_TW / 8;
    constexpr uint32_t wt_per_w = wt_per_i * wt_per_j / NW;

    // Iterate over warp tiles
    for (uint32_t idx = 0; idx < wt_per_w; ++idx) {
        // Warp tile indices
        const uint32_t warp_idx = warp + idx * NW;
        const uint32_t wt_i = warp_idx / wt_per_j;
        const uint32_t wt_j = warp_idx % wt_per_j;

        // Move buffer
        float *wb = b + wt_i * 8 * SMEM_TW + wt_j * 8;

        // Scalar load
        const uint32_t b_idx_x = local_idx_to_global<8, SMEM_TW>((thread % 4) * 8 + (thread / 4));
        const uint32_t b_idx_y = local_idx_to_global<8, SMEM_TW>((thread % 4) * 8 + (thread / 4) + 32);
        float2 B = {wb[b_idx_x], wb[b_idx_y]};
        __syncthreads();

        // Vector store (threads in a warp are synchronized so no explicit sync needed)
        const uint32_t i = thread / (SMEM_TW / 2);
        const uint32_t j = (thread % (SMEM_TW / 2)) * 2;
        wb += i * (SMEM_TW + 1) + j;
        float2 *wb2 = reinterpret_cast<float2*>(wb);
        *wb2 = B;
    }
}

// Tensor core functions
__device__ void mma_16x8x8(float4 A, float2 B, float4 *C) {
    // Convert float registers to int registers
    uint32_t ax = __float_as_uint(A.x);
    uint32_t ay = __float_as_uint(A.y);
    uint32_t az = __float_as_uint(A.z);
    uint32_t aw = __float_as_uint(A.w);
    uint32_t bx = __float_as_uint(B.x);
    uint32_t by = __float_as_uint(B.y);
    uint32_t cx = __float_as_uint((*C).x);
    uint32_t cy = __float_as_uint((*C).y);
    uint32_t cz = __float_as_uint((*C).z);
    uint32_t cw = __float_as_uint((*C).w);

    // Call tensor core instruction using PTX
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3},  /* D matrix */ "
        "{%4, %5, %6, %7},  /* A matrix */ "
        "{%8, %9},          /* B matrix */ "
        "{%0, %1, %2, %3};" /* C matrix */
        // Outputs (read-write)
        : "+r"(cx), "+r"(cy), "+r"(cz), "+r"(cw)
        // Inputs (read-only)
        : "r"(ax), "r"(ay), "r"(az), "r"(aw), "r"(bx), "r"(by)
    );

    // Convert back and write back result to C rearranged
    (*C).x = __uint_as_float(cx);
    (*C).y = __uint_as_float(cy);
    (*C).z = __uint_as_float(cz);
    (*C).w = __uint_as_float(cw);
}

template <
    uint32_t NW, // Thread block size
    uint32_t SM_TH, uint32_t SM_TW, uint32_t SM_TD, // SM tile size
    uint32_t SMEM_TD, // SMEM tile size
    uint32_t W_TH, uint32_t W_TW // Warp tile size
>
__device__ void matmul_tile(
    const uint32_t size_i, const uint32_t size_j, const uint32_t size_k, // Matrix dimensions
    float const *a, float const *b, float *c, // Matrices in GMEM
    float *local_a, float *local_b, float *local_a_stage, float *local_b_stage // Matrices in SMEM
) {
    // Warp grid dimensions
    constexpr uint32_t wt_per_i = SM_TH / W_TH;
    constexpr uint32_t wt_per_j = SM_TW / W_TW;
    constexpr uint32_t wt_per_w = wt_per_i * wt_per_j / NW;

    // Thread block info
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;

    // Accumulate c results in registers
    float4 local_c[wt_per_w] = {0.0f};

    // Sync load first buffer
    load_buffer(a, size_k, local_a, SMEM_TD, SM_TH * SMEM_TD);
    load_buffer(b, size_j, local_b, SM_TW, SMEM_TD * SM_TW);

    // Iterate over SMEM tiles
    for (uint32_t smem_idx = 0; smem_idx < SM_TD / SMEM_TD - 1; ++smem_idx) {
        // Move global buffers to next SMEM tile
        a += SMEM_TD;
        b += SMEM_TD * size_j;

        // Async load stage buffer
        load_buffer_async(a, size_k, local_a_stage, SMEM_TD, SM_TH * SMEM_TD);
        load_buffer_async(b, size_j, local_b_stage, SM_TW, SMEM_TD * SM_TW);

        // Rearrange local_a and local_b
        rearrange_a_m16n8k8<NW, SM_TH, SMEM_TD>(local_a);
        rearrange_b_m16n8k8<NW, SMEM_TD, SM_TW>(local_b);
        // Wait for every warp to finish since multiple warps will use the same warp tile
        __syncthreads();
        
        // Iterate along k dimension
        for (uint32_t k = 0; k < SMEM_TD / W_TW; ++k) {
            // Iterate over warp tiles
            for (uint32_t c_idx = 0; c_idx < wt_per_w; ++c_idx) {
                // Warp tile indices
                const uint32_t warp_idx = warp + c_idx * NW;
                const uint32_t wt_i = warp_idx / wt_per_j;
                const uint32_t wt_j = warp_idx % wt_per_j;

                // Move buffers to warp tile
                float *wa = local_a + wt_i * W_TH * (SMEM_TD + 1) + k * W_TW;
                float *wb = local_b + k * W_TW * (SM_TW + 1) + wt_j * W_TW;

                // Vector load A, B from SMEM
                const uint32_t ai = thread / (SMEM_TD / 4);
                const uint32_t aj = (thread % (SMEM_TD / 4)) * 4;
                wa += ai * (SMEM_TD + 1) + aj;
                float4 *wa4 = reinterpret_cast<float4*>(wa);
                float4 A = *wa4;

                const uint32_t bi = thread / (SM_TW / 2);
                const uint32_t bj = (thread % (SM_TW / 2)) * 2;
                wb += bi * (SM_TW + 1) + bj;
                float2 *wb2 = reinterpret_cast<float2*>(wb);
                float2 B = *wb2;

                // Call tensor core function
                mma_16x8x8(A, B, &local_c[c_idx]);
            }
        }

        // Swap double buffers
        __pipeline_wait_prior(0);
        __syncthreads();
        std::swap(local_a, local_a_stage);
        std::swap(local_b, local_b_stage);
    }
    // Process last block
    rearrange_a_m16n8k8<NW, SM_TH, SMEM_TD>(local_a);
    rearrange_b_m16n8k8<NW, SMEM_TD, SM_TW>(local_b);
    __syncthreads();
    for (uint32_t k = 0; k < SMEM_TD / W_TW; ++k) {
                for (uint32_t c_idx = 0; c_idx < wt_per_w; ++c_idx) {
            const uint32_t warp_idx = warp + c_idx * NW;
            const uint32_t wt_i = warp_idx / wt_per_j;
            const uint32_t wt_j = warp_idx % wt_per_j;
            float *wa = local_a + wt_i * W_TH * (SMEM_TD + 1) + k * W_TW;
            float *wb = local_b + k * W_TW * (SM_TW + 1) + wt_j * W_TW;
            const uint32_t ai = thread / (SMEM_TD / 4);
            const uint32_t aj = (thread % (SMEM_TD / 4)) * 4;
            wa += ai * (SMEM_TD + 1) + aj;
            float4 *wa4 = reinterpret_cast<float4*>(wa);
            float4 A = *wa4;
            const uint32_t bi = thread / (SM_TW / 2);
            const uint32_t bj = (thread % (SM_TW / 2)) * 2;
            wb += bi * (SM_TW + 1) + bj;
            float2 *wb2 = reinterpret_cast<float2*>(wb);
            float2 B = *wb2;
            mma_16x8x8(A, B, &local_c[c_idx]);
        }
    }

    // Write back to memory
    const uint32_t reduce_offset = size_k / SM_TD;
    const uint32_t reduce_size_j = size_j * reduce_offset;
    for (uint32_t c_idx = 0; c_idx < wt_per_w; ++c_idx) {
        // Warp tile indices
        const uint32_t warp_idx = warp + c_idx * NW;
        const uint32_t wt_i = warp_idx / wt_per_j;
        const uint32_t wt_j = warp_idx % wt_per_j;
        // Global offset
        const uint32_t start_i = wt_i * W_TH;
        const uint32_t start_j = wt_j * W_TW;
        // Local offset
        const uint32_t cx_idx = thread * 2;
        const uint32_t cy_idx = thread * 2 + 1;
        const uint32_t cz_idx = thread * 2 + 64;
        const uint32_t cw_idx = thread * 2 + 65;
        // Convert to i,j
        const uint32_t cx_i = start_i + cx_idx / W_TW;
        const uint32_t cx_j = start_j + cx_idx % W_TW;
        const uint32_t cy_i = start_i + cy_idx / W_TW;
        const uint32_t cy_j = start_j + cy_idx % W_TW;
        const uint32_t cz_i = start_i + cz_idx / W_TW;
        const uint32_t cz_j = start_j + cz_idx % W_TW;
        const uint32_t cw_i = start_i + cw_idx / W_TW;
        const uint32_t cw_j = start_j + cw_idx % W_TW;
        // Scalar store to memory
        c[cx_i * reduce_size_j + cx_j * reduce_offset] = local_c[c_idx].x;
        c[cy_i * reduce_size_j + cy_j * reduce_offset] = local_c[c_idx].y;
        c[cz_i * reduce_size_j + cz_j * reduce_offset] = local_c[c_idx].z;
        c[cw_i * reduce_size_j + cw_j * reduce_offset] = local_c[c_idx].w;
    }

    // Make sure the whole tile is done before moving on
    __syncthreads();
}

template <
    uint32_t NW, // Thread block size
    uint32_t SM_TH, uint32_t SM_TW, uint32_t SM_TD, // SM tile size
    uint32_t SMEM_TD, // SMEM tile size
    uint32_t W_TH, uint32_t W_TW // Warp tile size
>
__launch_bounds__(NW*32)
__global__ void matmul_tensor(
    const int32_t size_i, const int32_t size_j, const int32_t size_k,
    float const *a,  float const *b, float *reduce_c
) {
    // SM grid dimensions
    const uint32_t smt_per_i = size_i / SM_TH;
    const uint32_t smt_per_j = size_j / SM_TW;
    const uint32_t smt_per_k = size_k / SM_TD;

    // Setup the block's SMEM
    extern __shared__ float sram[];
    // Split the SMEM into a double buffer
    constexpr uint32_t a_double_buffer_size = SM_TH * (SMEM_TD + 1);
    constexpr uint32_t b_double_buffer_size = SMEM_TD * (SM_TW + 1);
    float *smemt_a = sram;
    float *smemt_a_stage = smemt_a + a_double_buffer_size;
    float *smemt_b = smemt_a_stage + a_double_buffer_size;
    float *smemt_b_stage = smemt_b + b_double_buffer_size;

    // Conversions between c and reduce_c
    const uint32_t reduce_offset = size_k / SM_TD;
    const uint32_t reduce_size_j = size_j * reduce_offset;

    // Iterate over SM tiles
    for (uint32_t sm_idx = blockIdx.x; sm_idx < smt_per_i * smt_per_j * smt_per_k; sm_idx += gridDim.x) {
        // SM tile indices
        const uint32_t smt_i = sm_idx / (smt_per_j * smt_per_k);
        const uint32_t smt_j = (sm_idx % (smt_per_j * smt_per_k)) / smt_per_k;
        const uint32_t smt_k = (sm_idx % (smt_per_j * smt_per_k)) % smt_per_k;

        // Move global buffers to SM tile
        const float *smt_a = a + smt_i * SM_TH * size_k + smt_k * SM_TD;
        const float *smt_b = b + smt_k * SM_TD * size_j + smt_j * SM_TW;
        float *smt_reduce_c = reduce_c + smt_i * SM_TH * reduce_size_j + smt_j * SM_TW * reduce_offset + smt_k;

        matmul_tile<NW, SM_TH, SM_TW, SM_TD, SMEM_TD, W_TH, W_TW>(
            size_i, size_j, size_k,
            smt_a, smt_b, smt_reduce_c,
            smemt_a, smemt_b,
            smemt_a_stage, smemt_b_stage
        );
    }
}

template <uint32_t SM_TD, uint32_t V_SIZE>
__launch_bounds__(1024)
__global__ void matmul_reduce(
    const uint32_t size_i, const uint32_t size_j, const uint32_t size_k,
    float *c, const float *reduce_c
) {
    const uint32_t reduce_j_offset = size_k / SM_TD;
    const uint32_t reduce_size_j   = size_j * reduce_j_offset;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_i * size_j; idx += gridDim.x * blockDim.x) {
        // Load a c_ij array from GMEM
        const uint32_t i = idx / size_j;
        const uint32_t c_j = idx % size_j;
        const uint32_t reduce_c_j = c_j * reduce_j_offset;
        const float *base = &reduce_c[i * reduce_size_j + reduce_c_j];
        float c_ij = 0.0f;
        // Select the vector size
        if constexpr (V_SIZE == 1) {
            #pragma unroll
            for (uint32_t k = 0; k < reduce_j_offset; ++k)
                c_ij += base[k];
        }
        else if constexpr (V_SIZE == 2) {
            const float2 *vptr = reinterpret_cast<const float2*>(base);
            const uint32_t vecCount = reduce_j_offset / 2;
            #pragma unroll
            for (uint32_t vk = 0; vk < vecCount; ++vk) {
                float2 v = vptr[vk];
                c_ij += v.x + v.y;
            }
        }
        else if constexpr (V_SIZE == 4) {
            const float4 *vptr = reinterpret_cast<const float4*>(base);
            const uint32_t vecCount = reduce_j_offset / 4;
            #pragma unroll
            for (uint32_t vk = 0; vk < vecCount; ++vk) {
                float4 v = vptr[vk];
                c_ij += v.x + v.y + v.z + v.w;
            }
        }
        // Write back at the end
        c[i * size_j + c_j] = c_ij;
    }
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    // return (size_t)size_i * (size_t)size_j * (size_t)(size_k / 512) * sizeof(float); // Uses the minimum TD
    return 0;
}

template <
    uint32_t B, uint32_t W, uint32_t T, // Kernel dimensions
    uint32_t SM_TH, uint32_t SM_TW, uint32_t SM_TD, uint32_t SMEM_TD, uint32_t W_TH, uint32_t W_TW, // Work dimensions
    uint32_t V_SIZE = 1 // Reduce dimensions
>
void launch_specialized_kernel(
    const int32_t size_i, const int32_t size_j, const int32_t size_k,
    float const *a, float const *b, float *c, void *workspace) {
    // Set dynamic shared memory size
    // Add 1 pad to avoid bank conflicts? Is it added to height or width?
    constexpr int shmem_size_bytes = (SM_TH * (SMEM_TD + 1) + SMEM_TD * (SM_TW + 1)) * 2 * sizeof(float);
    cudaFuncSetAttribute(
        matmul_tensor<W, SM_TH, SM_TW, SM_TD, SMEM_TD, W_TH, W_TW>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes
    );
    if (SM_TD == size_k) {
        // No need to reduce
        matmul_tensor<W, SM_TH, SM_TW, SM_TD, SMEM_TD, W_TH, W_TW><<<B, W*T, shmem_size_bytes>>>(
            size_i, size_j, size_k, a, b, c
        );
        return;
    } else {
        matmul_tensor<W, SM_TH, SM_TW, SM_TD, SMEM_TD, W_TH, W_TW><<<B, W*T, shmem_size_bytes>>>(
            size_i, size_j, size_k, a, b, (float*)workspace
        );
        matmul_reduce<SM_TD, V_SIZE><<<B, 32*T>>>(
            size_i, size_j, size_k,
            c, (float*)workspace
        );
        // matmul_reduce_vec<SM_TD, V_SIZE><<<B, 32*T>>>(
        //     size_i, size_j, size_k,
        //     c, (float*)workspace
        // );
    }
}

void launch_matmul_tensor(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    // Thread block dimensions
    constexpr uint32_t B = 48;
    constexpr uint32_t W = 8; // Tuning parameter
    constexpr uint32_t T = 32;

    // Warp tile dimensions
    constexpr uint32_t W_TH = 16;
    constexpr uint32_t W_TW = 8;

    // SM tile dimensions
    constexpr uint32_t SM_TH = W/2 * W_TH; // Tuning parameter
    constexpr uint32_t SM_TW = T * W_TW; // Tuning parameter
    constexpr uint32_t SM_TD = 3072; // Constant for these problem sizes

    // SMEM tile dimensions
    constexpr uint32_t SMEM_TD = 1 * 32; // Tuning parameter

    if (size_i == 3072 || size_i == 2048 || size_i == 1024 || size_i == 512 || size_i == 256) {
        launch_specialized_kernel<
            B, W, T,
            SM_TH, SM_TW, SM_TD, // 64, 256, 3072
            SMEM_TD, // 32
            W_TH, W_TW // 16, 8
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 128) {
        launch_specialized_kernel<
            B, W, T,
            SM_TH/2, SM_TW, SM_TD, // 32, 256, 3072
            SMEM_TD, // 32
            W_TH, W_TW // 16, 8
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 64) {
        launch_specialized_kernel<
            B, W, T,
            SM_TH/2, SM_TW/2, SM_TD, // 32, 128, 3072
            SMEM_TD, // 32
            W_TH, W_TW // 16, 8
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 32) { 
        launch_specialized_kernel<
            B, W, T,
            SM_TH/4, SM_TW/2, SM_TD, // 16, 128, 3072
            SMEM_TD, // 32
            W_TH, W_TW // 16, 8
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else if (size_i == 16) {
        launch_specialized_kernel<
            B, W, T,
            SM_TH/4, SM_TW/4, SM_TD, // 16, 64, 3072
            SMEM_TD, // 32
            W_TH, W_TW // 16, 8
        >(size_i, size_j, size_k, a, b, c, workspace);
    } else {
        return;
    }
}

}; // namespace matmul_tensor

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_k * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_k * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
            mse += diff * diff;
            ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
        }
    }
    mse /= size_i * size_j;
    ref_mean_square /= size_i * size_j;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-3) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
            },
            [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
            });

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
    CUDA_CHECK(cudaFree(flush_gpu));
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_5_BASELINE_IMPL

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

#endif

struct MatmulTensor {
    constexpr static char const *name = "matmul_tensor";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_tensor::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_tensor::launch_matmul_tensor(size_i, size_j, size_k, a, b, c, workspace);
    }
};

BenchmarkResults get_cublas_fma_results() {
    // Hard-coded data collected on A4000 GPU
    return BenchmarkResults{
        "cublas_fma",
        {
            {{3072, 3072, 3072}, 3.152},
            {{2048, 3072, 3072}, 2.174},
            {{1024, 3072, 3072}, 1.090},
            {{512, 3072, 3072}, 0.559},
            {{256, 3072, 3072}, 0.356},
            {{128, 3072, 3072}, 0.256},
            {{64, 3072, 3072}, 0.194},
            {{32, 3072, 3072}, 0.181},
            {{16, 3072, 3072}, 0.181},
        }};
}

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_5_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulTensor>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

void print_speedup(
    std::vector<BenchmarkConfig> const &configs,
    BenchmarkResults const &first,
    BenchmarkResults const &second) {
    printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
    printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
    printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
        auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
        auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
        if (it_first != first.elapsed_ms.end() && it_second != second.elapsed_ms.end()) {
            printf("  %6.02fx", it_first->second / it_second->second);
        } else {
            printf("  %7s", "-");
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";


    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        // {2048, 3072, 3072},
        // {1024, 3072, 3072},
        // {512, 3072, 3072},
        // {256, 3072, 3072},
        // {128, 3072, 3072},
        // {64, 3072, 3072},
        // {32, 3072, 3072},
        // {16, 3072, 3072},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            print_speedup(configs, results.at(i), results.at(j));
        }
    }

    printf("\n-----------------------------------------------------------\n");
    printf("---- Comparison to non-tensor-core cuBLAS performance: ----\n");
    printf("-----------------------------------------------------------\n");

    print_speedup(configs, get_cublas_fma_results(), results.at(results.size() - 1));

    write_json_results("out/results.json", results);

    return 0;
}
