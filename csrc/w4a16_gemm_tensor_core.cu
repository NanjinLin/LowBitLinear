#include <ATen/ATen.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <climits>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace wmma = nvcuda::wmma;

// W4A16: x[M,K] FP16, packed_w[N,ceil(K/8)] INT32,
// scales/zeros[N,ceil(K/64)], y[M,N] FP16.
// One block / one warp / one WMMA output tile (16 x 16).
// Full tiles: cp.async copies FP16 x and packed INT4 words into 2 shared stages.
// Partial/unaligned tiles: safe masked loads, same Tensor Core computation.
__global__ void w4a16_gemm_tensor_core_kernel(
    const half* x, const int32_t* packed_w,
    const float* scales, const int32_t* zeros,
    half* y, int M, int N, int K)
{
    __shared__ __align__(32) half sx[2][16][16];
    __shared__ __align__(32) int32_t sp[2][16][2];
    __shared__ __align__(32) half sw[16][16]; // B[K,N], FP16 after dequant
    __shared__ __align__(32) float sy[16][16];

    int tid = threadIdx.x;
    int m0 = blockIdx.y * 16;
    int n0 = blockIdx.x * 16;
    int packed_cols = (K + 7) / 8;
    int num_groups = (K + 63) / 64;
    int num_tiles = (K + 15) / 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                           half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                           half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16,
                           float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // FP16 x needs 16-byte aligned rows for a 16B/thread cp.async.
    // Packed INT4 needs only 4-byte alignment for a 4B/thread cp.async.
    bool use_async = (m0 + 16 <= M) && (n0 + 16 <= N) &&
                     (K % 16 == 0) &&
                     ((reinterpret_cast<unsigned long long>(x) & 15ULL) == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (use_async)
    {
        // At iteration t: prefetch tile t, then consume tile t-1.
        for (int t = 0; t < num_tiles; ++t)
        {
            int stage = t & 1;
            int k0 = t * 16;
            int row = tid / 2;        // 0..15
            int col = (tid % 2) * 8;  // 0 or 8

            const half* src_x = x + (m0 + row) * K + k0 + col;
            const int32_t* src_w = packed_w + (n0 + row) * packed_cols + k0 / 8 + (tid % 2);
            unsigned dst_x = static_cast<unsigned>(__cvta_generic_to_shared(&sx[stage][row][col]));
            unsigned dst_w = static_cast<unsigned>(__cvta_generic_to_shared(&sp[stage][row][tid % 2]));

            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                         :: "r"(dst_x), "l"(src_x) : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                         :: "r"(dst_w), "l"(src_w) : "memory");
            asm volatile("cp.async.commit_group;" ::: "memory");

            if (t > 0)
            {
                // All groups except the newest (tile t) must be complete.
                asm volatile("cp.async.wait_group 1;" ::: "memory");
                __syncthreads();

                int previous = (t - 1) & 1;
                int group = ((t - 1) * 16) / 64;

                // Convert packed INT4[N,K] to FP16 B[K,N] in shared memory.
                for (int i = tid; i < 256; i += 32)
                {
                    int k_local = i / 16;
                    int n_local = i % 16;
                    uint32_t bits = static_cast<uint32_t>(sp[previous][n_local][k_local / 8]);
                    int q = (bits >> (4 * (k_local % 8))) & 0xF;
                    int n = n0 + n_local;
                    int idx = n * num_groups + group;
                    sw[k_local][n_local] = __float2half_rn(
                        float(q - zeros[idx]) * scales[idx]);
                }
                __syncthreads();

                wmma::load_matrix_sync(a_frag, &sx[previous][0][0], 16);
                wmma::load_matrix_sync(b_frag, &sw[0][0], 16);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                __syncthreads(); // sw and this shared stage are safe to reuse.
            }
        }

        // Drain the pipeline and compute the last tile.
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        int last = (num_tiles - 1) & 1;
        int last_group = ((num_tiles - 1) * 16) / 64;
        for (int i = tid; i < 256; i += 32)
        {
            int k_local = i / 16;
            int n_local = i % 16;
            uint32_t bits = static_cast<uint32_t>(sp[last][n_local][k_local / 8]);
            int q = (bits >> (4 * (k_local % 8))) & 0xF;
            int n = n0 + n_local;
            int idx = n * num_groups + last_group;
            sw[k_local][n_local] = __float2half_rn(float(q - zeros[idx]) * scales[idx]);
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, &sx[last][0][0], 16);
        wmma::load_matrix_sync(b_frag, &sw[0][0], 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    else
#endif
    {
        // Safe masked fallback for M/N/K tails and misaligned x.
        for (int k0 = 0; k0 < K; k0 += 16)
        {
            for (int i = tid; i < 256; i += 32)
            {
                int k_local = i / 16;
                int n_local = i % 16;
                int m_local = i / 16;
                int x_k = k0 + (i % 16);
                int w_k = k0 + k_local;
                int m = m0 + m_local;
                int n = n0 + n_local;

                sx[0][m_local][i % 16] = (m < M && x_k < K)
                    ? x[m * K + x_k] : __float2half_rn(0.0f);

                if (n < N && w_k < K)
                {
                    uint32_t bits = static_cast<uint32_t>(
                        packed_w[n * packed_cols + w_k / 8]);
                    int q = (bits >> (4 * (w_k % 8))) & 0xF;
                    int idx = n * num_groups + w_k / 64;
                    sw[k_local][n_local] = __float2half_rn(
                        float(q - zeros[idx]) * scales[idx]);
                }
                else
                {
                    sw[k_local][n_local] = __float2half_rn(0.0f);
                }
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, &sx[0][0][0], 16);
            wmma::load_matrix_sync(b_frag, &sw[0][0], 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
    }

    wmma::store_matrix_sync(&sy[0][0], c_frag, 16, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 256; i += 32)
    {
        int m = m0 + i / 16;
        int n = n0 + i % 16;
        if (m < M && n < N)
            y[m * N + n] = __float2half_rn(sy[i / 16][i % 16]);
    }
}

at::Tensor w4a16_gemm_tensor_core_launcher(
    const at::Tensor& x, const at::Tensor& packed_w,
    const at::Tensor& scales, const at::Tensor& zeros)
{
    TORCH_CHECK(x.dim() == 2 && packed_w.dim() == 2 &&
                scales.dim() == 2 && zeros.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(x.is_cuda() && packed_w.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(x.device() == packed_w.device() && x.device() == scales.device() &&
                x.device() == zeros.device(), "All inputs must be on the same CUDA device");
    TORCH_CHECK(x.is_contiguous() && packed_w.is_contiguous() &&
                scales.is_contiguous() && zeros.is_contiguous(), "All inputs must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf && packed_w.scalar_type() == at::kInt &&
                scales.scalar_type() == at::kFloat && zeros.scalar_type() == at::kInt,
                "Expected x FP16, packed_w INT32, scales FP32, zeros INT32");

    int64_t M64 = x.size(0), K64 = x.size(1), N64 = packed_w.size(0);
    TORCH_CHECK(M64 > 0 && N64 > 0 && K64 > 0 &&
                M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
                "M/N/K must be positive and fit in int32");
    int M = static_cast<int>(M64);
    int N = static_cast<int>(N64);
    int K = static_cast<int>(K64);
    TORCH_CHECK(packed_w.size(1) == (K64 + 7) / 8,
                "packed_w must have shape [N,ceil(K/8)]");
    TORCH_CHECK(scales.size(0) == N64 && scales.size(1) == (K64 + 63) / 64 &&
                zeros.size(0) == N64 && zeros.size(1) == (K64 + 63) / 64,
                "scales and zeros must have shape [N,ceil(K/64)]");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto y = at::empty({M64, N64}, x.options());
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    dim3 block(32);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    w4a16_gemm_tensor_core_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<int32_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<int32_t>(),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

TORCH_LIBRARY(lowbitlinear_wmma_w4a16, m)
{
    m.def("w4a16_gemm_tensor_core(Tensor x, Tensor packed_w, Tensor scales, Tensor zeros) -> Tensor");
}

TORCH_LIBRARY_IMPL(lowbitlinear_wmma_w4a16, CUDA, m)
{
    m.impl("w4a16_gemm_tensor_core", &w4a16_gemm_tensor_core_launcher);
}
