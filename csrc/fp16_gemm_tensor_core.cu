#include <ATen/ATen.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace wmma = nvcuda::wmma;

// A [M,K] x B [K,N] -> C [M,N], FP16 inputs/output, FP32 accumulation.
// One block = one warp = one 16x16 output tile.
__global__ void fp16_gemm_tensor_core_kernel(
    const half *A, const half *B, half *C, int M, int N, int K)
{
    __shared__ __align__(32) half sa[2][16][16];
    __shared__ __align__(32) half sb[2][16][16];
    __shared__ __align__(32) float sc[16][16];

    int tid = threadIdx.x;
    int m0 = blockIdx.y * 16;
    int n0 = blockIdx.x * 16;
    int num_tiles = (K + 15) / 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                   half, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                   half, wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16,
                   float>
        c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Each thread copies 8 adjacent FP16 (16 bytes) for A and B.
    // A/B bases and every row must be 16-byte aligned; all tiles must be full.
    bool use_async = (M >= m0 + 16) && (N >= n0 + 16) &&
                     (K % 16 == 0) && (N % 8 == 0) &&
                     ((reinterpret_cast<unsigned long long>(A) & 15ULL) == 0) &&
                     ((reinterpret_cast<unsigned long long>(B) & 15ULL) == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (use_async)
    {
        for (int t = 0; t < num_tiles; ++t)
        {
            int stage = t & 1; // Ping-pong between shared buffer 0 and 1.
            int k0 = t * 16;
            int row = tid / 2;       // 0..15
            int col = (tid % 2) * 8; // 0 or 8

            const half *src_a = A + (m0 + row) * K + k0 + col;
            const half *src_b = B + (k0 + row) * N + n0 + col;
            unsigned dst_a = static_cast<unsigned>(
                __cvta_generic_to_shared(&sa[stage][row][col]));
            unsigned dst_b = static_cast<unsigned>(
                __cvta_generic_to_shared(&sb[stage][row][col]));

            // Issue Tile t: Global -> Shared, do NOT immediately wait for it.
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst_a), "l"(src_a) : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst_b), "l"(src_b) : "memory");
            asm volatile("cp.async.commit_group;" ::: "memory");

            if (t > 0)
            {
                // Keep at most ONE newest group pending: Tile t may still copy,
                // but Tile t-1 must have completed in every lane.
                asm volatile("cp.async.wait_group 1;" ::: "memory");
                __syncthreads();

                int previous = (t - 1) & 1;
                wmma::load_matrix_sync(a_frag, &sa[previous][0][0], 16);
                wmma::load_matrix_sync(b_frag, &sb[previous][0][0], 16);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                // After MMA, the previous shared stage is free to reuse.
            }
        }

        // Epilogue: consume the very last prefetched Tile.
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        int last = (num_tiles - 1) & 1;
        wmma::load_matrix_sync(a_frag, &sa[last][0][0], 16);
        wmma::load_matrix_sync(b_frag, &sb[last][0][0], 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    else
#endif
    {
        // No cp.async on unaligned or boundary tiles: use the tested safe path.
        for (int k0 = 0; k0 < K; k0 += 16)
        {
            for (int i = tid; i < 256; i += 32)
            {
                int row = i / 16;
                int col = i % 16;
                int m = m0 + row;
                int n = n0 + col;
                int ka = k0 + col;
                int kb = k0 + row;

                sa[0][row][col] = (m < M && ka < K)
                                      ? A[m * K + ka]
                                      : __float2half(0.0f);
                sb[0][row][col] = (kb < K && n < N)
                                      ? B[kb * N + n]
                                      : __float2half(0.0f);
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, &sa[0][0][0], 16);
            wmma::load_matrix_sync(b_frag, &sb[0][0][0], 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
    }

    // Store the FP32 fragment to shared memory, then mask the final FP16 write.
    wmma::store_matrix_sync(
        &sc[0][0], c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < 256; i += 32)
    {
        int m = m0 + i / 16;
        int n = n0 + i % 16;
        if (m < M && n < N)
        {
            C[m * N + n] = __float2half(sc[i / 16][i % 16]);
        }
    }
}

at::Tensor fp16_gemm_tensor_core_launcher(
    const at::Tensor &A, const at::Tensor &B)
{
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same GPU");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf,
                "A and B must be FP16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "A and B must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions");

    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    TORCH_CHECK(M > 0 && N > 0 && K > 0, "M, N, K must be positive");

    c10::cuda::CUDAGuard guard(A.device());
    auto C = at::empty({M, N}, A.options());

    dim3 block(32);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fp16_gemm_tensor_core_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

TORCH_LIBRARY(lowbitlinear_wmma, m)
{
    m.def("fp16_gemm_tensor_core(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(lowbitlinear_wmma, CUDA, m)
{
    m.impl("fp16_gemm_tensor_core", &fp16_gemm_tensor_core_launcher);
}
