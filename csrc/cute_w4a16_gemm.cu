#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cstdint>

using namespace cute;

// Y[M,N] = X[M,K] @ dequant(packed_w[N,ceil(K/8)]).T
// x: FP16; packed_w: int32 (8 unsigned INT4 per word, low nibble first)
// scales: FP32 [N,ceil(K/64)]; zeros: int32 [N,ceil(K/64)].
// group_size=64; FP32 dequant -> FP16 MMA input; FP32 accumulation -> FP16 output.
// One CTA computes a 32x32 output tile; each CTA works through K in 64-wide stages.
__global__ void cute_w4a16_gemm_kernel(
    half_t const *X,
    int32_t const *packed_w,
    float const *scales,
    int32_t const *zeros,
    half_t *Y,
    int64_t M,
    int64_t N,
    int64_t K)
{
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 64;

    const int64_t m0 = blockIdx.y * BM;
    const int64_t n0 = blockIdx.x * BN;
    const int64_t packed_k = (K + 7) / 8;
    const int64_t group_count = (K + 63) / 64;

    // Tshared memory
    auto smem_atom = composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, Shape<_8, _8>>,
               Stride<_8, Stride<_1, _64>>>{});
    auto smem_layout = tile_to_shape(smem_atom, make_shape(_32{}, _64{}));

    __align__(16) __shared__ half_t smemX[BM * BK];
    __align__(16) __shared__ half_t smemW[BN * BK];
    Tensor sX = make_tensor(make_smem_ptr(smemX), smem_layout);
    Tensor sW = make_tensor(make_smem_ptr(smemW), smem_layout);

    // tiled copy
    auto g2s = make_tiled_copy(
        Copy_Atom<UniversalCopy<half_t>, half_t>{},
        Layout<Shape<_4, _32>, Stride<_32, _1>>{},
        Layout<Shape<_1, _1>>{});
    auto thr_g2s = g2s.get_slice(threadIdx.x);

    // mma setup
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32F16F16F32_TN{},
        Layout<Shape<_2, _2>>{},
        Tile<_32, _32, _16>{});
    auto thr_mma = mma.get_slice(threadIdx.x);

    Tensor tCrX = thr_mma.partition_fragment_A(sX);
    Tensor tCrW = thr_mma.partition_fragment_B(sW);

    // Y tile prepare
    Tensor gYtile = make_tensor(
        make_gmem_ptr(Y + m0 * N + n0),
        make_layout(Shape<_32, _32>{}, make_stride(N, _1{})));
    Tensor tYgY = thr_mma.partition_C(gYtile);
    Tensor tCrY = thr_mma.make_fragment_C(tYgY);
    clear(tCrY); // Once per output tile, NOT once per K stage.

    // Local logical output coordinates, partitioned identically to gYtile.
    // 在store时判断是否越界
    Tensor cY = make_identity_tensor(make_shape(_32{}, _32{}));
    Tensor tYcY = thr_mma.partition_C(cY);

    // ldmatrix -> register MMA operand path.
    auto ld_atom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>{};
    auto ldX = make_tiled_copy_A(ld_atom, mma);
    auto ldW = make_tiled_copy_B(ld_atom, mma);
    auto thr_ldX = ldX.get_slice(threadIdx.x);
    auto thr_ldW = ldW.get_slice(threadIdx.x);
    Tensor tXsX = thr_ldX.partition_S(sX);
    Tensor tWsW = thr_ldW.partition_S(sW);
    Tensor tXrX = thr_ldX.retile_D(tCrX);
    Tensor tWrW = thr_ldW.retile_D(tCrW);

    // global -> shared memory  with boundary mask
    for (int64_t k0 = 0; k0 < K; k0 += BK)
    {
        // Global X -> Shared. Only the final M/K tail needs masking.
        if (m0 + BM <= M && k0 + BK <= K)
        {
            Tensor gXtile = make_tensor(
                make_gmem_ptr(X + m0 * K + k0),
                make_layout(Shape<_32, _64>{}, make_stride(K, _1{})));
            copy(g2s, thr_g2s.partition_S(gXtile), thr_g2s.partition_D(sX));
        }
        else
        {
            // Every shared element is written, even if its global coordinate
            // is outside the real matrix: the MMA then sees exactly zero.
            for (int linear = threadIdx.x; linear < BM * BK;
                 linear += blockDim.x)
            {
                const int lm = linear / BK;
                const int lk = linear % BK;
                const int64_t gm = m0 + lm;
                const int64_t gk = k0 + lk;
                sX(lm, lk) = (gm < M && gk < K)
                                 ? X[gm * K + gk]
                                 : half_t(0.0f);
            }
        }

        // Packed INT4 -> FP32 dequant -> FP16 Swizzled Shared.
        // Preserve the original per-element scalar unpack/dequant path.
        const int WORDS_PER_ROW = BK / 8;
        const int64_t group = k0 / 64;

        for (int linear = threadIdx.x; linear < BN * WORDS_PER_ROW;
             linear += blockDim.x)
        {
            const int ln = linear / WORDS_PER_ROW;
            const int p = linear % WORDS_PER_ROW;
            const int64_t gn = n0 + ln;
            const int64_t k_base = k0 + p * 8;

            uint32_t bits = 0;
            float scale = 0.0f;
            int zero = 0;

            // 每个packed INT32只读取一次
            if (gn < N && k_base < K)
            {
                bits = static_cast<uint32_t>(
                    packed_w[gn * packed_k + (k_base / 8)]);
                const int64_t param_idx = gn * group_count + group;
                scale = scales[param_idx];
                zero = zeros[param_idx];
            }

// 解包8个INT4
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                const int lk = p * 8 + j;
                const int64_t gk = k_base + j;
                half_t value = half_t(0.0f);

                if (gn < N && gk < K)
                {
                    const int q = (bits >> (4 * j)) & 0xF;
                    const float w = float(q - zero) * scale;
                    value = half_t(w);
                }
                sW(ln, lk) = value;
            }
        }
        __syncthreads();

        const int k_blocks = size<2>(tCrX); // 64 / 16 = 4
        for (int kb = 0; kb < k_blocks; ++kb)
        {
            copy(ld_atom, tXsX(_, _, kb), tXrX(_, _, kb));
            copy(ld_atom, tWsW(_, _, kb), tWrW(_, _, kb));
            gemm(mma, tCrX(_, _, kb), tCrW(_, _, kb), tCrY);
        }
        __syncthreads();
    }

    // Mask only the final M/N output tile. Invalid output never gets stored.
    const auto valid_shape = make_shape(M - m0, N - n0);
    for (int i = 0; i < size(tCrY); ++i)
    {
        if (elem_less(tYcY(i), valid_shape))
        {
            tYgY(i) = half_t(tCrY(i));
        }
    }
}

at::Tensor cute_w4a16_gemm_launcher(
    at::Tensor const &X,
    at::Tensor const &packed_w,
    at::Tensor const &scales,
    at::Tensor const &zeros)
{
    TORCH_CHECK(X.is_cuda() && packed_w.is_cuda() &&
                    scales.is_cuda() && zeros.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(X.device() == packed_w.device() &&
                    X.device() == scales.device() && X.device() == zeros.device(),
                "all tensors must be on the same GPU");
    TORCH_CHECK(X.scalar_type() == at::kHalf, "X must be float16");
    TORCH_CHECK(packed_w.scalar_type() == at::kInt, "packed_w must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kFloat, "scales must be float32");
    TORCH_CHECK(zeros.scalar_type() == at::kInt, "zeros must be int32");
    TORCH_CHECK(X.dim() == 2 && packed_w.dim() == 2 &&
                    scales.dim() == 2 && zeros.dim() == 2,
                "all tensors must be 2D");
    TORCH_CHECK(X.is_contiguous() && packed_w.is_contiguous() &&
                    scales.is_contiguous() && zeros.is_contiguous(),
                "all tensors must be contiguous");

    const int64_t M = X.size(0);
    const int64_t K = X.size(1);
    const int64_t N = packed_w.size(0);
    const int64_t packed_k = (K + 7) / 8;
    const int64_t groups = (K + 63) / 64;
    TORCH_CHECK(packed_w.size(1) == packed_k,
                "packed_w must have shape [N, ceil(K/8)]");
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == groups,
                "scales must have shape [N, ceil(K/64)]");
    TORCH_CHECK(zeros.size(0) == N && zeros.size(1) == groups,
                "zeros must have shape [N, ceil(K/64)]");

    if (M == 0 || N == 0)
    {
        return at::empty({M, N}, X.options());
    }
    if (K == 0)
    {
        return at::zeros({M, N}, X.options());
    }

    at::Tensor Y = at::empty({M, N}, X.options());
    const dim3 grid(
        static_cast<unsigned int>((N + 31) / 32),
        static_cast<unsigned int>((M + 31) / 32));
    cute_w4a16_gemm_kernel<<<grid, 128, 0,
                             c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<half_t const *>(X.data_ptr<at::Half>()),
        packed_w.data_ptr<int32_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<int32_t>(),
        reinterpret_cast<half_t *>(Y.data_ptr<at::Half>()),
        M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

TORCH_LIBRARY(lowbitlinear_cute_w4a16, m)
{
    m.def("gemm(Tensor X, Tensor packed_w, Tensor scales, Tensor zeros) -> Tensor");
}
TORCH_LIBRARY_IMPL(lowbitlinear_cute_w4a16, CUDA, m)
{
    m.impl("gemm", &cute_w4a16_gemm_launcher);
}
