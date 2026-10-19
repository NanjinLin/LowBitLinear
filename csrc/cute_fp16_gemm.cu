#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>

using namespace cute;

// C[32,32] = A[32,64] @ B[32,64].T
// 4 warps FP16存储 FP32计算
__global__ void cute_fp16_gemm_kernel(
    half_t const *A,
    half_t const *B,
    half_t *C)
{
    // global view
    Tensor gA = make_tensor(make_gmem_ptr(A),
                            make_layout(Shape<_32, _64>{}, Stride<_64, _1>{}));
    Tensor gB = make_tensor(make_gmem_ptr(B),
                            make_layout(Shape<_32, _64>{}, Stride<_64, _1>{}));
    Tensor gC = make_tensor(make_gmem_ptr(C),
                            make_layout(Shape<_32, _32>{}, Stride<_32, _1>{}));

    // swizzled shared memory: logical [32, 64]
    auto smem_atom = composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, Shape<_8, _8>>,
               Stride<_8, Stride<_1, _64>>>{});
    auto smem_layout = tile_to_shape(smem_atom, make_shape(_32{}, _64{}));

    __align__(16) __shared__ half_t smemA[32 * 64];
    __align__(16) __shared__ half_t smemB[32 * 64];
    Tensor sA = make_tensor(make_smem_ptr(smemA), smem_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), smem_layout);

    // global -> shared
    auto g2s = make_tiled_copy(
        Copy_Atom<UniversalCopy<half_t>, half_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _1>>{});
    auto thr_g2s = g2s.get_slice(threadIdx.x);

    copy(g2s, thr_g2s.partition_S(gA), thr_g2s.partition_D(sA));
    copy(g2s, thr_g2s.partition_S(gB), thr_g2s.partition_D(sB));
    __syncthreads();

    // 4-warp MMA
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32F16F16F32_TN{},
        // M N方向各两个mma_atom
        Layout<Shape<_2, _2>>{},
        Tile<_32, _32, _16>{});
    auto thr_mma = mma.get_slice(threadIdx.x);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    clear(tCrC);

    // shared -> register
    auto ld_atom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>{};
    auto ldA = make_tiled_copy_A(ld_atom, mma);
    auto ldB = make_tiled_copy_B(ld_atom, mma);

    auto thr_ldA = ldA.get_slice(threadIdx.x);
    auto thr_ldB = ldB.get_slice(threadIdx.x);

    Tensor tXsA = thr_ldA.partition_S(sA);
    Tensor tXsB = thr_ldB.partition_S(sB);
    Tensor tXrA = thr_ldA.retile_D(tCrA);
    Tensor tXrB = thr_ldB.retile_D(tCrB);

    // shared -> register -> mma
    int k_blocks = size<2>(tCrA);
    for (int kb = 0; kb < k_blocks; kb++)
    {
        copy(ld_atom, tXsA(_, _, kb), tXrA(_, _, kb));
        copy(ld_atom, tXsB(_, _, kb), tXrB(_, _, kb));
        gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrC);
    }

    // fp32 -> fp16
    for (int i = 0; i < size(tCrC); i++)
    {
        tCgC(i) = half_t(tCrC(i));
    }
}

at::Tensor cute_fp16_gemm_launcher(
    at::Tensor const &A, at::Tensor const &B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device(), "A and B must share a device");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf,
                "A and B must be float16");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 32 && A.size(1) == 64,
                "A must be [32,64]");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == 32 && B.size(1) == 64,
                "B must be [32,64]");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "A and B must be contiguous");

    auto C = at::empty({32, 32}, A.options());
    cute_fp16_gemm_kernel<<<
        1, 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<cute::half_t const *>(A.data_ptr<at::Half>()),
        reinterpret_cast<cute::half_t const *>(B.data_ptr<at::Half>()),
        reinterpret_cast<cute::half_t *>(C.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

TORCH_LIBRARY(lowbitlinear_cute_fp16, m)
{
    m.def("cute_fp16_gemm(Tensor A, Tensor B) -> Tensor");
}
TORCH_LIBRARY_IMPL(lowbitlinear_cute_fp16, CUDA, m)
{
    m.impl("cute_fp16_gemm", &cute_fp16_gemm_launcher);
}