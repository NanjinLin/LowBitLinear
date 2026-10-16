#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" __global__ void w4a16_gemv_warp_vec(
    const half *x,
    const int32_t *packed_w,
    const float *scales,
    const int32_t *zeros,
    half *y,
    int N,
    int K,
    int group_size)
{
    constexpr int WARP_SIZE = 32;

    // 当前线程属于第几个warp
    int warp_in_block = threadIdx.x / WARP_SIZE;

    // 当前线程在warp中编号
    int lane = threadIdx.x % WARP_SIZE;

    // 一个block包含多少warp
    int warps_per_block = blockDim.x / WARP_SIZE;

    // warp_idx
    int n = blockIdx.x * warps_per_block + warp_in_block;

    if (n >= N)
    {
        return;
    }

    // 每行多少个Packed word
    int words_per_row = K / 8;

    float acc = 0.0f;

    // 当前线程负责的Packed word
    for (int word_idx = lane; word_idx < words_per_row; word_idx += WARP_SIZE)
    {
        int packed_offset = n * words_per_row + word_idx;
        uint32_t word = static_cast<uint32_t>(packed_w[packed_offset]);

        // 这个word包含的第一个权重的k
        int k_start = word_idx * 8;

        // 当前word属于哪个量化group
        int group_idx = k_start / group_size;

        int meta_offset = n * (K / group_size) + group_idx;

        float scale = scales[meta_offset];
        int zero = zeros[meta_offset];

        const uint4 x_vec = reinterpret_cast<const uint4 *>(x)[word_idx];

#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            // 取出两个half
            uint32_t bits;
            if (j < 2)
            {
                bits = x_vec.x;
            }
            else if (j < 4)
            {
                bits = x_vec.y;
            }
            else if (j < 6)
            {
                bits = x_vec.z;
            }
            else
            {
                bits = x_vec.w;
            }

            // 取出其中一个
            unsigned short half_bits = static_cast<unsigned short>(
                (j % 2 == 0) ? bits : (bits >> 16));
            float x_val = __half2float(__ushort_as_half(half_bits));

            // INT4解包+反量化
            int q = (word >> (j * 4)) & 0xF;
            float w = (q - zero) * scale;

            acc += x_val * w;
        }
    }

    for (int offset = 16; offset > 0; offset /= 2)
    {
        acc += __shfl_down_sync(
            0xffffffff,
            acc,
            offset);
    }

    if (lane == 0)
    {
        y[n] = __float2half_rn(acc);
    }
}

at::Tensor w4a16_gemv_warp_vec_launcher(
    const at::Tensor &x,
    const at::Tensor &packed_w,
    const at::Tensor &scales,
    const at::Tensor &zeros,
    int64_t group_size)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "Expected x shape [1, K]");
    TORCH_CHECK(x.scalar_type() == at::kHalf,
                "x must be FP16");

    TORCH_CHECK(packed_w.scalar_type() == at::kInt,
                "packed_w must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kFloat,
                "scales must be FP32");
    TORCH_CHECK(zeros.scalar_type() == at::kInt,
                "zeros must be int32");

    TORCH_CHECK(x.is_contiguous() &&
                    packed_w.is_contiguous() &&
                    scales.is_contiguous() &&
                    zeros.is_contiguous(),
                "All tensors must be contiguous");

    TORCH_CHECK(packed_w.is_cuda() &&
                    scales.is_cuda() &&
                    zeros.is_cuda(),
                "All tensors must be CUDA");

    TORCH_CHECK(x.device() == packed_w.device() &&
                    x.device() == scales.device() &&
                    x.device() == zeros.device(),
                "All tensors must be on the same GPU");

    const int64_t K = x.size(1);
    TORCH_CHECK(group_size == 64 && K > 0 && K % 64 == 0,
                "Expected group_size=64 and K divisible by 64");

    TORCH_CHECK(packed_w.dim() == 2,
                "packed_w must be 2D");

    const int64_t N = packed_w.size(0);
    TORCH_CHECK(N > 0 && N <= INT32_MAX && K <= INT32_MAX,
                "Invalid N or K");

    TORCH_CHECK(packed_w.size(1) == K / 8,
                "Invalid packed weight shape");
    TORCH_CHECK(scales.sizes() == at::IntArrayRef({N, K / 64}) &&
                    zeros.sizes() == scales.sizes(),
                "Invalid scales or zeros shape");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<at::Half>()) % 16 == 0,
                "x must be 16-byte aligned for uint4 loads");

    c10::cuda::CUDAGuard device_guard(x.device());

    auto y = at::empty({1, N}, x.options());

    constexpr int threads = 128;
    constexpr int warps_per_block = threads / 32;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;

    w4a16_gemv_warp_vec<<<
        blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<int32_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<int32_t>(),
        reinterpret_cast<half *>(y.data_ptr<at::Half>()),
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<int>(group_size));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA launch failed: ",
                cudaGetErrorString(err));

    return y;
}

TORCH_LIBRARY_FRAGMENT(lowbitlinear_ext, m)
{
    m.def(
        "w4a16_gemv_warp_vec(Tensor x, Tensor packed_w, "
        "Tensor scales, Tensor zeros, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(lowbitlinear_ext, CUDA, m)
{
    m.impl("w4a16_gemv_warp_vec", &w4a16_gemv_warp_vec_launcher);
}