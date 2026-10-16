#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" __global__ void w4a16_gemv_naive(
    const half *x,
    const int32_t *packed_w,
    const float *scales,
    const int32_t *zeros,
    half *y,
    int N,
    int K,
    int group_size)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N)
    {
        return;
    }

    float acc = 0.0f;

    for (int k = 0; k < K; k++)
    {

        // 当前权重的packed int32
        int word_idx = n * (K / 8) + k / 8;
        uint32_t word = static_cast<uint32_t>(packed_w[word_idx]);

        // 提取int4权重
        int shift = (k % 8) * 4;
        int q = (word >> shift) & 0xF;

        // 获取当前group的量化参数
        int group_idx = n * (K / group_size) + k / group_size;
        float scale = scales[group_idx];
        int zero = zeros[group_idx];

        // 反量化+类型转换
        float w = (q - zero) * scale;

        // GEMV
        acc += __half2float(x[k]) * w;
    }

    y[n] = __float2half_rn(acc);
}

at::Tensor w4a16_gemv_launcher(
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

    c10::cuda::CUDAGuard device_guard(x.device());

    auto y = at::empty({1, N}, x.options());

    constexpr int threads = 128;
    const int blocks = (N + threads - 1) / threads;

    w4a16_gemv_naive<<<
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

TORCH_LIBRARY(lowbitlinear_ext, m)
{
    m.def(
        "w4a16_gemv(Tensor x, Tensor packed_w, "
        "Tensor scales, Tensor zeros, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(lowbitlinear_ext, CUDA, m)
{
    m.impl("w4a16_gemv", &w4a16_gemv_launcher);
}