#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

constexpr int BM = 8;
constexpr int BN = 8;
constexpr int BK = 8;

extern "C" __global__ void w4a16_gemm(
    const half *x,
    const int32_t *packed_w,
    const float *scales,
    const int32_t *zeros,
    half *y,
    int M,
    int N,
    int K)
{
    __shared__ half sx[BM][BK];
    // padding
    __shared__ float sw[BN][BK + 1];
    // memory coalesing
    // 每行暂存8个packed INT32 也就是一个量化group的64个INT4
    __shared__ uint32_t sp[BN][8];

    int tid = threadIdx.x;

    // 当前block起始位置
    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;

    // 每个thread对应的输出位置
    int local_m = tid / BN;
    int local_n = tid % BN;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // 定义寄存器变量 避免同一组重复加载
    float scale_reg = 0.0f;
    int zero_reg = 0;

    // K boundary mask
    int packed_cols = (K + 7) / 8;

    for (int g = 0; g < (K + 63) / 64; g++)
    {
        // memory coalescing
        // 一个block负责8行 一个warp负责4行 每行8个packed INT32 一个thread一个
        // 每8个thread一组 memory coalescing
        for (int phase = 0; phase < 2; ++phase)
        {
            int rn = phase * 4 + tid / 8;
            int p = tid % 8;

            int n = n0 + rn;

            int packed_col = g * 8 + p;

            if (n < N && packed_col < packed_cols)
            {
                int packed_idx = n * packed_cols + packed_col;
                sp[rn][p] =
                    static_cast<uint32_t>(packed_w[packed_idx]);
            }
            else
            {
                sp[rn][p] = 0;
            }
        }

        // quantization
        if (tid < BN)
        {
            int n = n0 + tid;

            if (n < N)
            {
                int group_idx = n * ((K + 63) / 64) + g;

                scale_reg = scales[group_idx];
                zero_reg = zeros[group_idx];
            }
            else
            {
                scale_reg = 0.0f;
                zero_reg = 0.0;
            }
        }

        __syncthreads();

        // 遍历一组里的8个package
        for (int t = 0; t < 8; t++)
        {
            int k0 = g * 64 + t * BK;

            // load X tile
            // 一个thread负责两个输出
            for (int idx = tid; idx < BM * BK; idx += blockDim.x)
            {
                int rm = idx / BK;
                int rk = idx % BK;

                // boundary mask
                int m = m0 + rm;
                int k = k0 + rk;

                if (m < M && k < K)
                {
                    sx[rm][rk] = x[m * K + k];
                }
                else
                {
                    sx[rm][rk] = __float2half(0.0f);
                }
            }
            // load W tile
            if (tid < BN)
            {
                // 每个thread负责解包一个package
                uint32_t bits = sp[tid][t];

                for (int j = 0; j < BK; ++j)
                {
                    int q = (bits >> (4 * j)) & 0xF;

                    sw[tid][j] =
                        float(q - zero_reg) * scale_reg;
                }
            }

            __syncthreads();

            // calculation
            for (int rk = 0; rk < BK; rk++)
            {
                float a0 = __half2float(sx[local_m][rk]);
                float a1 = __half2float(sx[local_m + 4][rk]);
                float b = sw[local_n][rk];

                acc0 += a0 * b;
                acc1 += a1 * b;
            }

            __syncthreads();
        }
    }

    // regTile + boudary mask
    int m0_out = m0 + local_m;
    int m1_out = m0 + local_m + 4;
    int n = n0 + local_n;
    if (m0_out < M && n < N)
    {
        y[m0_out * N + n] = __float2half(acc0);
    }
    if (m1_out < M && n < N)
    {
        y[m1_out * N + n] = __float2half(acc1);
    }
}

at::Tensor w4a16_gemm_launcher(
    const at::Tensor &x,
    const at::Tensor &w,
    const at::Tensor &scales,
    const at::Tensor &zeros)
{
    TORCH_CHECK(x.dim() == 2,
                "Expected x shape [M, K]");
    TORCH_CHECK(w.dim() == 2,
                "w must be 2D");

    // get shape
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    // check
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x/w must be CUDA");
    TORCH_CHECK(w.size(1) == (K + 7) / 8,
                "packed_w must have shape [N, (K + 7)/8]");
    TORCH_CHECK(x.scalar_type() == at::kHalf,
                "x must be FP16");

    TORCH_CHECK(w.scalar_type() == at::kInt,
                "w must be INT32");
    TORCH_CHECK(x.is_contiguous() &&
                    w.is_contiguous(),
                "All tensors must be contiguous");
    TORCH_CHECK(x.device() == w.device(),
                "All tensors must be on the same GPU");

    TORCH_CHECK(M > 0 && N > 0 && K > 0);

    TORCH_CHECK(scales.dim() == 2,
                "scales must be 2D");

    TORCH_CHECK(zeros.dim() == 2,
                "zeros must be 2D");

    TORCH_CHECK(
        scales.size(0) == N &&
            scales.size(1) == (K + 63) / 64,
        "scales must have shape [N, (K + 63)/64]");

    TORCH_CHECK(
        zeros.size(0) == N &&
            zeros.size(1) == (K + 63) / 64,
        "zeros must have shape [N, (K + 63)/64]");

    TORCH_CHECK(scales.scalar_type() == at::kFloat);
    TORCH_CHECK(zeros.scalar_type() == at::kInt);

    TORCH_CHECK(scales.is_cuda() && zeros.is_cuda());

    TORCH_CHECK(scales.is_contiguous());
    TORCH_CHECK(zeros.is_contiguous());

    TORCH_CHECK(scales.device() == x.device());
    TORCH_CHECK(zeros.device() == x.device());

    // allocate output
    auto y = at::empty({M, N}, x.options());

    // get pointer
    const half *x_ptr =
        reinterpret_cast<const half *>(x.data_ptr<at::Half>());

    const int32_t *w_ptr =
        w.data_ptr<int32_t>();

    const float *scales_ptr =
        scales.data_ptr<float>();

    const int32_t *zeros_ptr =
        zeros.data_ptr<int32_t>();

    half *y_ptr =
        reinterpret_cast<half *>(y.data_ptr<at::Half>());

    // launch kernel
    dim3 block(32);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    c10::cuda::CUDAGuard device_guard(x.device());

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream().stream();

    w4a16_gemm<<<grid, block, 0, stream>>>(
        x_ptr, w_ptr, scales_ptr, zeros_ptr, y_ptr, M, N, K);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}

TORCH_LIBRARY_FRAGMENT(lowbitlinear_ext, m)
{
    m.def("w4a16_gemm(Tensor x, Tensor packed_w, Tensor scales, Tensor zeros) -> Tensor");
}

TORCH_LIBRARY_IMPL(lowbitlinear_ext, CUDA, m)
{
    m.impl("w4a16_gemm", &w4a16_gemm_launcher);
}