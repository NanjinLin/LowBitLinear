
import torch
import lowbitlinear

torch.manual_seed(0)

M, N, K = 8, 16, 128
GROUP_SIZE = 64

# 1. FP16 Activation
x = torch.randn(
    M, K, device="cuda", dtype=torch.float16
)

# 2. Generate INT4 values: [0, 15]
q = torch.randint(
    0, 16, (N, K), dtype=torch.int64
)

# 3. Pack eight INT4 values into one INT32
shifts = torch.arange(8, dtype=torch.int64) * 4

packed_w = (
    (q.reshape(N, K // 8, 8) << shifts)
    .sum(dim=-1)
    .to(torch.int32)
    .cuda()
)

# 4. Group-wise scales and zeros
scales = (
    torch.rand(N, K // GROUP_SIZE) * 0.1 + 0.01
).float().cuda()

zeros = torch.randint(
    0, 16,
    (N, K // GROUP_SIZE),
    dtype=torch.int32
).cuda()

# 5. CUDA fused implementation
y_cuda = torch.ops.lowbitlinear_ext.w4a16_gemm(
    x, packed_w, scales, zeros
)

# 6. Independent FP32 reference
q_fp32 = q.cuda().float()

expanded_scales = scales.repeat_interleave(
    GROUP_SIZE, dim=1
)

expanded_zeros = zeros.float().repeat_interleave(
    GROUP_SIZE, dim=1
)

w_ref = (q_fp32 - expanded_zeros) * expanded_scales

torch.backends.cuda.matmul.allow_tf32 = False
y_ref = (x.float() @ w_ref.T).half()

# 7. Compare
torch.cuda.synchronize()

diff = (y_cuda.float() - y_ref.float()).abs()

print("CUDA Shape:", y_cuda.shape)
print("Reference Shape:", y_ref.shape)
print("Max Abs Error:", diff.max().item())
print("Mean Abs Error:", diff.mean().item())

torch.testing.assert_close(
    y_cuda,
    y_ref,
    rtol=2e-3,
    atol=2e-3
)

print("W4A16 GEMM Correctness Passed!")
