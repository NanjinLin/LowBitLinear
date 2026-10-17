import torch
import torch.nn.functional as F
import lowbitlinear

from lowbitlinear.quantization import (
    quantize_int4,
    pack_int4,
    unpack_int4,
    dequantize_int4,
)

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False

shapes = [
    # M boundary
    (1, 8, 64),
    (5, 8, 64),
    # N boundary
    (4, 1, 64),
    (4, 9, 64),
    # K boundary (8-pack and 64-group tails)
    (4, 8, 1),
    (4, 8, 7),
    (4, 8, 8),
    (4, 8, 63),
    (4, 8, 65),
    (4, 8, 70),
    (4, 8, 127),
    (4, 8, 129),
    # M, N, K boundaries together
    (1, 1, 1),
    (5, 9, 70),
    (6, 10, 71),
    (9, 17, 127),
    (13, 25, 129),
    # Aligned regression test
    (32, 64, 512),
]

for M, N, K in shapes:
    x = torch.randn(
        M, K, device="cuda", dtype=torch.float16
    )

    w = torch.randn(
        N, K, device="cuda", dtype=torch.float16
    )

    # Existing quantizer requires K % 64 == 0.
    # Pad only the test's weights before quantization.
    k_pad64 = ((K + 63) // 64) * 64
    k_pad8 = ((K + 7) // 8) * 8
    w_padded = F.pad(w, (0, k_pad64 - K))
    q_full, scales, zeros = quantize_int4(w_padded, group_size=64)

    # Store only ceil(K/8) packed words per row.
    q_packed = F.pad(q_full[:, :K], (0, k_pad8 - K))
    packed_w = pack_int4(q_packed)

    # Verify packing
    torch.testing.assert_close(
        unpack_int4(packed_w), q_packed,
        rtol=0, atol=0
    )

    # Fused CUDA GEMM
    y_cuda = torch.ops.lowbitlinear_ext.w4a16_gemm(
        x, packed_w, scales, zeros
    )

    # FP32 Reference
    w_ref = dequantize_int4(
        q_full, scales, zeros, group_size=64
    )[:,:K]

    y_ref = (x.float() @ w_ref.T).half()

    diff = (y_cuda.float() - y_ref.float()).abs()

    max_err = diff.max().item()
    mean_err = diff.mean().item()

    torch.testing.assert_close(
        y_cuda, y_ref,
        rtol=2e-3, atol=2e-3
    )

    print(
        f"M={M}, N={N}, K={K}: PASS | "
        f"max={max_err:.6f}, mean={mean_err:.6f}"
    )