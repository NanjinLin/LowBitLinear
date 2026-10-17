"""End-to-end correctness for arbitrary M, N, K with real quantization and no manual padding.

Run from the repository root:
    python tests/test_w4a16_gemm_e2e.py

Requires the updated lowbitlinear/quantization.py and a built CUDA extension.
"""

import torch
import lowbitlinear  # Import package to register lowbitlinear_ext operators

from lowbitlinear.quantization import (
    quantize_int4,
    dequantize_int4,
    pack_int4,
    unpack_int4,
)


torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False

SHAPES = [
    (1, 8, 64),
    (5, 8, 64),
    (4, 1, 64),
    (4, 9, 64),
    (4, 8, 1),
    (4, 8, 7),
    (4, 8, 8),
    (4, 8, 63),
    (4, 8, 65),
    (4, 8, 70),
    (4, 8, 127),
    (4, 8, 129),
    (1, 1, 1),
    (5, 9, 70),
    (6, 10, 71),
    (9, 17, 127),
    (13, 25, 129),
    (32, 64, 512),
]

# Test each operator actually registered in your build. No missing op is assumed to pass.
OP_CANDIDATES = [
    "w4a16_gemm",
    "w4a16_gemm_padding_baseline"
]

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this end-to-end test")

    ops = [
        (name, getattr(torch.ops.lowbitlinear_ext, name))
        for name in OP_CANDIDATES
        if hasattr(torch.ops.lowbitlinear_ext, name)
    ]
    if not ops:
        raise RuntimeError(
            "No W4A16 GEMM operator found in torch.ops.lowbitlinear_ext; "
            "check extension build, import and registration."
        )

    print("Testing registered GEMM operators:", ", ".join(name for name, _ in ops))

    passed = 0
    for M, N, K in SHAPES:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)

        # End-to-end quantize and pack: NO user-side padding.
        q, scales, zeros = quantize_int4(w, group_size=64)
        packed_w = pack_int4(q)

        assert q.shape == (N, K)
        assert packed_w.shape == (N, (K + 7) // 8)
        assert scales.shape == (N, (K + 63) // 64)
        assert zeros.shape == scales.shape
        torch.testing.assert_close(
            unpack_int4(packed_w, original_k=K), q, rtol=0, atol=0
        )

        # Reference uses quantized-and-dequantized weights, not the original FP16 weights.
        w_ref = dequantize_int4(q, scales, zeros, group_size=64)
        y_ref = (x.float() @ w_ref.T).half()

        for op_name, op in ops:
            y_cuda = op(x, packed_w, scales, zeros)
            assert y_cuda.shape == (M, N)
            torch.testing.assert_close(y_cuda, y_ref, rtol=2e-3, atol=2e-3)

            diff = (y_cuda.float() - y_ref.float()).abs()
            print(
                f"{op_name}: M={M}, N={N}, K={K}: PASS | "
                f"max={diff.max().item():.6f}, mean={diff.mean().item():.6f}"
            )
            passed += 1

    print(f"\nALL PASS: {passed} checks ({len(SHAPES)} shapes × {len(ops)} operators)")


if __name__ == "__main__":
    main()
