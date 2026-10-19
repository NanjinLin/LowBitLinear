"""Correctness test for one-warp WMMA + cp.async pipelined GEMM.

Run from the LowBitLinear repo root:
    MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.6 python tests/test_fp16_gemm_tensor_core_pipeline.py

This test compiles only the new .cu file; it does not alter setup.py.
"""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
load(
    name="lowbitlinear_wmma_pipeline_demo",
    sources=[str(root / "csrc" / "fp16_gemm_tensor_core.cu")],
    extra_cuda_cflags=["-O2"],
    is_python_module=False,
    verbose=True,
)

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False

# Aligned full tiles -> cp.async pipeline; partial or unaligned -> masked fallback.
shapes = [
    (16, 16, 16),     # one Tile: pipeline prologue/epilogue
    (16, 16, 32),     # two Tiles: wait_group 1 exercised
    (16, 16, 48),     # three Tiles: shared buffers reused
    (16, 16, 64),
    (32, 32, 128),
    (48, 64, 256),    # several blocks + many K Tiles
    (16, 24, 32),     # N%16 != 0, mix of full/boundary blocks
    (17, 32, 32),     # M boundary
    (16, 16, 24),     # K boundary
    (5, 9, 70),       # all boundaries
    (17, 23, 129),
    (1, 1, 1),
]

for M, N, K in shapes:
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    actual = torch.ops.lowbitlinear_wmma.fp16_gemm_tensor_core(A, B)
    expected = (A.float() @ B.float()).half()
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    diff = (actual.float() - expected.float()).abs().max().item()
    print(f"M={M}, N={N}, K={K}: PASS | max_err={diff:.6f}")

# Contiguous but intentionally misaligned device allocations: fallback path.
M, N, K = 16, 16, 32
A = torch.randn(M * K + 1, device="cuda", dtype=torch.float16)[1:].view(M, K)
B = torch.randn(K * N + 1, device="cuda", dtype=torch.float16)[1:].view(K, N)
assert A.is_contiguous() and B.is_contiguous()
actual = torch.ops.lowbitlinear_wmma.fp16_gemm_tensor_core(A, B)
expected = (A.float() @ B.float()).half()
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
print("misaligned contiguous inputs: PASS")
print(f"ALL PASS: {len(shapes) + 1} cases")
