import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
CUTLASS_INCLUDE = os.path.expanduser(
    os.environ.get("CUTLASS_INCLUDE", "~/projects/cutlass/include")
)

load(
    name="lowbitlinear_cute_fp16",
    sources=[str(ROOT / "csrc" / "cute_fp16_gemm.cu")],
    extra_include_paths=[os.path.expanduser("~/projects/cutlass/include")],
    extra_cflags=["-std=c++20"],
    extra_cuda_cflags=["-std=c++20", "-lineinfo"],
    is_python_module=False,
    verbose=True,
)

op = torch.ops.lowbitlinear_cute_fp16.cute_fp16_gemm

def check(A, B, name, atol=1e-2, rtol=1e-2):
    C = op(A, B)
    torch.cuda.synchronize()
    ref = (A.float() @ B.float().T).half()
    torch.testing.assert_close(C, ref, atol=atol, rtol=rtol)
    err = (C.float() - ref.float()).abs().max().item()
    print(f"{name}: PASS (max_abs_err={err:.6f})")

A = torch.ones((32, 64), device="cuda", dtype=torch.float16)
B = torch.arange(1, 65, device="cuda", dtype=torch.float16).repeat(32, 1)
check(A, B, "K accumulation", atol=0, rtol=0)  # output all 2080

A = (torch.arange(32 * 64, device="cuda") % 101).half().reshape(32, 64)
B = torch.zeros((32, 64), device="cuda", dtype=torch.float16)
B[torch.arange(32), torch.arange(32)] = 1
check(A, B, "One-hot B mapping", atol=0, rtol=0)

torch.manual_seed(42)
A = torch.randn((32, 64), device="cuda", dtype=torch.float16)
B = torch.randn((32, 64), device="cuda", dtype=torch.float16)
check(A, B, "Random FP16")
print("All tests passed")
