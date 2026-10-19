"""Minimal CuTe W4A16 target for Nsight Compute.

Run from the LowBitLinear repository root (example M=N=K=512):
    MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.6 ncu \
      --kernel-name regex:cute_w4a16_gemm_kernel \
      --launch-skip 20 --launch-count 1 \
      --set full --force-overwrite \
      --export profiling/cute_w4a16_512 \
      python profiling/profile_cute_w4a16.py 512 512 512
"""
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    M, N, K = map(int, sys.argv[1:4])

    # Load CuTe op independently of the project's installed extension.
    load(
        name="lowbitlinear_cute_ncu",
        sources=[str(ROOT / "csrc" / "cute_w4a16_gemm.cu")],
        extra_include_paths=[str(Path.home() / "projects/cutlass/include")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
        is_python_module=False,
        verbose=False,
    )

    # Import quantization without importing lowbitlinear/__init__.py.
    spec = importlib.util.spec_from_file_location(
        "quantization_for_cute_ncu", ROOT / "lowbitlinear/quantization.py"
    )
    quant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant)

    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.float16)
    q, scales, zeros = quant.quantize_int4(weight, group_size=64)
    packed = quant.pack_int4(q)

    def run():
        return torch.ops.lowbitlinear_cute_w4a16.gemm(
            x, packed, scales, zeros
        )

    with torch.inference_mode():
        for _ in range(20):
            run()
        torch.cuda.synchronize()

        # Exactly one target invocation after 20 warmup launches.
        run()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
