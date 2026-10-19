"""W4A16 CUDA Graph benchmark: PyTorch Reference / SIMT / WMMA / CuTe.

Run from the LowBitLinear repository root:
    MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.6 python benchmarks/bench_w4a16_four_way.py
"""
import importlib.util
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]


def reference(x, packed, scales, zeros):
    """Unpack INT4 + FP32 dequant + FP32 matmul + FP16 output, all on GPU."""
    n = packed.shape[0]
    k = x.shape[1]
    shifts = torch.arange(8, device=x.device, dtype=torch.int32) * 4
    q = ((packed.unsqueeze(-1) >> shifts) & 15).reshape(n, -1)[:, :k]
    groups = torch.arange(k, device=x.device) // 64
    w = (q.float() - zeros[:, groups].float()) * scales[:, groups]
    return (x.float() @ w.T).half()


def graph_us(op, args, graph_iters=64, trials=7):
    for _ in range(10):
        op(*args)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            output = op(*args)

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(trials):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / graph_iters)
    return statistics.median(times)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    # Reuse the repo's quantizer without importing the package's compiled extension.
    spec = importlib.util.spec_from_file_location(
        "quantization_for_benchmark", ROOT / "lowbitlinear" / "quantization.py"
    )
    quant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant)

    sources = [
        ROOT / "csrc" / "w4a16_gemm.cu",
        ROOT / "csrc" / "w4a16_gemm_tensor_core.cu",
        ROOT / "csrc" / "cute_w4a16_gemm.cu",
    ]
    for src in sources:
        if not src.is_file():
            raise FileNotFoundError(src)

    load(
        name="lowbitlinear_w4a16_four_way",
        sources=[str(src) for src in sources],
        extra_include_paths=[str(Path.home() / "projects" / "cutlass" / "include")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        is_python_module=False,
        verbose=True,
    )

    ops = {
        "Reference": reference,
        "SIMT": torch.ops.lowbitlinear_ext.w4a16_gemm,
        "WMMA": torch.ops.lowbitlinear_wmma_w4a16.w4a16_gemm_tensor_core,
        "CuTe": torch.ops.lowbitlinear_cute_w4a16.gemm,
    }

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False

    # Same shape sweep style as bench_w4a16_tensor_core.py.
    shapes = [
        (32, 64, 512),
        (64, 128, 512),
        (128, 256, 512),
        (256, 256, 512),
        (512, 512, 512),
        (128, 256, 1024),
    ]

    print("\n=== W4A16 CUDA Graph latency (median, us; lower is better) ===")
    print(f"{'M,N,K':<17} {'Reference':>10} {'SIMT':>10} {'WMMA':>10} {'CuTe':>10}  Ranking")

    for m, n, k in shapes:
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16)
        q, scales, zeros = quant.quantize_int4(w, group_size=64)
        packed = quant.pack_int4(q)
        args = (x, packed, scales, zeros)

        # SIMT accumulates using FP32 dequant; WMMA/CuTe round dequant to FP16.
        w32 = quant.dequantize_int4(q, scales, zeros, group_size=64)
        ref_simt = (x.float() @ w32.T).half()
        ref_tc = (x.float() @ w32.half().float().T).half()
        torch.testing.assert_close(reference(*args), ref_simt, atol=2e-2, rtol=5e-3)
        torch.testing.assert_close(ops["SIMT"](*args), ref_simt, atol=2e-2, rtol=5e-3)
        for name in ("WMMA", "CuTe"):
            torch.testing.assert_close(ops[name](*args), ref_tc, atol=2e-2, rtol=5e-3)

        times = {name: graph_us(op, args) for name, op in ops.items()}
        ranking = " < ".join(sorted(times, key=times.get))
        print(
            f"{m:3},{n:3},{k:4}     "
            f"{times['Reference']:10.3f} {times['SIMT']:10.3f} "
            f"{times['WMMA']:10.3f} {times['CuTe']:10.3f}  {ranking}"
        )

    print("\nReference includes GPU INT4 unpack + FP32 dequant + FP32 matmul.")
    print("All times exclude quantization/packing of original FP16 weights and JIT compilation.")
    print("WMMA/CuTe may differ numerically from Reference/SIMT due to FP16 dequant rounding.")


if __name__ == "__main__":
    main()
