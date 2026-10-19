"""Compare both existing SIMT W4A16 GEMMs against fused W4A16 WMMA + cp.async.

Run from the repository root:
    MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.6 python tests/test_bench_w4a16_tensor_core.py

Skip latency measurement:
    MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.6 python tests/test_bench_w4a16_tensor_core.py --check-only

Both operators are built together from the two local CUDA source files, so this
script does not modify setup.py or import a potentially conflicting binary.
"""
import argparse
import importlib.util
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    # Import the existing quantization implementation directly, without
    # importing lowbitlinear/__init__.py (which may load a second C++ extension).
    quant_path = ROOT / "lowbitlinear" / "quantization.py"
    if not quant_path.exists():
        raise FileNotFoundError(f"Missing project quantizer: {quant_path}")
    spec = importlib.util.spec_from_file_location("quantization_for_wmma_test", quant_path)
    quant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant)

    sources = [ROOT / "csrc" / "w4a16_gemm.cu",
               ROOT / "csrc" / "w4a16_gemm_padding_baseline.cu",
               ROOT / "csrc" / "w4a16_gemm_tensor_core.cu"]
    for file in sources:
        if not file.exists():
            raise FileNotFoundError(file)
    load(
        name="lowbitlinear_w4a16_wmma_compare",
        sources=[str(f) for f in sources],
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
        verbose=True,
    )

    simt = torch.ops.lowbitlinear_ext.w4a16_gemm
    baseline = torch.ops.lowbitlinear_ext.w4a16_gemm_padding
    tc = torch.ops.lowbitlinear_wmma_w4a16.w4a16_gemm_tensor_core
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)

    # Non-multiples exercise the masked fallback. Multiples of 16 exercise
    # the cp.async double-buffer pipeline (including repeated shared reuse).
    check_shapes = [
        (1, 8, 1), (5, 17, 70), (16, 16, 16), (16, 32, 64),
        (17, 33, 80), (32, 64, 128), (32, 64, 512),
        (64, 128, 512), (128, 256, 512), (256, 256, 512),
        (512, 512, 512),
    ]
    print("\n=== Correctness (11 shapes) ===")
    for M, N, K in check_shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)
        q, scales, zeros = quant.quantize_int4(w, group_size=64)
        packed = quant.pack_int4(q)
        weight_fp32 = quant.dequantize_int4(q, scales, zeros, group_size=64)
        # Distinct references are intentional: Tensor Cores consume FP16
        # dequantized weights whereas the SIMT kernel consumes FP32 weights.
        ref_simt = (x.float() @ weight_fp32.T).half()
        ref_tc = (x.float() @ weight_fp32.half().float().T).half()

        y_simt = simt(x, packed, scales, zeros)
        y_tc = tc(x, packed, scales, zeros)
        torch.testing.assert_close(y_simt, ref_simt, rtol=5e-3, atol=2e-2)
        torch.testing.assert_close(y_tc, ref_tc, rtol=5e-3, atol=2e-2)
        if M % 4 == 0 and N % 8 == 0 and K % 64 == 0:
            y_base = baseline(x, packed, scales, zeros)
            torch.testing.assert_close(y_base, ref_simt, rtol=5e-3, atol=2e-2)
        diff = (y_tc.float() - y_simt.float()).abs()
        print(f"PASS M={M:3}, N={N:3}, K={K:3} | "
              f"SIMT-vs-TC max={diff.max().item():.5f}, "
              f"mean={diff.mean().item():.5f}")

    print("ALL PASS: 11 shapes")
    if args.check_only:
        return

    def graph_us(op, op_args, graph_iters=64, trials=7):
        for _ in range(10):
            _ = op(*op_args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_iters):
                output = op(*op_args)
        # Retain output referenced by the captured graph.
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        values = []
        for _ in range(trials):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) * 1000.0 / graph_iters)
        return statistics.median(values)

    print("\n=== CUDA Graph kernel latency (median, microseconds) ===")
    print("M,N,K        Baseline(us)  Regtile(us)  WMMA(us)  Regtile/WMMA")
    for M, N, K in [(32, 64, 512), (64, 128, 512),
                    (128, 256, 512), (256, 256, 512),
                    (512, 512, 512), (128, 256, 1024)]:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)
        q, scales, zeros = quant.quantize_int4(w, group_size=64)
        packed = quant.pack_int4(q)
        inputs = (x, packed, scales, zeros)
        baseline_us = graph_us(baseline, inputs)
        simt_us = graph_us(simt, inputs)
        tc_us = graph_us(tc, inputs)

        print(
            f"{M:3},{N:3},{K:4}       "
            f"{baseline_us:9.3f}    "
            f"{simt_us:9.3f}   "
            f"{tc_us:9.3f}     "
            f"{simt_us / tc_us:8.3f}x"
        )

    print("\nTimes exclude quantization, packing and compilation.")
    print("Each kernel performs its own INT4 dequantization on the GPU.")


if __name__ == "__main__":
    main()