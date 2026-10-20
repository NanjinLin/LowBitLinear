"""Benchmark the installed W4A16 public API, a matched packed reference and FP16 compute baseline."""
import argparse
import json
from pathlib import Path

import torch

if __package__:
    from ._common import (GEMV_SHAPES, GEMM_SHAPES, parse_shape, route_for, require_cuda,
                          precision_policy, make_inputs, decoded_weight, expected_output,
                          packed_reference, check_output, measure, summarize, metadata, gpu_snapshot)
else:
    from _common import (GEMV_SHAPES, GEMM_SHAPES, parse_shape, route_for, require_cuda,
                         precision_policy, make_inputs, decoded_weight, expected_output,
                         packed_reference, check_output, measure, summarize, metadata, gpu_snapshot)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", choices=("all", "gemv", "gemm"), default="all")
    result.add_argument("--shapes", type=parse_shape, nargs="+", help="Override suite shapes: MxNxK ...")
    result.add_argument("--mode", choices=("api", "graph"), default="api")
    result.add_argument("--warmup", type=int, default=50)
    result.add_argument("--repeats", type=int, default=30, help="Independent timing samples, at least 3")
    result.add_argument("--graph-batch", type=int, default=32, help="Calls captured in each graph; graph mode only")
    result.add_argument("--graph-warmup", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--json", type=Path, help="Write metadata, raw samples, correctness errors and same-run ratios")
    return result


def run_case(shape, device, args):
    x, layer = make_inputs(shape, device, args.seed)
    route = route_for(shape)
    oracle = expected_output(x, layer, route)
    fp16_weight = decoded_weight(layer).half()
    fp16_oracle = (x.float() @ fp16_weight.float().T).half()
    functions = {
        "w4a16_api": lambda: layer(x),
        "packed_reference": lambda: packed_reference(x, layer, route),
        "predequant_fp16": lambda: x @ fp16_weight.T,
    }
    expected = {"w4a16_api": oracle, "packed_reference": oracle, "predequant_fp16": fp16_oracle}
    errors = {name: check_output(fn(), expected[name]) for name, fn in functions.items()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples, orders = measure(functions, expected, args)
    results = {name: summarize(values) for name, values in samples.items()}
    target_us = results["w4a16_api"]["median_us"]
    speedup = results["packed_reference"]["median_us"] / target_us
    fp16_ratio = results["predequant_fp16"]["median_us"] / target_us
    m, n, k = shape
    print(f"| {m}x{n}x{k} | {route} | {target_us:.3f} | "
          f"{results['packed_reference']['median_us']:.3f} | "
          f"{results['predequant_fp16']['median_us']:.3f} | {speedup:.3f} | {fp16_ratio:.3f} |", flush=True)
    return {
        "m": m, "n": n, "k": k, "route": route, "mode": args.mode,
        "operator": ("lowbitlinear_ext::w4a16_gemv_warp_vec" if route == "gemv" else "lowbitlinear_cute_w4a16::gemm"),
        "callable": "QuantizedLinear.forward", "results": results,
        "max_abs_error": errors, "atol": 2e-3, "rtol": 1e-3,
        "speedup_vs_matched_packed_reference": speedup,
        "predequant_fp16_over_w4a16_context_ratio": fp16_ratio,
        "sample_orders": orders, "peak_tensor_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.warmup < 1 or args.repeats < 3 or args.graph_batch < 1 or args.graph_warmup < 1:
        cli.error("warmup/graph-batch/graph-warmup must be positive; repeats must be >=3")
    if args.shapes is None:
        args.shapes = GEMV_SHAPES if args.suite == "gemv" else GEMM_SHAPES if args.suite == "gemm" else GEMV_SHAPES + GEMM_SHAPES
    if len(set(args.shapes)) != len(args.shapes):
        cli.error("Duplicate shapes are not allowed")
    if args.suite != "all" and any(route_for(shape) != args.suite for shape in args.shapes):
        cli.error("Shapes must select the requested public API route: GEMV requires M=1, N%128=0, K%64=0; use --suite all for mixed routes")
    device = require_cuda(args.device)
    with precision_policy():
        info = metadata(args)
        info["baselines"] = {
            "packed_reference": "Same packed INT4 state, unpack+FP32 dequant+matmul each call; CuTe path additionally rounds weights to FP16 before FP32 matmul",
            "predequant_fp16": "Predecoded FP16 weights, FP16 matmul with FP32 accumulation. Excludes unpack/dequant, changes weight storage; GEMV also changes dequant rounding. Context only, not an equivalent W4A16 replacement",
        }
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        print("\nAll latencies are median us. Ratios use only this run and timing mode.", flush=True)
        print("FP16/W4A16 is a compute-only context ratio, not an equivalent-operator speedup.", flush=True)
        print("| MxNxK | route | W4A16 us | packed ref us | predecoded FP16 us | ref/W4A16 | FP16/W4A16 context |", flush=True)
        print("|---|---|---:|---:|---:|---:|---:|", flush=True)
        cases = [run_case(shape, device, args) for shape in args.shapes]
        info["gpu_after"] = gpu_snapshot()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"metadata": info, "cases": cases}, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"metadata": info, "cases": cases}


if __name__ == "__main__":
    main()
