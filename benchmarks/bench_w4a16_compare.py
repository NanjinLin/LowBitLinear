"""Installed-extension W4A16 comparison. Run from the repository root.

Default: python benchmarks/bench_w4a16_compare.py
Graph replay times contain only captured CUDA GEMV launches, amortized over inner.
All preparation, allocation during capture, loading and correctness checks precede
measurement. Samples alternate kernels within this run; ratios never join runs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import subprocess

import torch

from lowbitlinear._extension import load_cuda_extension
from lowbitlinear.quantization import quantize_int4, pack_int4, unpack_int4, dequantize_int4

OPS = {
    "naive": "w4a16_gemv",
    "warp": "w4a16_gemv_warp",
    "vec": "w4a16_gemv_warp_vec",
}
SHAPES = ((128, 256), (1024, 1024), (2048, 4096), (4096, 4096))
ATOL, RTOL = 2e-3, 1e-3


def percentile(values, q):
    """Linearly interpolated sample percentile, retaining all raw batch samples."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def capture_graph(fn, warmup, inner, graph_warmup):
    # PyTorch recommends warmup on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            output = fn()
    for _ in range(graph_warmup):
        graph.replay()
    torch.cuda.synchronize()
    # Keep the captured output/storage alive until the graph is released.
    return graph, output


def benchmark_graphs(functions, warmup=50, repeats=30, inner=100, graph_warmup=5, seed=42):
    captures = {
        name: capture_graph(fn, warmup, inner, graph_warmup)
        for name, fn in functions.items()
    }
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Materialize CUDA events before measurement.
    start.record()
    end.record()
    end.synchronize()
    samples = {name: [] for name in functions}
    rng = random.Random(seed)
    names = list(functions)
    orders = []
    for sample in range(repeats):
        # Shuffle each cycle, then rotate: each kernel occupies every position.
        if sample % len(names) == 0:
            rng.shuffle(names)
        offset = sample % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        for name in order:
            start.record()
            captures[name][0].replay()
            end.record()
            end.synchronize()  # Never measure asynchronous host enqueue time.
            samples[name].append(start.elapsed_time(end) * 1000.0 / inner)
    return samples, orders


def gpu_snapshot():
    fields = "index,uuid,name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.total"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv"],
            text=True, capture_output=True, timeout=10,
        )
        return {"returncode": result.returncode, "csv": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def power_snapshot():
    supplies = {}
    for directory in Path("/sys/class/power_supply").glob("*"):
        values = {}
        for name in ("type", "online", "status", "capacity", "power_now"):
            try:
                values[name] = (directory / name).read_text().strip()
            except OSError:
                pass
        supplies[directory.name] = values
    return supplies or {"status": "unavailable; AC/battery condition unverified"}


def run_case(n, k, args):
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    before = {"gpu": gpu_snapshot(), "power": power_snapshot()}
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    q, scales, zeros = quantize_int4(weight)
    packed = pack_int4(q)
    # Oracle uses exactly the packed state passed to every implementation.
    weight_fp32 = dequantize_int4(unpack_int4(packed), scales, zeros)
    reference_fp32 = x.float() @ weight_fp32.T
    reference = reference_fp32.half()
    functions = {}
    errors = {}
    for name in args.kernels:
        op = getattr(torch.ops.lowbitlinear_ext, OPS[name])
        functions[name] = lambda op=op: op(x, packed, scales, zeros, 64)
        actual = functions[name]()
        torch.testing.assert_close(actual, reference, atol=ATOL, rtol=RTOL)
        delta = (actual.float() - reference.float()).abs()
        errors[name] = {
            "max_abs_vs_fp16_reference": delta.max().item(),
            "mean_abs_vs_fp16_reference": delta.mean().item(),
            "max_abs_vs_unrounded_fp32": (actual.float() - reference_fp32).abs().max().item(),
        }
    del weight, q, weight_fp32  # Avoid retaining full weights during graph timing.
    torch.cuda.synchronize()
    samples, orders = benchmark_graphs(
        functions, args.warmup, args.repeats, args.inner, args.graph_warmup, args.seed,
    )
    after = {"gpu": gpu_snapshot(), "power": power_snapshot()}
    medians = {name: statistics.median(values) for name, values in samples.items()}
    rows = []
    for name, values in samples.items():
        row = {
            "kernel": name, "median_us": medians[name], "p90_us": percentile(values, 0.9),
            "speedup_vs_naive": medians.get("naive", medians[name]) / medians[name] if "naive" in medians else None,
            "speedup_vs_warp": medians.get("warp", medians[name]) / medians[name] if "warp" in medians else None,
            "errors": errors[name], "samples_us": values,
        }
        rows.append(row)
        naive_ratio = f"{row['speedup_vs_naive']:.2f}x" if row['speedup_vs_naive'] is not None else "n/a"
        warp_ratio = f"{row['speedup_vs_warp']:.2f}x" if row['speedup_vs_warp'] is not None else "n/a"
        print(f"{n:5d} {k:5d} {name:>6} {row['median_us']:11.3f} {row['p90_us']:10.3f} "
              f"{naive_ratio:>9} {warp_ratio:>9} {errors[name]['max_abs_vs_fp16_reference']:12.6f}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  Peak tensor memory: {peak:.1f} MiB; GPU telemetry after: {after['gpu']}", flush=True)
    if args.profile:
        # Optional profiling is outside the timed region.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for fn in functions.values():
                fn()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))
    return {"n": n, "k": k, "rows": rows, "sample_orders": orders,
            "telemetry_before": before, "telemetry_after": after, "peak_tensor_memory_mib": peak}


def parse_shape(value):
    try:
        n, k = map(int, value.lower().split("x"))
        if n <= 0 or k <= 0 or k % 64:
            raise ValueError
        return n, k
    except ValueError:
        raise argparse.ArgumentTypeError("Use NxK, with N>0 and K a positive multiple of 64") from None


def main(default_kernels=("naive", "warp", "vec"), default_shapes=SHAPES):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=parse_shape, nargs="+", default=default_shapes)
    parser.add_argument("--kernels", choices=tuple(OPS), nargs="+", default=default_kernels)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--graph-warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--inner", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--power-condition", default="auto-detected below; performance mode/clock locking unverified")
    parser.add_argument("--json", type=Path, help="Save metadata, raw samples, errors and same-run speedups")
    parser.add_argument("--profile", action="store_true", help="Also print untimed torch.profiler statistics")
    args = parser.parse_args()
    if min(args.warmup, args.graph_warmup, args.inner) <= 0 or args.repeats < 2:
        parser.error("Warmups and inner must be positive; repeats must be >=2")
    if len(set(args.kernels)) != len(args.kernels):
        parser.error("Duplicate kernels are not allowed")
    if not torch.cuda.is_available():
        parser.exit(1, "CUDA GPU unavailable; benchmark was not run.\n")
    torch.cuda.set_device(args.device)
    load_cuda_extension(required=True)
    for name in args.kernels:
        if not hasattr(torch.ops.lowbitlinear_ext, OPS[name]):
            parser.exit(1, f"Missing {OPS[name]}: rebuild the unified extension with pip install -e . --no-build-isolation\n")
    # Disable TF32 so the correctness oracle uses FP32 multiplication.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_capability(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "extension": str(__import__("lowbitlinear")._extension._loaded_library),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "measurement": "CUDA Events around graph replay / inner; microseconds per GEMV; P90 of batch-average samples",
        "semantics": "All kernels: same packed INT4 + FP32 scales/int32 zeros, FP16 activation, FP32 accumulate, FP16 output",
        "oracle": {"tf32": False, "atol": ATOL, "rtol": RTOL, "source": "same packed state; FP32 matmul then FP16"},
        "gpu_before": gpu_snapshot(), "power_before": power_snapshot(),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print("N     K     Kernel  Median(µs)    P90(µs)  Naive/This Warp/This MaxAbsError", flush=True)
    try:
        with torch.inference_mode():
            cases = [run_case(n, k, args) for n, k in args.shapes]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"metadata": metadata, "cases": cases}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
