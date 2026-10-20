"""Shared inputs, precision policy and timing for the two benchmark entry points."""
from contextlib import contextmanager
from datetime import datetime, timezone
import math
import platform
import random
import statistics
import subprocess
from time import perf_counter_ns

import torch

from lowbitlinear import QuantizedLinear
from lowbitlinear._extension import load_cuda_extension
from lowbitlinear.linear import linear_int4_reference
from lowbitlinear.quantization import unpack_int4, dequantize_int4

GROUP_SIZE = 64
ATOL, RTOL = 2e-3, 1e-3
GEMV_SHAPES = [(1, 128, 256), (1, 1024, 1024), (1, 2048, 4096)]
GEMM_SHAPES = [(1, 1, 1), (17, 23, 65), (33, 37, 129),
               (32, 64, 512), (128, 256, 1024), (512, 2048, 2048)]


def parse_shape(value):
    import argparse
    try:
        shape = tuple(int(v) for v in value.lower().split("x"))
        if len(shape) != 3 or min(shape) <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Use positive MxNxK, e.g. 17x23x65") from None
    return shape


def route_for(shape):
    m, n, k = shape
    return "gemv" if m == 1 and n % 128 == 0 and k % 64 == 0 else "gemm"


def require_cuda(device_index):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; no benchmark measurements were produced")
    if not 0 <= device_index < torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device index: {device_index}")
    torch.cuda.set_device(device_index)
    load_cuda_extension(required=True)
    for name in ("lowbitlinear_ext::w4a16_gemv_warp_vec", "lowbitlinear_cute_w4a16::gemm"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA"):
            raise RuntimeError(f"Missing CUDA registration: {name}")
    return torch.device("cuda", device_index)


@contextmanager
def precision_policy():
    tf32 = torch.backends.cuda.matmul.allow_tf32
    reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced


def make_inputs(shape, device, seed):
    m, n, k = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator)
    weight = torch.randn(n, k, device=device, dtype=torch.float16, generator=generator)
    layer = QuantizedLinear.from_float(weight, bits=4, group_size=GROUP_SIZE)
    return x, layer


def decoded_weight(layer):
    q = unpack_int4(layer.packed_weight, original_k=layer.in_features)
    return dequantize_int4(q, layer.scales, layer.zeros, GROUP_SIZE)


def expected_output(x, layer, route):
    weight = decoded_weight(layer)
    if route == "gemm":
        weight = weight.half().float()  # Match the actual CuTe MMA operand policy.
    return (x.float() @ weight.T).half()


def packed_reference(x, layer, route):
    if route == "gemv":
        return linear_int4_reference(x, layer.packed_weight, layer.scales, layer.zeros, GROUP_SIZE)
    # Same packed format and CuTe rounding policy. Unpack/dequant remain in the timed call.
    return (x.float() @ decoded_weight(layer).half().float().T).half()


def check_output(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.isfinite(actual).all(), "Nonfinite operator output"
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    return (actual.float() - expected.float()).abs().max().item()


def capture(fn, expected, args):
    # Capture happens once, outside measurement, after side-stream warmup.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(args.graph_batch):
            output = fn()
    for _ in range(args.graph_warmup):
        graph.replay()
    torch.cuda.synchronize()
    check_output(output, expected)
    # Keep static output and graph-owned storage alive through every replay.
    return graph, output


def measure(functions, expected, args):
    captures = {}
    if args.mode == "graph":
        captures = {name: capture(fn, expected[name], args) for name, fn in functions.items()}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()  # Materialize event resources before timing.
    else:
        for fn in functions.values():
            for _ in range(args.warmup):
                fn()
        torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    orders = []
    names = list(functions)
    rng = random.Random(args.seed)
    for sample in range(args.repeats):
        # Rotations within shuffled cycles give each implementation every position.
        if sample % len(names) == 0:
            rng.shuffle(names)
        offset = sample % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        for name in order:
            if args.mode == "graph":
                start.record()
                captures[name][0].replay()
                end.record()
                end.synchronize()
                elapsed_us = start.elapsed_time(end) * 1000 / args.graph_batch
            else:
                # A completed single request: includes Python/dispatch/allocation and final sync.
                # The pre-sync is outside the timer so previous work cannot contaminate it.
                torch.cuda.synchronize()
                before = perf_counter_ns()
                output = functions[name]()
                torch.cuda.synchronize()
                elapsed_us = (perf_counter_ns() - before) / 1000
                del output
            if not math.isfinite(elapsed_us) or elapsed_us <= 0:
                raise RuntimeError(f"Invalid {name} latency: {elapsed_us} us")
            samples[name].append(elapsed_us)
    if captures:
        for name, (_, output) in captures.items():
            check_output(output, expected[name])
    return samples, orders


def summarize(samples):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * .9
    lower = int(position)
    p90 = ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)
    return {"median_us": statistics.median(samples), "p90_us": p90,
            "min_us": min(samples), "max_us": max(samples), "samples_us": samples}


def gpu_snapshot():
    fields = "index,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}  # Telemetry failure is explicit and never fabricates data.
    return {"returncode": result.returncode, "csv": result.stdout.strip(), "stderr": result.stderr.strip()}


def metadata(args):
    from lowbitlinear import _extension
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(),
        "device_index": torch.cuda.current_device(), "sm": list(torch.cuda.get_device_capability()),
        "gpu_memory_bytes": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        "torch": torch.__version__, "torch_build_cuda": torch.version.cuda,
        "extension": str(_extension._loaded_library), "dtype": "float16", "group_size": GROUP_SIZE,
        "tf32": False, "fp16_reduced_precision_reduction": False,
        "parameters": {key: str(value) if key == "json" and value is not None else value
                       for key, value in vars(args).items()},
        "timing": ("Synchronized single-call wall latency including Python, dispatch, allocation and final device sync; us"
                   if args.mode == "api" else
                   "CUDA Events around one graph replay / graph_batch; amortized GPU replay us, including graph scheduling; not API wall latency"),
        "weights_prepared_outside_measurement": True,
        "clock_locking": "not requested; laptop power/thermal state may vary",
        "gpu_before": gpu_snapshot(),
    }
