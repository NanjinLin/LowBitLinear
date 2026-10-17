import torch
import lowbitlinear

from lowbitlinear.quantization import (
    quantize_int4,
    pack_int4,
    unpack_int4,
    dequantize_int4,
)


torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False

SHAPES = [
    (32, 64, 512),
    (64, 128, 512),
    (128, 256, 512),
    (256, 256, 512),
    (512, 512, 512),
]


def benchmark_graph(fn, repeats=100, batch=16):
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(warmup_stream):
        for _ in range(20):
            fn()

    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for _ in range(batch):
            fn()

    for _ in range(10):
        graph.replay()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(repeats):
        graph.replay()

    end.record()
    end.synchronize()

    return start.elapsed_time(end) * 1000 / (repeats * batch)

for M, N, K in SHAPES:
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)

    q, scales, zeros = quantize_int4(w, 64)
    packed_w = pack_int4(q)

    def fused_baseline():
        return torch.ops.lowbitlinear_ext.w4a16_gemm_padding(
            x, packed_w, scales, zeros
        )

    def fused_regtile():
        return torch.ops.lowbitlinear_ext.w4a16_gemm(
            x, packed_w, scales, zeros
        )

    def unfused_reference():
        q_unpacked = unpack_int4(packed_w)
        w_dequant = dequantize_int4(q_unpacked, scales, zeros, 64)
        return (x.float() @ w_dequant.T).half()

    w_dequant = dequantize_int4(
        unpack_int4(packed_w), scales, zeros, 64
    )

    def predequant_gemm():
        return (x.float() @ w_dequant.T).half()

    t_baseline = benchmark_graph(fused_baseline)
    t_regtile = benchmark_graph(fused_regtile)
    t_unfused = benchmark_graph(unfused_reference)
    t_predequant = benchmark_graph(predequant_gemm)

    print(f"\nM={M}, N={N}, K={K}")
    print(f"Baseline GEMM:    {t_baseline:.3f} us")
    print(f"Register Tiling:  {t_regtile:.3f} us")
    print(f"Regtile Speedup:  {t_baseline / t_regtile:.3f}x")
    print(f"Unfused Ref:      {t_unfused:.3f} us")
    print(f"Predequant GEMM:  {t_predequant:.3f} us")