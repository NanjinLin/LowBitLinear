"""Profile one installed W4A16 operator; setup/checks/warmup are outside the profiler range."""
import argparse

import torch

if __package__:
    from ._common import parse_shape, require_cuda, precision_policy, make_inputs, expected_output, check_output
else:
    from _common import parse_shape, require_cuda, precision_policy, make_inputs, expected_output, check_output


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--operator", choices=("gemv", "gemm"), required=True)
    cli.add_argument("--shape", type=parse_shape, required=True, help="MxNxK; raw GEMV allows any positive N")
    cli.add_argument("--warmup", type=int, default=50)
    cli.add_argument("--launches", type=int, default=1, help="Target calls inside cudaProfilerStart/Stop")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--device", type=int, default=0)
    args = cli.parse_args(argv)
    if min(args.warmup, args.launches) < 1:
        cli.error("warmup and launches must be positive")
    m, n, k = args.shape
    if args.operator == "gemv" and (m != 1 or k % 64 != 0):
        cli.error("Raw Warp Vec GEMV requires M=1 and K%64=0")
    device = require_cuda(args.device)
    with precision_policy():
        x, layer = make_inputs(args.shape, device, args.seed)
        tensors = x, layer.packed_weight, layer.scales, layer.zeros
        if args.operator == "gemv":
            def run():
                return torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(*tensors, 64)
            name = "w4a16_gemv_warp_vec"
        else:
            def run():
                return torch.ops.lowbitlinear_cute_w4a16.gemm(*tensors)
            name = "cute_w4a16_gemm_kernel"
        expected = expected_output(x, layer, args.operator)
        error = check_output(run(), expected)
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
        try:
            for _ in range(args.launches):
                output = run()
            torch.cuda.synchronize()
        finally:
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
        check_output(output, expected)
        print(f"PASS {name}: shape={m}x{n}x{k}, dtype=float16, group_size=64, "
              f"seed={args.seed}, warmup={args.warmup}, profiled_launches={args.launches}, "
              f"max_abs_error={error:.6g}; GPU={torch.cuda.get_device_name()}; "
              f"torch={torch.__version__}; torch_build_cuda={torch.version.cuda}")
        print("No latency is reported here; profiler instrumentation changes execution time.")


if __name__ == "__main__":
    main()
