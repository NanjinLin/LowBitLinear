import sys

import torch

from lowbitlinear.quantization import (
    quantize_int4,
    pack_int4,
)


def main():
    M, N, K = map(int, sys.argv[1:4])
    x = torch.randn(
        M, K, device="cuda", dtype=torch.float16
    )

    weight = torch.randn(
        N, K, device="cuda", dtype=torch.float16
    )

    q, scales, zeros = quantize_int4(weight, 64)
    packed = pack_int4(q)

    def run():
        return torch.ops.lowbitlinear_ext.w4a16_gemm(
            x, packed, scales, zeros
        )

    with torch.inference_mode():
        for _ in range(20):
            run()

        torch.cuda.synchronize()

        # Target kernel for profiling
        run()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
