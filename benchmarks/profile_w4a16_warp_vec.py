import sys

import torch

from lowbitlinear.quantization import (
    quantize_int4,
    pack_int4,
)


def main():
    N, K = map(int, sys.argv[1:3])
    x = torch.randn(
        1, K, device="cuda", dtype=torch.float16
    )

    weight = torch.randn(
        N, K, device="cuda", dtype=torch.float16
    )

    q, scales, zeros = quantize_int4(weight, 64)
    packed = pack_int4(q)

    def run():
        return torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(
            x, packed, scales, zeros, 64
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
