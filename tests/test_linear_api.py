import torch

from lowbitlinear import QuantizedLinear
from lowbitlinear.quantization import (
    unpack_int4,
    dequantize_int4,
)

torch.manual_seed(42)

shapes = [
    (1, 9, 70),
    (5, 9, 70),
    (6, 10, 128),
    (9, 17, 129),
]

for M, N, K in shapes:
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)

    layer = QuantizedLinear.from_float(w, group_size=64)

    y = layer(x)

    q = unpack_int4(layer.packed_weight, original_k=K)
    w_ref = dequantize_int4(q, layer.scales, layer.zeros, 64)
    y_ref = (x.float() @ w_ref.T).half()

    assert layer.in_features == K
    assert layer.out_features == N

    torch.testing.assert_close(
        y, y_ref,
        atol=2e-2,
        rtol=1e-2
    )

    print(f"M={M}, N={N}, K={K}: PASS")