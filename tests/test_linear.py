import torch
import torch.nn.functional as F

from lowbitlinear.quantization import (
    quantize_int4,
    dequantize_int4,
    pack_int4,
)
from lowbitlinear.linear import linear_int4_reference


def test_linear_int4():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    M, N, K = 16, 128, 256

    x = torch.randn(M, K, device=device, dtype=torch.float16)
    weight = torch.randn(N, K, device=device, dtype=torch.float16)

    # Quantization + Packing
    q, scales, zeros = quantize_int4(weight, group_size=64)
    packed = pack_int4(q)

    # 我们自己的 Linear
    y_custom = linear_int4_reference(
        x, packed, scales, zeros, group_size=64
    )

    # 用未打包的量化权重构建参考结果
    restored = dequantize_int4(q, scales, zeros, group_size=64)
    y_reference = F.linear(x, restored.to(x.dtype))

    # 原始 FP16 Linear（用于观察量化误差）
    y_original = F.linear(x, weight)

    assert y_custom.shape == (M, N)
    assert torch.allclose(y_custom, y_reference, atol=1e-3, rtol=1e-3)

    print("Output Shape:", y_custom.shape)
    print("Implementation Error:",
          (y_custom - y_reference).abs().max().item())
    print("Quantization Error:",
          (y_custom - y_original).abs().max().item())

from lowbitlinear import QuantizedLinear


def test_quantized_linear_api():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight = torch.randn(128, 256, device=device, dtype=torch.float16)
    x = torch.randn(16, 256, device=device, dtype=torch.float16)

    layer = QuantizedLinear.from_float(
        weight,
        bits=4,
        group_size=64
    )

    y = layer(x)

    y_ref = linear_int4_reference(
        x,
        layer.packed_weight,
        layer.scales,
        layer.zeros,
        group_size=64
    )

    assert y.shape == (16, 128)
    assert torch.equal(y, y_ref)

    print("API Output Shape:", y.shape)
    print("API Correctness: PASS")