import pytest
import torch

from lowbitlinear import QuantizedLinear
from lowbitlinear.quantization import quantize_int4, dequantize_int4, pack_int4
from lowbitlinear.linear import linear_int4_reference


@pytest.mark.parametrize("m", [1, 16])
def test_linear_int4_reference(m):
    torch.manual_seed(42)
    x = torch.randn(m, 256, dtype=torch.float16)
    q, scales, zeros = quantize_int4(torch.randn(129, 256))
    packed = pack_int4(q)
    # Same quantized state; no intermediate FP16 dequantized weight rounding.
    restored = dequantize_int4(q, scales, zeros)
    expected = (x.float() @ restored.T).half()
    torch.testing.assert_close(linear_int4_reference(x, packed, scales, zeros), expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_quantized_linear_api(m, weight_dtype):
    layer = QuantizedLinear.from_float(torch.randn(129, 256, dtype=weight_dtype))
    x = torch.randn(m, 256, dtype=torch.float16)
    expected = linear_int4_reference(x, layer.packed_weight, layer.scales, layer.zeros)
    assert torch.equal(layer(x), expected)
    assert set(layer.state_dict()) == {"packed_weight", "scales", "zeros"}
    restored = QuantizedLinear.from_float(torch.zeros(129, 256, dtype=weight_dtype))
    restored.load_state_dict(layer.state_dict())
    assert torch.equal(restored(x), expected)


@pytest.mark.parametrize("bits", [2, 3, 8, 16, True])
def test_unsupported_bits(bits):
    with pytest.raises(ValueError, match="bits=4"):
        QuantizedLinear.from_float(torch.ones(2, 64), bits=bits)


@pytest.mark.parametrize("x,error", [
    (torch.ones(1, 64), TypeError),
    (torch.ones(1, 64, dtype=torch.bfloat16), TypeError),
    (torch.ones(64, dtype=torch.float16), ValueError),
    (torch.ones(1, 32, dtype=torch.float16), ValueError),
    (torch.ones(0, 64, dtype=torch.float16), ValueError),
])
def test_invalid_forward_input(x, error):
    layer = QuantizedLinear.from_float(torch.ones(2, 64))
    with pytest.raises(error):
        layer(x)


def test_invalid_constructor_metadata():
    with pytest.raises(ValueError, match="shape"):
        QuantizedLinear(torch.zeros(2, 8, dtype=torch.int32), torch.ones(2, 2), torch.zeros(2, 1, dtype=torch.int32))


def test_forward_device_mismatch(cuda_device):
    layer = QuantizedLinear.from_float(torch.ones(2, 64))
    with pytest.raises(ValueError, match="same device"):
        layer(torch.ones(1, 64, dtype=torch.float16, device=cuda_device))


def test_constructor_device_mismatch(cuda_device):
    with pytest.raises(ValueError, match="same device"):
        QuantizedLinear(torch.zeros(2, 8, dtype=torch.int32), torch.ones(2, 1, device=cuda_device), torch.zeros(2, 1, dtype=torch.int32))
