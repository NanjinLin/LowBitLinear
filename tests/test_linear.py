"""Python API, CPU/reference semantics and state validation."""
import pytest
import torch

from lowbitlinear import QuantizedLinear
from lowbitlinear.quantization import quantize_int4, pack_int4
from lowbitlinear.linear import linear_int4_reference


@pytest.mark.parametrize("m,n,k,group_size", [(1, 129, 256, 64), (16, 5, 65, 64), (5, 9, 70, 32), (9, 17, 129, 63), (1, 1, 1, 1)])
def test_linear_int4_reference(m, n, k, group_size, packed_reference):
    x = torch.randn(m, k, dtype=torch.float16)
    q, scales, zeros = quantize_int4(torch.randn(n, k), group_size)
    args = x, pack_int4(q), scales, zeros
    actual = linear_int4_reference(*args, group_size)
    torch.testing.assert_close(actual, packed_reference(*args, group_size), rtol=0, atol=0)


@pytest.mark.parametrize("m,n,k,group_size", [(1, 129, 256, 64), (5, 9, 70, 64), (9, 17, 129, 63)])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_api_state_dict_roundtrip(m, n, k, group_size, weight_dtype, packed_reference):
    layer = QuantizedLinear.from_float(torch.randn(n, k, dtype=weight_dtype, requires_grad=True), group_size=group_size)
    x = torch.randn(m, k, dtype=torch.float16)
    expected = packed_reference(x, layer.packed_weight, layer.scales, layer.zeros, group_size)
    assert layer.in_features == k and layer.out_features == n and layer.group_size == group_size
    assert layer(x).dtype == torch.float16 and layer(x).shape == (m, n)
    assert torch.equal(layer(x), expected)
    assert set(layer.state_dict()) == {"packed_weight", "scales", "zeros"}
    assert list(layer.parameters()) == []
    assert all(not buffer.requires_grad for buffer in layer.buffers())
    # Shape/group attributes are constructor metadata, not state_dict buffers.
    restored = QuantizedLinear.from_float(torch.zeros(n, k), group_size=group_size)
    restored.load_state_dict(layer.state_dict())
    assert torch.equal(restored(x), expected)


def test_direct_constructor_with_partial_word(packed_reference):
    q, scales, zeros = quantize_int4(torch.randn(9, 70))
    packed = pack_int4(q)
    layer = QuantizedLinear(packed, scales, zeros, in_features=70)
    x = torch.randn(5, 70, dtype=torch.float16)
    assert torch.equal(layer(x), packed_reference(x, packed, scales, zeros))


@pytest.mark.parametrize("bits", [2, 3, 8, 16, True])
def test_unsupported_bits(bits):
    with pytest.raises(ValueError, match="bits=4"):
        QuantizedLinear.from_float(torch.ones(2, 64), bits=bits)


@pytest.mark.parametrize("x,error,match", [
    (None, TypeError, "Tensor"),
    (torch.ones(1, 64), TypeError, "dtype"),
    (torch.ones(1, 64, dtype=torch.bfloat16), TypeError, "dtype"),
    (torch.ones(64, dtype=torch.float16), ValueError, "2D"),
    (torch.ones(1, 32, dtype=torch.float16), ValueError, "incompatible K"),
    (torch.ones(0, 64, dtype=torch.float16), ValueError, "nonempty"),
    (torch.ones(1, 128, dtype=torch.float16)[:, ::2], ValueError, "contiguous"),
])
def test_invalid_forward_input(x, error, match):
    layer = QuantizedLinear.from_float(torch.ones(2, 64))
    with pytest.raises(error, match=match):
        layer(x)


@pytest.mark.parametrize("field,error,match", [
    ("packed_dtype", TypeError, "dtype"), ("packed_shape", ValueError, "shape"),
    ("scales_dtype", TypeError, "dtype"), ("zeros_dtype", TypeError, "dtype"),
    ("scales_shape", ValueError, "shape"), ("zeros_shape", ValueError, "shape"),
    ("group_size", ValueError, "group_size"),
])
def test_invalid_constructor_metadata(field, error, match):
    packed = torch.zeros(2, 8, dtype=torch.int32)
    scales = torch.ones(2, 1)
    zeros = torch.zeros(2, 1, dtype=torch.int32)
    group_size = 64
    if field == "packed_dtype": packed = packed.long()
    elif field == "packed_shape": packed = packed[:, :-1]
    elif field == "scales_dtype": scales = scales.half()
    elif field == "zeros_dtype": zeros = zeros.long()
    elif field == "scales_shape": scales = scales.repeat(1, 2)
    elif field == "zeros_shape": zeros = zeros.repeat(1, 2)
    elif field == "group_size": group_size = 0
    with pytest.raises(error, match=match):
        QuantizedLinear(packed, scales, zeros, group_size, in_features=64)


@pytest.mark.parametrize("in_features", [0, -1, True, 1.5, None, 65])
def test_invalid_in_features(in_features):
    with pytest.raises(ValueError, match="in_features|shape"):
        QuantizedLinear(torch.zeros(2, 8, dtype=torch.int32), torch.ones(2, 1),
                        torch.zeros(2, 1, dtype=torch.int32), in_features=in_features)


def test_forward_revalidates_changed_buffers():
    layer = QuantizedLinear.from_float(torch.ones(2, 64))
    layer.scales = torch.ones(2, 2)
    with pytest.raises(ValueError, match="shape"):
        layer(torch.ones(1, 64, dtype=torch.float16))


def test_forward_device_mismatch(cuda_device):
    layer = QuantizedLinear.from_float(torch.ones(2, 64))
    with pytest.raises(ValueError, match="same device"):
        layer(torch.ones(1, 64, dtype=torch.float16, device=cuda_device))


def test_constructor_device_mismatch(cuda_device):
    with pytest.raises(ValueError, match="same device"):
        QuantizedLinear(torch.zeros(2, 8, dtype=torch.int32), torch.ones(2, 1, device=cuda_device),
                        torch.zeros(2, 1, dtype=torch.int32), in_features=64)


def test_module_to_cuda_moves_all_buffers(cuda_ops, packed_reference):
    layer = QuantizedLinear.from_float(torch.randn(9, 70)).to(cuda_ops)
    assert all(buffer.device == cuda_ops for buffer in layer.buffers())
    x = torch.randn(5, 70, device=cuda_ops, dtype=torch.float16)
    expected = packed_reference(x, layer.packed_weight, layer.scales, layer.zeros, mma=True)
    torch.testing.assert_close(layer(x), expected, atol=2e-3, rtol=1e-3)
