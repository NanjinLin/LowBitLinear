import pytest
import torch
from lowbitlinear.quantization import quantize_int4, dequantize_int4, pack_int4, unpack_int4


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("kind", ["random", "zero", "positive", "negative", "constant"])
def test_int4_quantization(dtype, group_size, kind):
    torch.manual_seed(42)
    weight = torch.randn(3, 256, dtype=dtype)
    if kind == "zero":
        weight.zero_()
    elif kind == "positive":
        weight.abs_()
    elif kind == "negative":
        weight = -weight.abs()
    elif kind == "constant":
        weight.fill_(2.5)
    q, scales, zeros = quantize_int4(weight, group_size)
    restored = dequantize_int4(q, scales, zeros, group_size)
    assert q.shape == weight.shape and q.dtype == torch.uint8
    assert scales.shape == zeros.shape == (3, 256 // group_size)
    assert scales.dtype == torch.float32 and zeros.dtype == torch.int32
    assert torch.all(scales > 0) and torch.isfinite(scales).all()
    assert q.min() >= 0 and q.max() <= 15
    assert zeros.min() >= 0 and zeros.max() <= 15
    # Rounding/clamping at an affine endpoint can cost up to one scale step.
    assert torch.all((weight.float() - restored).abs() <= scales.repeat_interleave(group_size, 1) + 1e-5)
    if kind == "zero":
        assert torch.equal(restored, weight.float())
    torch.testing.assert_close(
        restored,
        (q.reshape(3, -1, group_size).float() - zeros[..., None])
        .mul(scales[..., None]).reshape_as(restored),
        rtol=0, atol=0,
    )


def test_int4_packing_bit_order_and_signed_word():
    q = torch.tensor([[1, 10, 3, 15, 5, 0, 9, 2, 0, 1, 2, 3, 4, 5, 6, 15]], dtype=torch.uint8)
    packed = pack_int4(q)
    assert packed.dtype == torch.int32 and packed.shape == (1, 2)
    assert packed[0, 0].item() == 0x2905F3A1
    assert packed[0, 1].item() == 0xF6543210 - 2**32
    assert torch.equal(unpack_int4(packed), q)


@pytest.mark.parametrize("n,k", [(1, 8), (3, 64), (127, 256)])
def test_int4_pack_unpack(n, k):
    q = (torch.arange(n * k).reshape(n, k) % 16).to(torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)


@pytest.mark.parametrize("weight,error", [
    (torch.ones(2, 64, dtype=torch.int32), TypeError),
    (torch.ones(2, 64, dtype=torch.float64), TypeError),
    (torch.ones(64), ValueError),
    (torch.ones(0, 64), ValueError),
    (torch.ones(2, 65), ValueError),
    (torch.empty(2, 64, device="meta"), ValueError),
])
def test_quantization_invalid_weight(weight, error):
    with pytest.raises(error):
        quantize_int4(weight)


@pytest.mark.parametrize("group_size", [0, -1, 63, 1.5, True])
def test_invalid_group_size(group_size):
    with pytest.raises(ValueError, match="group_size"):
        quantize_int4(torch.ones(2, 64), group_size)


@pytest.mark.parametrize("q,error", [
    (torch.ones(2, 8, dtype=torch.int32), TypeError),
    (torch.ones(2, 9, dtype=torch.uint8), ValueError),
    (torch.full((2, 8), 16, dtype=torch.uint8), ValueError),
])
def test_pack_invalid(q, error):
    with pytest.raises(error):
        pack_int4(q)


def test_unpack_invalid_dtype():
    with pytest.raises(TypeError):
        unpack_int4(torch.ones(2, 8, dtype=torch.int64))


@pytest.mark.parametrize("field", ["scales_dtype", "zeros_dtype", "scales_shape", "zeros_shape"])
def test_dequantization_invalid_metadata(field):
    q, scales, zeros = quantize_int4(torch.ones(2, 64))
    if field == "scales_dtype":
        scales = scales.half()
    elif field == "zeros_dtype":
        zeros = zeros.long()
    elif field == "scales_shape":
        scales = scales.repeat(1, 2)
    else:
        zeros = zeros.repeat(1, 2)
    with pytest.raises((TypeError, ValueError)):
        dequantize_int4(q, scales, zeros)


def test_quantization_cuda_matches_cpu(cuda_device):
    weight = torch.linspace(-2, 3, 3 * 256, dtype=torch.float16).reshape(3, 256)
    cpu = quantize_int4(weight)
    gpu = quantize_int4(weight.to(cuda_device))
    for actual, expected in zip(gpu, cpu):
        # CPU division vs CUDA reciprocal/multiply can differ by one FP32 ULP.
        if expected.dtype == torch.float32:
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-7, atol=1e-8)
        else:
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    packed = pack_int4(gpu[0])
    assert torch.equal(unpack_int4(packed), gpu[0])
