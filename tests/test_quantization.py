"""Affine quantization, partial groups and the public INT4 storage format."""
import pytest
import torch

from lowbitlinear.quantization import quantize_int4, dequantize_int4, pack_int4, unpack_int4


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("k,group_size", [(1, 64), (7, 32), (65, 64), (129, 63), (256, 128)])
@pytest.mark.parametrize("kind", ["random", "zero", "positive", "negative", "constant"])
def test_affine_quantization(dtype, k, group_size, kind):
    weight = torch.randn(3, k, dtype=dtype)
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
    assert scales.shape == zeros.shape == (3, (k + group_size - 1) // group_size)
    assert scales.dtype == torch.float32 and zeros.dtype == torch.int32
    assert restored.dtype == torch.float32 and restored.device == weight.device
    for g, start in enumerate(range(0, k, group_size)):
        block = weight[:, start:start + group_size].float()
        lower = block.amin(1).minimum(torch.zeros(3))
        upper = block.amax(1).maximum(torch.zeros(3))
        scale = (upper - lower) / 15
        scale = torch.where(scale > 0, scale, 1.)
        zero = torch.round(-lower / scale).clamp(0, 15).int()
        expected_q = torch.round(block / scale[:, None] + zero[:, None]).clamp(0, 15).byte()
        torch.testing.assert_close(scales[:, g], scale, rtol=0, atol=0)
        assert torch.equal(zeros[:, g], zero)
        assert torch.equal(q[:, start:start + group_size], expected_q)
        expected_w = (expected_q.float() - zero[:, None]) * scale[:, None]
        torch.testing.assert_close(restored[:, start:start + group_size], expected_w, rtol=0, atol=0)
        # Rounded zero points can clip endpoints by up to one quantization step.
        assert torch.all((block - expected_w).abs() <= scale[:, None] + 1e-5)
    if kind == "zero":
        assert torch.equal(restored, weight.float())


def test_quantization_rounds_ties_to_even():
    weight = torch.tensor([[0., .5, 1.5, 2.5, 3.5, 15.]])
    q, scales, zeros = quantize_int4(weight, group_size=6)
    assert torch.equal(q, torch.tensor([[0, 0, 2, 2, 4, 15]], dtype=torch.uint8))
    assert scales.item() == 1 and zeros.item() == 0


def test_packing_bit_order_and_signed_word():
    q = torch.tensor([[1, 10, 3, 15, 5, 0, 9, 2, 0, 1, 2, 3, 4, 5, 6, 15]], dtype=torch.uint8)
    packed = pack_int4(q)
    assert packed.dtype == torch.int32 and packed.shape == (1, 2)
    assert packed.tolist() == [[0x2905F3A1, 0xF6543210 - 2**32]]
    assert torch.equal(unpack_int4(packed), q)


@pytest.mark.parametrize("k", [1, 7, 8, 9, 63, 64, 65, 129])
def test_pack_unpack_partial_words(k):
    q = (torch.arange(3 * k).reshape(3, k) % 16).byte()
    packed = pack_int4(q)
    assert packed.shape == (3, (k + 7) // 8)
    # Check raw words independently so two reciprocal bugs cannot cancel out.
    expected = []
    for row in q.tolist():
        words = []
        for start in range(0, k, 8):
            word = sum(value << (4 * j) for j, value in enumerate(row[start:start + 8]))
            words.append(word if word < 2**31 else word - 2**32)
        expected.append(words)
    assert packed.tolist() == expected
    assert torch.equal(unpack_int4(packed, original_k=k), q)
    padded = unpack_int4(packed)
    assert torch.equal(padded[:, :k], q)
    assert torch.count_nonzero(padded[:, k:]) == 0


@pytest.mark.parametrize("weight,error", [
    (None, TypeError),
    (torch.ones(2, 64, dtype=torch.int32), TypeError),
    (torch.ones(2, 64, dtype=torch.float64), TypeError),
    (torch.ones(64), ValueError),
    (torch.ones(0, 64), ValueError),
    (torch.ones(2, 0), ValueError),
    (torch.empty(2, 64, device="meta"), ValueError),
])
def test_invalid_weight(weight, error):
    with pytest.raises(error):
        quantize_int4(weight)


@pytest.mark.parametrize("group_size", [0, -1, 1.5, True, None])
def test_invalid_group_size(group_size):
    with pytest.raises(ValueError, match="group_size"):
        quantize_int4(torch.ones(2, 64), group_size)


@pytest.mark.parametrize("q,error", [
    (torch.ones(2, 8, dtype=torch.int32), TypeError),
    (torch.full((2, 8), 16, dtype=torch.uint8), ValueError),
    (torch.empty(0, 8, dtype=torch.uint8), ValueError),
    (torch.ones(8, dtype=torch.uint8), ValueError),
])
def test_pack_invalid(q, error):
    with pytest.raises(error):
        pack_int4(q)


@pytest.mark.parametrize("original_k", [0, -1, True, 1.5, 8, 17])
def test_unpack_invalid_original_k(original_k):
    with pytest.raises(ValueError, match="original_k"):
        unpack_int4(torch.zeros(2, 2, dtype=torch.int32), original_k)


def test_unpack_invalid_dtype():
    with pytest.raises(TypeError):
        unpack_int4(torch.ones(2, 8, dtype=torch.int64))


@pytest.mark.parametrize("field,error", [
    ("q_dtype", TypeError), ("q_rank", ValueError),
    ("scales_dtype", TypeError), ("zeros_dtype", TypeError),
    ("scales_shape", ValueError), ("zeros_shape", ValueError),
    ("group_size", ValueError),
])
def test_dequantization_invalid_metadata(field, error):
    q, scales, zeros = quantize_int4(torch.ones(2, 65))
    group_size = 64
    if field == "q_dtype": q = q.int()
    elif field == "q_rank": q = q.flatten()
    elif field == "scales_dtype": scales = scales.half()
    elif field == "zeros_dtype": zeros = zeros.long()
    elif field == "scales_shape": scales = scales[:, :1]
    elif field == "zeros_shape": zeros = zeros[:, :1]
    elif field == "group_size": group_size = 0
    with pytest.raises(error):
        dequantize_int4(q, scales, zeros, group_size)


@pytest.mark.parametrize("k,group_size", [(65, 64), (129, 63)])
def test_quantization_cuda_matches_cpu(cuda_device, k, group_size):
    # No extension needed: these are pure PyTorch operations.
    weight = torch.linspace(-2, 3, 3 * k, dtype=torch.float16).reshape(3, k)
    cpu = quantize_int4(weight, group_size)
    gpu = quantize_int4(weight.to(cuda_device), group_size)
    for actual, expected in zip(gpu, cpu):
        if expected.dtype == torch.float32:
            # CPU division vs CUDA reciprocal/multiply can differ by one FP32 ULP.
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-7, atol=1e-8)
        else:
            assert torch.equal(actual.cpu(), expected)
    packed = pack_int4(gpu[0])
    assert torch.equal(unpack_int4(packed, original_k=k), gpu[0])
    torch.testing.assert_close(
        dequantize_int4(*gpu, group_size).cpu(), dequantize_int4(*cpu, group_size),
        rtol=2e-7, atol=1e-7,
    )


def test_dequantization_device_mismatch(cuda_device):
    q, scales, zeros = quantize_int4(torch.ones(2, 65))
    with pytest.raises(ValueError, match="same device"):
        dequantize_int4(q.to(cuda_device), scales, zeros)


def test_known_affine_groups_and_partial_tail():
    weight = torch.tensor([[-2., 0., 3., 1., 2., 3., 0.],
                           [-3., -2., -1., 0., 0., 0., 2.]])
    q, scales, zeros = quantize_int4(weight, group_size=3)
    assert q.tolist() == [[0, 6, 15, 5, 10, 15, 0], [0, 5, 10, 0, 0, 0, 15]]
    assert zeros.tolist() == [[6, 0, 0], [15, 0, 0]]
    torch.testing.assert_close(scales, torch.tensor([[1 / 3, .2, 1.], [.2, 1., 2 / 15]]), rtol=0, atol=0)
    torch.testing.assert_close(dequantize_int4(q, scales, zeros, 3), weight, rtol=0, atol=1e-6)
