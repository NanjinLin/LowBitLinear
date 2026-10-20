"""Formal CuTe W4A16 operator, including masked MNK tails and empty dimensions."""
import pytest
import torch

from lowbitlinear.quantization import pack_int4, quantize_int4

ATOL, RTOL = 2e-3, 1e-3
SHAPES = [
    (1, 1, 1), (4, 9, 7), (4, 8, 8), (4, 8, 63), (32, 32, 64),
    (17, 23, 65), (5, 9, 70), (6, 10, 71), (31, 33, 127),
    (6, 10, 128), (33, 37, 129), (64, 65, 130), (79, 70, 192),
    (32, 64, 512), (7, 35, 4096),
    # Exercise both sides of the 64x64 CTA boundary, including K tails.
    (63, 63, 63), (65, 63, 64), (63, 65, 65), (65, 65, 129),
    (127, 129, 192),
]


@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("real_quant", [False, True], ids=["synthetic-int4", "real-quantization"])
def test_gemm(m, n, k, real_quant, cuda_ops, packed_reference):
    x = torch.randn(m, k, device=cuda_ops, dtype=torch.float16)
    if real_quant:
        # No manual weight padding: exercise the actual quantization pipeline.
        q, scales, zeros = quantize_int4(torch.randn(n, k, device=cuda_ops, dtype=torch.float16))
    else:
        q = torch.randint(0, 16, (n, k), device=cuda_ops, dtype=torch.uint8)
        q[:, 7::8] = 15
        groups = (k + 63) // 64
        scales = torch.rand(n, groups, device=cuda_ops) * .08 + .01
        zeros = torch.randint(0, 16, (n, groups), device=cuda_ops, dtype=torch.int32)
    args = x, pack_int4(q), scales, zeros
    expected = packed_reference(*args, mma=True)
    actual = torch.ops.lowbitlinear_cute_w4a16.gemm(*args)
    assert actual.shape == (m, n) and actual.dtype == torch.float16
    assert actual.device == x.device and torch.isfinite(actual).all()
    # FP16 MMA operands match the oracle; tolerance covers reduction/FP16 output rounding.
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("m,n,k", [(32, 32, 64), (17, 23, 129), (65, 65, 129)])
def test_exact_multistage_accumulation(m, n, k, cuda_ops):
    x = torch.ones(m, k, device=cuda_ops, dtype=torch.float16)
    packed = pack_int4(torch.full((n, k), 5, device=cuda_ops, dtype=torch.uint8))
    scales = torch.full((n, (k + 63) // 64), .25, device=cuda_ops)
    zeros = torch.full_like(scales, 3, dtype=torch.int32)
    actual = torch.ops.lowbitlinear_cute_w4a16.gemm(x, packed, scales, zeros)
    torch.testing.assert_close(actual, torch.full_like(actual, k * .5), rtol=0, atol=0)


@pytest.mark.parametrize("m,n,k", [(33, 65, 129), (65, 129, 129)])
def test_one_hot_mapping(m, n, k, cuda_ops):
    x = (torch.arange(m * k, device=cuda_ops).reshape(m, k) % 101).half()
    q = torch.zeros(n, k, dtype=torch.uint8, device=cuda_ops)
    positions = (torch.arange(n, device=cuda_ops) * 2) % k
    q[torch.arange(n, device=cuda_ops), positions] = 1
    scales = torch.ones(n, (k + 63) // 64, device=cuda_ops)
    zeros = torch.zeros_like(scales, dtype=torch.int32)
    actual = torch.ops.lowbitlinear_cute_w4a16.gemm(x, pack_int4(q), scales, zeros)
    torch.testing.assert_close(actual, x[:, positions], rtol=0, atol=0)


def test_fp16_dequantization_policy(cuda_ops, packed_reference):
    # This distinguishes the MMA oracle from the FP32-dequant GEMV/reference policy.
    x = torch.ones(1, 65, device=cuda_ops, dtype=torch.float16)
    q = torch.ones(1, 65, device=cuda_ops, dtype=torch.uint8)
    scales = torch.tensor([[1.0004, 1.03]], device=cuda_ops)
    zeros = torch.zeros(1, 2, device=cuda_ops, dtype=torch.int32)
    args = x, pack_int4(q), scales, zeros
    expected = packed_reference(*args, mma=True)
    assert not torch.equal(expected, packed_reference(*args))
    torch.testing.assert_close(torch.ops.lowbitlinear_cute_w4a16.gemm(*args), expected, rtol=0, atol=0)


@pytest.mark.parametrize("m,n,k", [(32, 32, 0), (0, 32, 65), (32, 0, 65), (0, 0, 0)])
def test_empty_dimensions(m, n, k, cuda_ops):
    x = torch.empty(m, k, device=cuda_ops, dtype=torch.float16)
    packed = torch.empty(n, (k + 7) // 8, device=cuda_ops, dtype=torch.int32)
    scales = torch.empty(n, (k + 63) // 64, device=cuda_ops)
    zeros = torch.empty_like(scales, dtype=torch.int32)
    actual = torch.ops.lowbitlinear_cute_w4a16.gemm(x, packed, scales, zeros)
    assert actual.shape == (m, n) and actual.dtype == x.dtype and actual.device == x.device
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("index", range(4), ids=["x", "packed", "scales", "zeros"])
@pytest.mark.parametrize("bad", ["dtype", "rank", "noncontiguous", "cpu"])
def test_invalid_tensor_contract(index, bad, gemv_inputs):
    args = list(gemv_inputs())
    tensor = args[index]
    if bad == "dtype":
        args[index] = tensor.double()
        match = ["float16", "int32", "float32", "int32"][index]
    elif bad == "rank":
        args[index] = tensor.flatten()
        match = "2D"
    elif bad == "noncontiguous":
        args[index] = torch.zeros(tensor.shape[0], tensor.shape[1] * 2,
                                 device=tensor.device, dtype=tensor.dtype)[:, ::2]
        assert not args[index].is_contiguous()
        match = "contiguous"
    else:
        args[index] = tensor.cpu()
        match = "CUDA"
    with pytest.raises(RuntimeError, match=match):
        torch.ops.lowbitlinear_cute_w4a16.gemm(*args)


@pytest.mark.parametrize("index,match", [(1, "packed_w must have shape"), (2, "scales must have shape"), (3, "zeros must have shape")])
def test_invalid_metadata_shape(index, match, gemv_inputs):
    args = list(gemv_inputs())
    args[index] = args[index][:, :-1].contiguous()
    with pytest.raises(RuntimeError, match=match):
        torch.ops.lowbitlinear_cute_w4a16.gemm(*args)


def test_uses_current_stream(gemv_inputs, packed_reference):
    args = gemv_inputs(n=37)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = args[0] * 2
        actual = torch.ops.lowbitlinear_cute_w4a16.gemm(x, *args[1:])
        consumed = actual + 1
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(consumed, packed_reference(x, *args[1:], mma=True) + 1, atol=ATOL, rtol=RTOL)
