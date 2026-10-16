import pytest
import torch

from lowbitlinear.linear import linear_int4_reference
from lowbitlinear.quantization import pack_int4

ATOL, RTOL = 2e-3, 1e-3


@pytest.mark.parametrize("op_name", ["w4a16_gemv_warp", "w4a16_gemv_warp_vec"])
@pytest.mark.parametrize("n,k", [(1, 64), (3, 128), (127, 256), (129, 256), (1024, 1024), (257, 4096)])
@pytest.mark.parametrize("seed", [42, 123])
def test_warp_gemv(op_name, n, k, seed, gemv_inputs):
    args = gemv_inputs(n, k, seed)
    actual = getattr(torch.ops.lowbitlinear_ext, op_name)(*args, 64)
    expected = linear_int4_reference(*args)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("op_name", ["w4a16_gemv", "w4a16_gemv_warp", "w4a16_gemv_warp_vec"])
def test_extreme_nibbles_and_zero_points(op_name, cuda_device):
    n, k = 5, 256
    q = (torch.arange(n * k, device=cuda_device).reshape(n, k) % 16).byte()
    packed = pack_int4(q)
    scales = torch.full((n, k // 64), 0.125, device=cuda_device)
    zeros = (torch.arange(n * (k // 64), device=cuda_device).reshape(n, -1) % 2 * 15).int()
    x = torch.linspace(-1, 1, k, dtype=torch.float16, device=cuda_device)[None, :]
    actual = getattr(torch.ops.lowbitlinear_ext, op_name)(x, packed, scales, zeros, 64)
    torch.testing.assert_close(actual, linear_int4_reference(x, packed, scales, zeros), atol=ATOL, rtol=RTOL)


def test_vec_rejects_misaligned_contiguous_input(gemv_inputs):
    x, packed, scales, zeros = gemv_inputs()
    storage = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)
    misaligned = storage[1:].view_as(x).copy_(x)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 16 != 0
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(misaligned, packed, scales, zeros, 64)


def test_vec_uses_current_stream(gemv_inputs):
    args = gemv_inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = args[0] * 2
        actual = torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(x, *args[1:], 64)
        consumed = actual + 1  # Dependent work on this stream must see the kernel output.
    torch.cuda.current_stream().wait_stream(stream)
    expected = linear_int4_reference(x, *args[1:]) + 1
    torch.testing.assert_close(consumed, expected, atol=ATOL, rtol=RTOL)
