"""Installed Warp Vec GEMV: FP32 dequantization and accumulation, FP16 output."""
import pytest
import torch

from lowbitlinear.quantization import pack_int4

ATOL, RTOL = 2e-3, 1e-3


@pytest.mark.parametrize("n,k", [(1, 64), (3, 128), (127, 256), (128, 256), (129, 256), (1024, 1024), (257, 4096)])
@pytest.mark.parametrize("seed", [42, 123])
def test_warp_vec_gemv(n, k, seed, gemv_inputs, packed_reference):
    args = gemv_inputs(n, k, seed)
    actual = torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(*args, 64)
    assert actual.shape == (1, n) and actual.dtype == torch.float16
    assert actual.device == args[0].device
    # FP32 reduction order and final FP16 rounding only; no quantization-error allowance.
    torch.testing.assert_close(actual, packed_reference(*args), atol=ATOL, rtol=RTOL)


def test_extreme_nibbles_and_zero_points(cuda_ops, packed_reference):
    n, k = 5, 256
    q = (torch.arange(n * k, device=cuda_ops).reshape(n, k) % 16).byte()
    packed = pack_int4(q)
    scales = torch.full((n, k // 64), 0.125, device=cuda_ops)
    zeros = (torch.arange(n * (k // 64), device=cuda_ops).reshape(n, -1) % 2 * 15).int()
    x = torch.linspace(-1, 1, k, dtype=torch.float16, device=cuda_ops)[None, :]
    assert torch.any(packed < 0)  # signed INT32 must be read as unsigned nibbles
    args = x, packed, scales, zeros
    actual = torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(*args, 64)
    torch.testing.assert_close(actual, packed_reference(*args), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("bad,match", [
    ("x_dtype", "FP16"), ("x_rank", "shape"), ("m", "shape"),
    ("k_tail", "divisible"), ("empty_k", "divisible"),
    ("packed_dtype", "int32"), ("packed_rank", "2D"),
    ("packed_shape", "packed weight shape"), ("scales_dtype", "FP32"),
    ("scales_shape", "scales or zeros shape"), ("zeros_dtype", "int32"),
    ("zeros_shape", "scales or zeros shape"), ("cpu_metadata", "CUDA"),
    ("group_size", "group_size=64"), ("empty_n", "Invalid N or K"),
    ("noncontiguous_x", "contiguous"), ("noncontiguous_packed", "contiguous"),
    ("noncontiguous_scales", "contiguous"), ("noncontiguous_zeros", "contiguous"),
])
def test_rejects_invalid_inputs(bad, match, gemv_inputs):
    args = list(gemv_inputs())
    group_size = 64
    if bad == "x_dtype": args[0] = args[0].float()
    elif bad == "x_rank": args[0] = args[0].squeeze(0)
    elif bad == "m": args[0] = args[0].repeat(2, 1)
    elif bad == "k_tail": args[0] = args[0][:, :65].contiguous()
    elif bad == "empty_k": args[0] = args[0][:, :0]
    elif bad == "packed_dtype": args[1] = args[1].long()
    elif bad == "packed_rank": args[1] = args[1].flatten()
    elif bad == "packed_shape": args[1] = args[1][:, :-1].contiguous()
    elif bad == "scales_dtype": args[2] = args[2].half()
    elif bad == "scales_shape": args[2] = args[2][:, :-1].contiguous()
    elif bad == "zeros_dtype": args[3] = args[3].long()
    elif bad == "zeros_shape": args[3] = args[3][:, :-1].contiguous()
    elif bad == "cpu_metadata": args[2] = args[2].cpu()
    elif bad == "group_size": group_size = 32
    elif bad == "empty_n": args[1:] = [t[:0] for t in args[1:]]
    elif bad.startswith("noncontiguous_"):
        index = {"noncontiguous_x": 0, "noncontiguous_packed": 1,
                 "noncontiguous_scales": 2, "noncontiguous_zeros": 3}[bad]
        tensor = args[index]
        args[index] = torch.zeros(tensor.shape[0], tensor.shape[1] * 2,
                                 device=tensor.device, dtype=tensor.dtype)[:, ::2]
        assert not args[index].is_contiguous()
    with pytest.raises(RuntimeError, match=match):
        torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(*args, group_size)


def test_rejects_cpu_input(gemv_inputs):
    args = [t.cpu() for t in gemv_inputs()]
    with pytest.raises(NotImplementedError, match="CPU"):
        torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(*args, 64)


def test_rejects_misaligned_contiguous_input(gemv_inputs):
    x, packed, scales, zeros = gemv_inputs()
    storage = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)
    misaligned = storage[1:].view_as(x).copy_(x)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 16 != 0
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(misaligned, packed, scales, zeros, 64)


def test_uses_current_stream(gemv_inputs, packed_reference):
    args = gemv_inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = args[0] * 2
        actual = torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(x, *args[1:], 64)
        consumed = actual + 1
    torch.cuda.current_stream().wait_stream(stream)
    expected = packed_reference(x, *args[1:]) + 1
    torch.testing.assert_close(consumed, expected, atol=ATOL, rtol=RTOL)
