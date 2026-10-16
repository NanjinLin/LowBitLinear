import pytest
import torch

from lowbitlinear.linear import linear_int4_reference

# FP32 accumulation can differ with serial vs warp/tree reduction. FP16 rounding
# adds ~1e-3 relative error between adjacent outputs. The 2e-3 absolute floor
# covers cancellation for unit-scale random inputs, K<=4096; never weight quantization error.
ATOL, RTOL = 2e-3, 1e-3
OPS = ("w4a16_gemv", "w4a16_gemv_warp", "w4a16_gemv_warp_vec")


@pytest.mark.parametrize("n,k", [(1, 64), (3, 128), (127, 256), (129, 256), (1024, 1024), (257, 4096)])
def test_w4a16_gemv_naive(n, k, gemv_inputs):
    x, packed, scales, zeros = gemv_inputs(n, k)
    actual = torch.ops.lowbitlinear_ext.w4a16_gemv(x, packed, scales, zeros, 64)
    expected = linear_int4_reference(x, packed, scales, zeros)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("bad", ["x_dtype", "x_rank", "m", "k", "packed_dtype", "packed_rank", "packed_shape", "scales_dtype", "scales_shape", "zeros_dtype", "zeros_shape", "cpu_metadata", "noncontiguous", "group_size", "empty_n"])
def test_cuda_rejects_invalid_inputs(op_name, bad, gemv_inputs):
    x, packed, scales, zeros = gemv_inputs()
    group_size = 64
    if bad == "x_dtype": x = x.float()
    elif bad == "x_rank": x = x.squeeze(0)
    elif bad == "m": x = x.repeat(2, 1)
    elif bad == "k": x = x[:, :128].contiguous()
    elif bad == "packed_dtype": packed = packed.long()
    elif bad == "packed_rank": packed = packed.flatten()
    elif bad == "packed_shape": packed = packed[:, :-1].contiguous()
    elif bad == "scales_dtype": scales = scales.half()
    elif bad == "scales_shape": scales = scales[:, :-1].contiguous()
    elif bad == "zeros_dtype": zeros = zeros.long()
    elif bad == "zeros_shape": zeros = zeros[:, :-1].contiguous()
    elif bad == "cpu_metadata": scales = scales.cpu()
    elif bad == "noncontiguous": x = torch.zeros(1, x.shape[1] * 2, device=x.device, dtype=x.dtype)[:, ::2]
    elif bad == "group_size": group_size = 32
    elif bad == "empty_n": packed, scales, zeros = packed[:0], scales[:0], zeros[:0]
    with pytest.raises(RuntimeError):
        getattr(torch.ops.lowbitlinear_ext, op_name)(x, packed, scales, zeros, group_size)


@pytest.mark.parametrize("op_name", OPS)
def test_cuda_op_rejects_cpu_input(op_name, gemv_inputs):
    x, packed, scales, zeros = (t.cpu() for t in gemv_inputs())
    with pytest.raises((RuntimeError, NotImplementedError)):
        getattr(torch.ops.lowbitlinear_ext, op_name)(x, packed, scales, zeros, 64)
