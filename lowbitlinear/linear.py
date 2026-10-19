import torch

from ._extension import load_cuda_extension
from .quantization import (
    unpack_int4, dequantize_int4, quantize_int4, pack_int4,
    _validate_matrix, _validate_metadata,
)


def _validate_packed_weight(
    packed_weight, scales, zeros, group_size, in_features
):
    _validate_matrix(packed_weight, "packed_weight", (torch.int32,))

    if (
        not isinstance(in_features, int)
        or isinstance(in_features, bool)
        or in_features <= 0
    ):
        raise ValueError("in_features must be a positive integer")

    N, packed_cols = packed_weight.shape

    expected_cols = (in_features + 7) // 8

    if packed_cols != expected_cols:
        raise ValueError("Invalid packed_weight shape")

    _validate_metadata(
        scales, zeros,
        N, in_features, group_size,
        packed_weight.device
    )

def _validate_input(x, packed_weight, in_features):
    _validate_matrix(x, "x", (torch.float16,))
    if x.shape[1] != in_features:
        raise ValueError("x and packed_weight have incompatible K")
    if x.device != packed_weight.device:
        raise ValueError("All tensors must be on the same device")


def linear_int4_reference(x, packed_weight, scales, zeros, group_size=64):
    """Dequantize the packed INT4 state to FP32, multiply in FP32, return FP16.

    x: [M, K] FP16; packed_weight: [N, K//8] int32;
    scales: [N, K//group_size] FP32; zeros: same shape, int32.
    Callers requiring an IEEE FP32 oracle must disable TF32 matmul.
    """
    _validate_matrix(x, "x", (torch.float16,))

    K = x.shape[1]

    _validate_packed_weight(
        packed_weight, scales, zeros,
        group_size, K
    )
    _validate_input(x, packed_weight, K)

    qweight = unpack_int4(
        packed_weight, original_k=K
    )

    weight = dequantize_int4(
        qweight, scales, zeros, group_size
    )

    y = x.float() @ weight.T

    return y.to(torch.float16)


class QuantizedLinear(torch.nn.Module):
    """W4A16 inference: CUDA Warp Vec for M=1/group_size=64, reference otherwise."""

    def __init__(self, packed_weight, scales, zeros, group_size=64, *, in_features):
        super().__init__()
        _validate_packed_weight(packed_weight, scales, zeros, group_size, in_features)
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = packed_weight.shape[0]
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)

    @classmethod
    def from_float(cls, weight, bits=4, group_size=64):
        if bits != 4 or isinstance(bits, bool):
            raise ValueError("Only bits=4 is supported")
        with torch.no_grad():
            q, scales, zeros = quantize_int4(weight, group_size)
            packed = pack_int4(q)
        return cls(packed, scales, zeros, group_size, in_features=weight.shape[1])

    def forward(self, x):
        _validate_packed_weight(self.packed_weight, self.scales, self.zeros, self.group_size, self.in_features)
        _validate_input(x, self.packed_weight, self.in_features)
        if not x.is_contiguous():
            raise ValueError("CUDA input x must be contiguous")
        if x.is_cuda and self.group_size == 64:
            load_cuda_extension(required=True)

            if (
                x.shape[0] == 1
                and self.in_features % 64 == 0
                and self.out_features % 128 == 0
            ):
                return torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(
                    x,
                    self.packed_weight,
                    self.scales,
                    self.zeros,
                    self.group_size
                )

            return torch.ops.lowbitlinear_cute_w4a16.gemm(
                x,
                self.packed_weight,
                self.scales,
                self.zeros
            )

        return linear_int4_reference(
            x,
            self.packed_weight,
            self.scales,
            self.zeros,
            self.group_size
        )