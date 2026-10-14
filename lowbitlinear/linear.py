import torch

from .quantization import unpack_int4, dequantize_int4, quantize_int4, pack_int4

def linear_int4_reference(x, packed_weight, scales, zeros, group_size=64):
    """
    x:             [M, K]     FP16
    packed_weight: [N, K//8]  int32
    scales:        [N, K//64] FP32
    zeros:         [N, K//64] int32

    output:        [M, N]     FP16
    """

    M, K = x.shape
    N, packed_K = packed_weight.shape

    #unpacking
    qweight = unpack_int4(packed_weight)

    #dequantization
    weight = dequantize_int4(qweight, scales, zeros, group_size)

    #convert FP32 to FP16
    weight = weight.to(x.dtype)

    return x @ weight.T

class QuantizedLinear(torch.nn.Module):
    def __init__(self, packed_weight, scales, zeros, group_size=64):
        super().__init__()

        self.group_size = group_size

        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)

    @classmethod
    def from_float(cls, weight, bits=4, group_size=64):
        with torch.no_grad():
            q, scales,zeros = quantize_int4(weight, group_size)
            packed = pack_int4(q)

        return cls(packed, scales, zeros, group_size)

    def forward(self, x):
        return linear_int4_reference(
            x,
            self.packed_weight,
            self.scales,
            self.zeros,
            self.group_size
        )
    