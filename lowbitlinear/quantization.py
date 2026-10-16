import torch


def _validate_matrix(tensor, name, dtypes):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {tensor.dtype}")
    if tensor.ndim != 2 or min(tensor.shape) <= 0:
        raise ValueError(f"{name} must be a nonempty 2D tensor")
    if tensor.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{name} must be on CPU or CUDA")


def _validate_group_size(k, group_size):
    if (not isinstance(group_size, int) or isinstance(group_size, bool)
            or group_size <= 0 or k % group_size):
        raise ValueError("group_size must be a positive integer dividing K")


def _validate_metadata(scales, zeros, n, k, group_size, device):
    _validate_group_size(k, group_size)
    _validate_matrix(scales, "scales", (torch.float32,))
    _validate_matrix(zeros, "zeros", (torch.int32,))
    if scales.shape != (n, k // group_size) or zeros.shape != scales.shape:
        raise ValueError("Invalid scales or zeros shape")
    if scales.device != device or zeros.device != device:
        raise ValueError("All tensors must be on the same device")

def quantize_int4(weight, group_size=64):
    """
    weight: [N, K] , FP16/FP32
    return:
        qweight: [N, K], uint8
        scales: [N, K/group_size], FP32
        zeros: [N, K/group_size], int32
    """
    _validate_matrix(weight, "weight", (torch.float16, torch.float32))
    N, K = weight.shape
    _validate_group_size(K, group_size)
    groups = weight.float().reshape(N, K // group_size, group_size)

    #每组计算Min/Max
    w_min = groups.amin(dim=-1).clamp(max=0)
    w_max = groups.amax(dim=-1).clamp(min=0)

    #计算scale
    raw_scale = (w_max - w_min) / 15.0
    scales = torch.where(
        raw_scale > 0,
        raw_scale,
        torch.ones_like(raw_scale)
    )

    #计算zero point
    zeros = torch.round(-w_min / scales).clamp(0, 15).to(torch.int32)

    #Quantization
    q = torch.round(
        groups / scales.unsqueeze(-1) + zeros.unsqueeze(-1)
    )

    q = q.clamp(0, 15).to(torch.uint8)

    return q.reshape(N, K), scales, zeros

def dequantize_int4(qweight, scales, zeros, group_size=64):
    _validate_matrix(qweight, "qweight", (torch.uint8,))
    N, K = qweight.shape
    _validate_metadata(scales, zeros, N, K, group_size, qweight.device)
    groups = qweight.reshape(N, K // group_size, group_size)
    w = (
        groups.float() - zeros.unsqueeze(-1).float()
    ) * scales.unsqueeze(-1)

    return w.reshape(N, K)

def pack_int4(qweight):
    """
    return: [N, K // 8], int32
    """
    _validate_matrix(qweight, "qweight", (torch.uint8,))
    N, K = qweight.shape
    if K % 8:
        raise ValueError("K must be divisible by 8 for INT4 packing")
    if torch.any(qweight > 15).item():
        raise ValueError("qweight values must be in [0, 15]")
    packed = torch.zeros(
        (N, K // 8),
        dtype=torch.int64,
        device=qweight.device
    )

    for i in range(8):
        packed |= (
            qweight[:, i::8].to(torch.int64) << (4 * i)
        )

    return packed.to(torch.int32)

def unpack_int4(packed):
    """
    packed: [N, K // 8], int32
    return: [N, K], uint8
    """
    _validate_matrix(packed, "packed", (torch.int32,))
    values = []

    for i in range(8):
        q = (packed >> (4 * i)) & 0xF
        values.append(q)

    unpacked = torch.stack(values, dim=-1)
    return unpacked.reshape(packed.shape[0], -1).to(torch.uint8)
