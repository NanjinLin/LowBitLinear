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
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size <= 0
    ):
        raise ValueError("group_size must be a positive integer")


def _validate_metadata(scales, zeros, n, k, group_size, device):
    _validate_group_size(k, group_size)
    _validate_matrix(scales, "scales", (torch.float32,))
    _validate_matrix(zeros, "zeros", (torch.int32,))
    if scales.shape != (n, (k + group_size - 1) // group_size) or zeros.shape != scales.shape:
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

    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    
    num_groups = (K + group_size - 1) // group_size

    q_list = []
    scale_list = []
    zero_list = []

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, K)
        group = weight[:, start:end].float()
        
        #每组计算Min/Max
        w_min = group.amin(dim=-1).clamp(max=0)
        w_max = group.amax(dim=-1).clamp(min=0)
        
        #计算scale
        raw_scale = (w_max - w_min) / 15.0
        scale = torch.where(
            raw_scale > 0,
            raw_scale,
            torch.ones_like(raw_scale)
        )
        
        #计算zero point
        zero = torch.round(-w_min / scale).clamp(0, 15).to(torch.int32)
        
        #Quantization
        q = torch.round(
            group / scale.unsqueeze(1) + zero.unsqueeze(1)
        ).clamp(0, 15).to(torch.uint8)

        q_list.append(q)
        scale_list.append(scale)
        zero_list.append(zero)

    qweight = torch.cat(q_list, dim=1)
    scales = torch.stack(scale_list, dim=1)
    zeros = torch.stack(zero_list,dim=1)

    return qweight, scales, zeros

def dequantize_int4(qweight, scales, zeros, group_size=64):
    _validate_matrix(qweight, "qweight", (torch.uint8,))
    N, K = qweight.shape
    _validate_metadata(scales, zeros, N, K, group_size, qweight.device)

    num_groups = (K + group_size - 1) // group_size

    w_list = []

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, K)

        q_group = qweight[:, start:end].float()

        scale = scales[:, g].unsqueeze(1)
        zero = zeros[:, g].float().unsqueeze(1)

        w_group = (q_group - zero) * scale

        w_list.append(w_group)

    return torch.cat(w_list, dim=1)

def pack_int4(qweight):
    """
    return: [N, K + 7 // 8], int32
    """
    _validate_matrix(qweight, "qweight", (torch.uint8,))
    N, K = qweight.shape
    
    if torch.any(qweight > 15).item():
        raise ValueError("qweight values must be in [0, 15]")

    packed_cols = (K + 7) // 8
    padded_K = packed_cols * 8

    #padding
    if padded_K > K:
        qweight = torch.nn.functional.pad(
            qweight, (0, padded_K - K), value=0
        )
    packed = torch.zeros(
        (N, packed_cols),
        dtype=torch.int64,
        device=qweight.device
    )

    for i in range(8):
        packed |= (
            qweight[:, i::8].to(torch.int64) << (4 * i)
        )

    return packed.to(torch.int32)

def unpack_int4(packed, original_k=None):
    """
    packed: [N, K + 7 // 8], int32
    return: [N, K], uint8
    """
    _validate_matrix(packed, "packed", (torch.int32,))

    values = []

    for i in range(8):
        q = (packed >> (4 * i)) & 0xF
        values.append(q)

    unpacked = torch.stack(values, dim=-1)
    unpacked = unpacked.reshape(packed.shape[0], -1)
    unpacked = unpacked.to(torch.uint8)

    if original_k is not None:
        if (
            not isinstance(original_k, int)
            or isinstance(original_k, bool)
            or original_k <= 0
            or (original_k + 7) // 8 != packed.shape[1]
        ):
            raise ValueError("Invalid original_k")

        unpacked = unpacked[:, :original_k]

    return unpacked
