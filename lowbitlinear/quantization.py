import torch

def quantize_int4(weight, group_size=64):
    """
    weight: [N, K] , FP16/FP32
    return:
        qweight: [N, K], unit8
        scales: [N, K/group_size], FP32
        zeros: [N, K/group_size], int32
    """
    N, K = weight.shape
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
    N, K = qweight.shape
    groups = qweight.reshape(N, K // group_size, group_size)
    w = (
        groups.float() - zeros.unsqueeze(-1).float()
    ) * scales.unsqueeze(-1)

    return w.reshape(N, K)

def pack_int4(qweight):
    """
    return: [N, K // 8], int32
    """
    N, K = qweight.shape
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
    values = []

    for i in range(8):
        q = (packed >> (4 * i)) & 0xF
        values.append(q)

    unpacked = torch.stack(values, dim=-1)
    return unpacked.reshape(packed.shape[0], -1).to(torch.uint8)