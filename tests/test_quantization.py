
import torch
from lowbitlinear.quantization import quantize_int4, dequantize_int4, pack_int4, unpack_int4


def test_int4_quantization():
    torch.manual_seed(42)

    weight = torch.randn(128, 256, dtype=torch.float16)

    q, scales, zeros = quantize_int4(weight, group_size=64)
    restored = dequantize_int4(q, scales, zeros, group_size=64)

    # Shape
    assert q.shape == (128, 256)
    assert scales.shape == (128, 4)
    assert zeros.shape == (128, 4)

    # INT4 range
    assert q.min().item() >= 0
    assert q.max().item() <= 15

    # Quantization error: within one scale step
    error = (weight.float() - restored).abs()
    tolerance = scales.repeat_interleave(64, dim=1)

    assert torch.all(error <= tolerance + 1e-5)

    print("Max Abs Error:", error.max().item())
    print("Mean Abs Error:", error.mean().item())


def test_zero_weights():
    weight = torch.zeros(2, 64)
    q, scales, zeros = quantize_int4(weight)
    restored = dequantize_int4(q, scales, zeros)

    assert torch.equal(restored, weight)

def test_int4_packing():
    q = torch.tensor(
        [[1, 10, 3, 15, 5, 0, 9, 2]],
        dtype=torch.uint8
    )

    packed = pack_int4(q)

    assert packed.shape == (1, 1)
    assert packed.dtype == torch.int32
    assert packed.item() == 0x2905F3A1

    print("Packed:", hex(packed.item()))

def test_int4_pack_unpack():
    q = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8,
         9, 10, 11, 12, 13, 14, 15, 0]
    ], dtype=torch.uint8)

    packed = pack_int4(q)
    unpacked = unpack_int4(packed)

    assert packed.shape == (1, 2)
    assert unpacked.shape == (1, 16)
    assert torch.equal(q, unpacked)

    print("Original:", q)
    print("Packed:", packed)
    print("Unpacked:", unpacked)


def test_int4_full_pipeline():
    torch.manual_seed(42)

    # 1. 创建原始 FP16 权重
    weight = torch.randn(2, 128, dtype=torch.float16)

    # 2. Quantization
    q, scales, zeros = quantize_int4(weight, group_size=64)

    # 3. Packing
    packed = pack_int4(q)

    # 4. Unpacking
    unpacked = unpack_int4(packed)

    # 5. Dequantization
    restored = dequantize_int4(
        unpacked, scales, zeros, group_size=64
    )

    # 检查形状
    assert packed.shape == (2, 16)
    assert restored.shape == weight.shape

    # Packing / Unpacking 必须完全一致
    assert torch.equal(q, unpacked)

    # 检查量化误差
    error = (weight.float() - restored).abs()
    tolerance = scales.repeat_interleave(64, dim=1)

    assert torch.all(error <= tolerance + 1e-5)

    print("Original Shape:", weight.shape)
    print("Packed Shape:", packed.shape)
    print("Max Error:", error.max().item())
    print("Mean Error:", error.mean().item())

    # 计算真实存储空间
    def nbytes(t):
        return t.numel() * t.element_size()

    original_bytes = nbytes(weight)
    compressed_bytes = (
        nbytes(packed) + nbytes(scales) + nbytes(zeros)
    )

    print("Original Bytes:", original_bytes)
    print("Compressed Bytes:", compressed_bytes)
    print(
        "Storage Reduction:",
        f"{(1 - compressed_bytes / original_bytes) * 100:.2f}%"
    )
