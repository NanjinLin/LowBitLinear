import torch
from lowbitlinear import QuantizedLinear

# 1. 准备输入和权重
device = "cuda"

weight = torch.randn(
    128, 256,
    device=device,
    dtype=torch.float16
)

x = torch.randn(
    16, 256,
    device=device,
    dtype=torch.float16
)

# 2. 创建量化 Linear 层
layer = QuantizedLinear.from_float(
    weight,
    bits=4,
    group_size=64
)

# 3. 执行 Forward
y = layer(x)

print("Output Shape:", y.shape)
print("Output Dtype:", y.dtype)

# 4. 查看量化后的权重存储
print("Packed Weight Shape:", layer.packed_weight.shape)
print("Scales Shape:", layer.scales.shape)
print("Zeros Shape:", layer.zeros.shape)

# 5. 计算存储空间
def nbytes(t):
    return t.numel() * t.element_size()

original_bytes = nbytes(weight)

quantized_bytes = (
    nbytes(layer.packed_weight)
    + nbytes(layer.scales)
    + nbytes(layer.zeros)
)

print("Original Bytes:", original_bytes)
print("Quantized Bytes:", quantized_bytes)
print(
    "Storage Reduction:",
    f"{(1 - quantized_bytes / original_bytes) * 100:.2f}%"
)
