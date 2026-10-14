# LowBitLinear

A low-bit quantized linear operator library for PyTorch.

The project focuses on INT3/INT4 weight quantization,
CUDA kernel development, and GPU performance engineering.

## Current Status

V0: INT4 PyTorch Reference Implementation

Implemented:
- Group-wise affine INT4 quantization
- INT4 bit packing and unpacking
- INT4 dequantization
- Quantized Linear reference implementation
- PyTorch QuantizedLinear API
- Correctness tests

The current version uses PyTorch operations.
Custom CUDA kernels are planned for V1/V2.

## Installation

Requirements:
- Python >= 3.10
- PyTorch >= 2.3

Install in development mode:

    python -m pip install -e .

## Quick Start

```python
import torch
from lowbitlinear import QuantizedLinear

device = "cuda"

weight = torch.randn(
    128, 256, device=device, dtype=torch.float16
)

x = torch.randn(
    16, 256, device=device, dtype=torch.float16
)

layer = QuantizedLinear.from_float(
    weight, bits=4, group_size=64
)

y = layer(x)

print(y.shape)  # [16, 128]
```

## Tests

Run all correctness tests:

    python -m pytest -q

Run the basic example:

    python examples/basic_usage.py

## Supported Features

- INT4 unsigned quantization
- Group-wise affine quantization
- Group size: 64 (default)
- FP16 weight and activation
- 2D input and weight tensors
- Inference-oriented forward computation

Current constraints:
- K must be divisible by group_size
- Only INT4 is implemented
- No custom CUDA kernels yet
- No performance guarantee
- No bias support
- Backward is not a project target

## Roadmap

- V0: Quantization + PyTorch Reference
- V1: CUDA Fused GEMV (INT3/INT4)
- V2: INT4 Tensor Core GEMM + Dispatch
- V3: Triton comparison and integration improvements

## References

- AWQ: https://github.com/mit-han-lab/llm-awq
- MiLo: https://github.com/Supercomputing-System-AI-Lab/MiLo
```
