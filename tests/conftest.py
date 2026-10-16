import pytest
import torch

from lowbitlinear._extension import load_cuda_extension
from lowbitlinear.quantization import quantize_int4, pack_int4


@pytest.fixture(scope="session", autouse=True)
def ieee_fp32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    # Missing/broken extension on a GPU machine is a failure, never a skip.
    load_cuda_extension(required=True)
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def gemv_inputs(cuda_device):
    def make(n=129, k=256, seed=42):
        generator = torch.Generator(device=cuda_device).manual_seed(seed)
        x = torch.randn(1, k, device=cuda_device, dtype=torch.float16, generator=generator)
        w = torch.randn(n, k, device=cuda_device, dtype=torch.float16, generator=generator)
        q, scales, zeros = quantize_int4(w)
        return x, pack_int4(q), scales, zeros
    return make
