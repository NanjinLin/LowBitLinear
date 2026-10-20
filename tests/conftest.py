"""Shared deterministic setup and an independent packed-weight oracle."""
import pytest
import torch

from lowbitlinear._extension import load_cuda_extension
from lowbitlinear.quantization import pack_int4, quantize_int4


@pytest.fixture(scope="session", autouse=True)
def ieee_fp32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture(autouse=True)
def reproducible_randomness():
    # Restore the caller's RNG state; each case is independent of test order.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        yield


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable; CPU coverage still runs")
    device = torch.device("cuda", torch.cuda.current_device())
    yield device
    # Attribute asynchronous errors to the case that launched the work.
    torch.cuda.synchronize(device)


@pytest.fixture
def cuda_ops(cuda_device):
    # A missing library, ABI error or missing registration on a GPU is a failure.
    load_cuda_extension(required=True)
    for name in (
        "lowbitlinear_ext::w4a16_gemv_warp_vec",
        "lowbitlinear_cute_w4a16::gemm",
    ):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA"), name
    return cuda_device


@pytest.fixture
def gemv_inputs(cuda_ops):
    def make(n=129, k=256, seed=42):
        generator = torch.Generator(device=cuda_ops).manual_seed(seed)
        x = torch.randn(1, k, device=cuda_ops, dtype=torch.float16, generator=generator)
        w = torch.randn(n, k, device=cuda_ops, dtype=torch.float16, generator=generator)
        q, scales, zeros = quantize_int4(w)
        return x, pack_int4(q), scales, zeros
    return make


@pytest.fixture
def packed_reference():
    def reference(x, packed, scales, zeros, group_size=64, *, mma=False):
        # Independent unpack + metadata indexing, including empty dimensions.
        # GEMV/reference dequantize in FP32. CuTe rounds weights to FP16 before MMA.
        k = x.shape[1]
        words = packed.to(torch.int64)
        shifts = torch.arange(8, device=x.device) * 4
        q = ((words[..., None] >> shifts) & 15).reshape(packed.shape[0], packed.shape[1] * 8)[:, :k]
        groups = torch.arange(k, device=x.device) // group_size
        weight = (q.float() - zeros[:, groups].float()) * scales[:, groups]
        if mma:
            weight = weight.half().float()
        return (x.float() @ weight.T).half()
    return reference
