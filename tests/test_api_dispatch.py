import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from lowbitlinear import QuantizedLinear
import lowbitlinear.linear as linear_module
from lowbitlinear._extension import load_cuda_extension


def test_m1_dispatch_calls_vec_and_executes_cuda(cuda_device, monkeypatch):
    layer = QuantizedLinear.from_float(torch.randn(129, 256, device=cuda_device, dtype=torch.float16))
    x = torch.randn(1, 256, device=cuda_device, dtype=torch.float16)
    expected = linear_module.linear_int4_reference(x, layer.packed_weight, layer.scales, layer.zeros)
    op = Mock(wraps=torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec)
    monkeypatch.setattr(torch.ops.lowbitlinear_ext, "w4a16_gemv_warp_vec", op)
    def forbidden(*args, **kwargs):
        pytest.fail("M=1/group_size=64 must not call reference fallback")
    monkeypatch.setattr(linear_module, "linear_int4_reference", forbidden)
    actual = layer(x)
    op.assert_called_once()
    assert op.call_args.args[0] is x
    assert op.call_args.args[1] is layer.packed_weight
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("device,m,group_size", [("cpu", 1, 64), ("cpu", 16, 64), ("cuda", 16, 64), ("cuda", 1, 32)])
def test_reference_fallback_dispatch(device, m, group_size, monkeypatch, request):
    if device == "cuda":
        device = request.getfixturevalue("cuda_device")
    layer = QuantizedLinear.from_float(torch.randn(5, 256, device=device), group_size=group_size)
    x = torch.randn(m, 256, device=device, dtype=torch.float16)
    expected = linear_module.linear_int4_reference(x, layer.packed_weight, layer.scales, layer.zeros, group_size)
    reference = Mock(wraps=linear_module.linear_int4_reference)
    monkeypatch.setattr(linear_module, "linear_int4_reference", reference)
    def forbidden(*args, **kwargs):
        pytest.fail("Fallback dispatched to a CUDA GEMV")
    for op in ("w4a16_gemv", "w4a16_gemv_warp", "w4a16_gemv_warp_vec"):
        monkeypatch.setattr(torch.ops.lowbitlinear_ext, op, forbidden, raising=False)
    assert torch.equal(layer(x), expected)
    reference.assert_called_once()


def test_extension_autoload_no_jit_and_idempotence(cuda_device):
    # Fresh process: proves import registration, independent of pytest execution order.
    code = """
import torch
import torch.utils.cpp_extension as cpp
def forbidden(*args, **kwargs):
    raise AssertionError('JIT compilation is forbidden')
cpp.load = forbidden
calls = []
original = torch.ops.load_library
def record(path):
    calls.append(path)
    return original(path)
torch.ops.load_library = record
import lowbitlinear
from lowbitlinear._extension import load_cuda_extension
for _ in range(3):
    assert load_cuda_extension(required=True)
assert len(calls) == 1, calls
for name in ('w4a16_gemv', 'w4a16_gemv_warp', 'w4a16_gemv_warp_vec'):
    assert torch._C._dispatch_has_kernel_for_dispatch_key('lowbitlinear_ext::' + name, 'CUDA')
layer = lowbitlinear.QuantizedLinear.from_float(torch.ones(3, 64, device='cuda'))
assert torch.equal(layer(torch.ones(1, 64, dtype=torch.float16, device='cuda')), torch.full((1, 3), 64., device='cuda', dtype=torch.float16))
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90)


def test_cpu_import_and_api_without_shared_library(tmp_path):
    source = Path(__file__).resolve().parents[1] / "lowbitlinear"
    package = tmp_path / "lowbitlinear"
    package.mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, package / path.name)
    code = """
import torch
from lowbitlinear import QuantizedLinear
from lowbitlinear._extension import load_cuda_extension
assert not load_cuda_extension()
layer = QuantizedLinear.from_float(torch.ones(3, 64))
assert torch.equal(layer(torch.ones(2, 64, dtype=torch.float16)), torch.full((2, 3), 64., dtype=torch.float16))
try:
    load_cuda_extension(required=True)
except ImportError as exc:
    assert '--no-build-isolation' in str(exc)
else:
    raise AssertionError('Expected an actionable missing-extension error')
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(tmp_path))
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True, timeout=90)
