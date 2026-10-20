"""Real dispatch and installed extension lifecycle, with no JIT compilation."""
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


@pytest.mark.parametrize("m,n,k,route", [
    (1, 128, 64, "gemv"), (1, 256, 256, "gemv"),
    (1, 127, 256, "gemm"), (1, 129, 256, "gemm"),
    (1, 128, 65, "gemm"), (16, 128, 64, "gemm"),
    (1, 1, 1, "gemm"), (5, 9, 70, "gemm"), (9, 17, 129, "gemm"),
])
def test_cuda_dispatch_executes_selected_kernel(m, n, k, route, cuda_ops, monkeypatch, packed_reference):
    layer = QuantizedLinear.from_float(torch.randn(n, k, device=cuda_ops, dtype=torch.float16))
    x = torch.randn(m, k, device=cuda_ops, dtype=torch.float16)
    expected = packed_reference(x, layer.packed_weight, layer.scales, layer.zeros, mma=route == "gemm")
    gemv = Mock(wraps=torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec)
    gemm = Mock(wraps=torch.ops.lowbitlinear_cute_w4a16.gemm)
    monkeypatch.setattr(torch.ops.lowbitlinear_ext, "w4a16_gemv_warp_vec", gemv)
    monkeypatch.setattr(torch.ops.lowbitlinear_cute_w4a16, "gemm", gemm)
    def forbidden(*args, **kwargs):
        pytest.fail("Supported CUDA group_size=64 must execute the installed kernel")
    monkeypatch.setattr(linear_module, "linear_int4_reference", forbidden)
    actual = layer(x)
    selected, other = (gemv, gemm) if route == "gemv" else (gemm, gemv)
    selected.assert_called_once()
    other.assert_not_called()
    assert selected.call_args.args[0] is x
    assert selected.call_args.args[1] is layer.packed_weight
    assert len(selected.call_args.args) == (5 if route == "gemv" else 4)
    if route == "gemv":
        assert selected.call_args.args[4] == 64
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("device,m,group_size", [("cpu", 1, 64), ("cpu", 16, 64), ("cpu", 5, 63), ("cuda", 1, 32), ("cuda", 5, 63)])
def test_reference_fallback_dispatch(device, m, group_size, monkeypatch, request, packed_reference):
    if device == "cuda":
        device = request.getfixturevalue("cuda_device")
    layer = QuantizedLinear.from_float(torch.randn(5, 129, device=device), group_size=group_size)
    x = torch.randn(m, 129, device=device, dtype=torch.float16)
    expected = packed_reference(x, layer.packed_weight, layer.scales, layer.zeros, group_size)
    reference = Mock(wraps=linear_module.linear_int4_reference)
    monkeypatch.setattr(linear_module, "linear_int4_reference", reference)
    # The fallback must not require an extension, even on a CUDA device.
    def forbidden(*args, **kwargs):
        pytest.fail("Reference fallback tried loading the extension")
    monkeypatch.setattr(linear_module, "load_cuda_extension", forbidden)
    assert torch.equal(layer(x), expected)
    reference.assert_called_once()


def test_extension_autoload_no_jit_and_idempotence(cuda_ops):
    # A fresh process prevents registration left by other tests from masking errors.
    code = """
import torch
import torch.utils.cpp_extension as cpp
def forbidden(*args, **kwargs):
    raise AssertionError('JIT compilation is forbidden')
cpp.load = forbidden
cpp.load_inline = forbidden
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
assert str(torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec.default._schema) == 'lowbitlinear_ext::w4a16_gemv_warp_vec(Tensor x, Tensor packed_w, Tensor scales, Tensor zeros, int group_size) -> Tensor'
assert str(torch.ops.lowbitlinear_cute_w4a16.gemm.default._schema) == 'lowbitlinear_cute_w4a16::gemm(Tensor X, Tensor packed_w, Tensor scales, Tensor zeros) -> Tensor'
for name in ('lowbitlinear_ext::w4a16_gemv_warp_vec', 'lowbitlinear_cute_w4a16::gemm'):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(name, 'CUDA'), name
for m, n, k in ((1, 128, 64), (5, 9, 70)):
    layer = lowbitlinear.QuantizedLinear.from_float(torch.ones(n, k, device='cuda'))
    actual = layer(torch.ones(m, k, dtype=torch.float16, device='cuda'))
    assert torch.equal(actual, torch.full_like(actual, float(k)))
torch.cuda.synchronize()
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90)


@pytest.fixture
def python_only_package(tmp_path):
    source = Path(__file__).resolve().parents[1] / "lowbitlinear"
    package = tmp_path / "lowbitlinear"
    package.mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, package / path.name)
    return tmp_path


def test_cpu_import_and_api_without_shared_library(python_only_package):
    tmp_path = python_only_package
    code = """
from pathlib import Path
import pytest
import torch
import lowbitlinear
from lowbitlinear import QuantizedLinear
from lowbitlinear._extension import load_cuda_extension
assert Path(lowbitlinear.__file__).resolve().parent == Path.cwd() / 'lowbitlinear'
assert not load_cuda_extension()
layer = QuantizedLinear.from_float(torch.ones(3, 65))
assert torch.equal(layer(torch.ones(2, 65, dtype=torch.float16)), torch.full((2, 3), 65., dtype=torch.float16))
with pytest.raises(ImportError, match='--no-build-isolation'):
    load_cuda_extension(required=True)
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(tmp_path))
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True, timeout=90)


def test_cuda_without_extension_is_an_error(python_only_package, cuda_device):
    # Do not turn a missing CUDA build into a silent CPU fallback or a skip.
    tmp_path = python_only_package
    code = """
import pytest
import torch
from lowbitlinear import QuantizedLinear
layer = QuantizedLinear.from_float(torch.ones(128, 64, device='cuda'))
with pytest.raises(ImportError, match='--no-build-isolation'):
    layer(torch.ones(1, 64, device='cuda', dtype=torch.float16))
"""
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True, timeout=90)

@pytest.mark.parametrize("op_name", ["gemv", "gemm"])
def test_operator_on_noncurrent_gpu(op_name, cuda_ops):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA GPUs required to verify a noncurrent input device")
    # A bad device launch can poison CUDA state; keep this regression in a child.
    code = """
import sys
import torch
import lowbitlinear
from lowbitlinear.quantization import pack_int4
with torch.cuda.device(1):
    x = torch.ones(1, 64, device='cuda:1', dtype=torch.float16)
    packed = pack_int4(torch.ones(128, 64, device='cuda:1', dtype=torch.uint8))
    scales = torch.ones(128, 1, device='cuda:1')
    zeros = torch.zeros(128, 1, device='cuda:1', dtype=torch.int32)
    torch.cuda.synchronize(1)
torch.cuda.set_device(0)
if sys.argv[1] == 'gemv':
    actual = torch.ops.lowbitlinear_ext.w4a16_gemv_warp_vec(x, packed, scales, zeros, 64)
else:
    actual = torch.ops.lowbitlinear_cute_w4a16.gemm(x, packed, scales, zeros)
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
assert torch.cuda.current_device() == 0
assert actual.device == x.device
assert torch.equal(actual, torch.full_like(actual, 64.))
"""
    subprocess.run([sys.executable, "-c", code, op_name], check=True, timeout=90)
