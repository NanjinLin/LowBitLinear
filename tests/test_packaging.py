"""Exercise source distributions built without selecting a CUDA extension."""
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("build_mode", ["no_toolkit", "cpu_torch"])
def test_sdist_contains_cuda_sources_and_runnable_tests(tmp_path, build_mode):
    # A fresh tree prevents an old egg-info/SOURCES.txt or compiled library
    # from hiding missing source-distribution entries.
    source = tmp_path / "source"
    source.mkdir()
    for name in ("MANIFEST.in", "LICENSE", "README.md", "pyproject.toml", "setup.py"):
        shutil.copy2(ROOT / name, source / name)
    for name in ("csrc", "lowbitlinear", "tests"):
        shutil.copytree(
            ROOT / name, source / name,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "*.so", "*.pyd"),
        )

    # Select each existing CPU-build condition in a child process. This checks
    # setup's branches; it does not claim to install a different PyTorch build.
    code = """
import runpy
import sys
import torch
from torch.utils import cpp_extension
if sys.argv[1] == 'no_toolkit':
    cpp_extension.CUDA_HOME = None
else:
    torch.version.cuda = None
sys.argv = ['setup.py', 'sdist', '--dist-dir', 'dist']
runpy.run_path('setup.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, build_mode], cwd=source,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout
    archives = list((source / "dist").glob("*.tar.gz"))
    assert len(archives) == 1
    extracted = tmp_path / "extracted"
    with tarfile.open(archives[0]) as archive:
        members = archive.getmembers()
        top_level = PurePosixPath(members[0].name).parts[0]
        files = {
            PurePosixPath(member.name).relative_to(top_level).as_posix()
            for member in members if member.isfile()
        }
        required = {
            "MANIFEST.in", "setup.py", "pyproject.toml",
            "csrc/w4a16_gemv_warp_vec.cu", "csrc/cute_w4a16_gemm.cu",
            *(path.relative_to(source).as_posix() for path in (source / "tests").rglob("*.py")),
        }
        assert required <= files, f"Missing from sdist: {sorted(required - files)}"
        assert not any(name.endswith((".so", ".pyd", ".pyc")) for name in files)
        # Read regular files explicitly: works on Python 3.10 too, and rejects
        # paths that would escape the temporary extraction directory.
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            if member.isfile():
                target = extracted.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    target.write_bytes(stream.read())

    # Reproduce the original missing-fixture failure from inside the archive.
    # These reference cases require neither a GPU nor an installed extension.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_linear.py",
         "-k", "test_linear_int4_reference"],
        cwd=extracted / top_level,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=""),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout
