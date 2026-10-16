"""Load installed operators once. Importing never JIT-compiles CUDA sources."""
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import torch

_loaded_library = None


def load_cuda_extension(*, required=False):
    global _loaded_library
    if _loaded_library is not None:
        return True
    package_dir = Path(__file__).resolve().parent
    libraries = sorted({
        p for suffix in EXTENSION_SUFFIXES
        for p in package_dir.glob(f"_C*{suffix}")
    })
    if torch.version.cuda is None or not libraries:
        if required:
            raise ImportError(
                "LowBitLinear CUDA extension is unavailable. Use CUDA-enabled PyTorch "
                "and rebuild with: python -m pip install -e . --no-build-isolation"
            )
        return False
    if len(libraries) != 1:
        raise ImportError(f"Ambiguous LowBitLinear extensions: {libraries}")
    # Do not swallow ABI / shared-library errors in a CUDA-enabled installation.
    torch.ops.load_library(str(libraries[0]))
    _loaded_library = libraries[0]
    return True


load_cuda_extension()
