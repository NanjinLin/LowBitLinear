import os
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# A CPU-only install keeps quantization and the reference path usable.
ext_modules = []
if torch.version.cuda is not None and CUDA_HOME is not None:
    # CUTLASS / CuTe 头文件所在目录
    cutlass_include = Path(
        os.environ.get(
            "CUTLASS_INCLUDE",
            "~/projects/cutlass/include"
        )
    ).expanduser()

    if not (cutlass_include / "cute" / "tensor.hpp").is_file():
        raise RuntimeError(
            f"CuTe headers not found: {cutlass_include}"
        )
    
    ext_modules.append(
        CUDAExtension(
            name="lowbitlinear._C",
            # Unique launchers/operators; Naive owns TORCH_LIBRARY, others use FRAGMENT.
            sources=[
                "csrc/w4a16_gemv_warp_vec.cu",
                "csrc/cute_w4a16_gemm.cu"
            ],

            include_dirs=[str(cutlass_include)],

            extra_compile_args={"cxx": ["-O2","-std=c++20"], "nvcc": ["-O3", "-std=c++20","-lineinfo"]},
        )
    )
setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExtension})
