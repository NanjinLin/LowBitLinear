from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# A CPU-only install keeps quantization and the reference path usable.
ext_modules = []
if torch.version.cuda is not None and CUDA_HOME is not None:
    ext_modules.append(
        CUDAExtension(
            name="lowbitlinear._C",
            # Unique launchers/operators; Naive owns TORCH_LIBRARY, others use FRAGMENT.
            sources=[
                "csrc/w4a16_gemv_naive.cu",
                "csrc/w4a16_gemv_warp.cu",
                "csrc/w4a16_gemv_warp_vec.cu",
                "csrc/w4a16_gemm.cu",
                "csrc/w4a16_gemm_padding_baseline.cu"
            ],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O3", "-lineinfo"]},
        )
    )
setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExtension})
