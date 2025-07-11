from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

# Gather all CUDA and C++ source files for selective_scan
sources = glob.glob('csrc/selective_scan/*.cpp') + glob.glob('csrc/selective_scan/*.cu')

setup(
    name='ned_model_cuda',
    ext_modules=[
        CUDAExtension(
            name='selective_scan_cuda',
            sources=sources,
            include_dirs=['csrc/selective_scan'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
) 