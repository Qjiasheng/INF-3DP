from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(ROOT_DIR, "includes")]

setup(
    name="qupkg",
    ext_modules=[ #  extension modules to be compiled from C/C++ or CUDA source files.
        CUDAExtension(  # means compiling a CUDA extension module
            name="qupkg.extract_isocontours", 
            sources=[
                "isocontours.cu",
                "intersection.cu",
                "ext.cpp"],
            include_dirs=include_dirs,
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension 
    }
)