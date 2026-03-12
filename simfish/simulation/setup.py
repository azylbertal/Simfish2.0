# setup.py for geometry module
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Define the extension module with OpenMP support
ext_modules = [
    Pybind11Extension(
        "geometry",  # Module name
        ["geometry.cpp"],  # Source files
        cxx_std=17,  # Use C++17
        extra_compile_args=[
            "-O3",  # Maximum optimization
            "-march=native",  # Optimize for your CPU
            "-fopenmp",  # Enable OpenMP for parallelization
            "-ffast-math",  # Fast math optimizations (be careful with this)
        ],
        extra_link_args=["-fopenmp"],
        include_dirs=[pybind11.get_include()],
        define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", None)],  # Better error messages
    ),
]

setup(
    name="geometry",
    version="1.0.0",
    author="Asaph Zylbertal",
    description="Fast C++ implementation of geometry functions for fish visual system simulation",
    long_description="Optimized C++ implementations of ray tracing, circle edge detection, and visual system calculations",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pybind11",
    ],
)
