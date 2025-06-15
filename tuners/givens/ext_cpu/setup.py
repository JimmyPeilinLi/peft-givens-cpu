from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="goft_cpu_ext",
    ext_modules=[
        CppExtension(
            "goft_cpu_ext",
            ["binding.cpp", "goft_cpu.cpp"],
            extra_compile_args=["-O3", "-fopenmp", "-march=native", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
