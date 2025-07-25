import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

# Add compute capabilities for modern GPUs including RTX 5090 (sm_120)
nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
    # Explicitly include compute capabilities for modern GPUs
    '--generate-code=arch=compute_70,code=sm_70',   # V100, RTX 20 series
    '--generate-code=arch=compute_75,code=sm_75',   # RTX 20 series
    '--generate-code=arch=compute_80,code=sm_80',   # A100, RTX 30 series
    '--generate-code=arch=compute_86,code=sm_86',   # RTX 30 series
    '--generate-code=arch=compute_89,code=sm_89',   # RTX 40 series
    '--generate-code=arch=compute_90,code=sm_90',   # H100, RTX 40 series
    '--generate-code=arch=compute_120,code=sm_120', # RTX 5090
    '--generate-code=arch=compute_120,code=compute_120', # PTX for future compatibility
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++17']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

setup(
    name='cuda_backend', # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name='_cuda_backend', # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'gridencoder.cu',
                'pdf.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
