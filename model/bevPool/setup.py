import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)
def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
# /usr/local/cuda-11.8/bin/nvcc -o bev_pool.so -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
#     -D__CUDA_NO_HALF2_OPERATORS__ -gencode=arch=compute_70,code=sm_70 \
#     -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 \
#     -gencode=arch=compute_86,code=sm_86 src/bev_pool.cpp src/bev_pool_cuda.cu \
#     -I /usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include \
#     -I /usr/local/lib/python3.10/dist-packages/torch/include \
#     -L /usr/local/lib/python3.10/dist-packages/torch/lib/libc10.so \
#     -L /usr/local/lib/python3.10/dist-packages/torch/lib/libc10_cuda.so \
#     -L /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch.so \
#     -L /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda.so \
#     -L /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_python.so \
#     -L /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda_linalg.so \
#     -L /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_global_deps.so \
#     -L /usr/local/lib/python3.10/dist-packages/torch/lib/libcaffe2_nvrtc.so

if __name__ == '__main__':
    setup(
        name='bev_pool_now',
        ext_modules=[
            make_cuda_ext(
                name='bev_pool_ext',
                module='bev_pool',
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_cuda.cu',
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )