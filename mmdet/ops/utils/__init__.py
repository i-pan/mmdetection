# from . import compiling_info
try:
    from .compiling_info import get_compiler_version, get_compiling_cuda_version
except ModuleNotFoundError:
    print('Unable to import from `compiling_info`')

# get_compiler_version = compiling_info.get_compiler_version
# get_compiling_cuda_version = compiling_info.get_compiling_cuda_version

__all__ = ['get_compiler_version', 'get_compiling_cuda_version']
