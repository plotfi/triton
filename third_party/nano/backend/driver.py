# Triton Nano Backend - Minimal driver for simple kernels like vector add

import functools
import os
import subprocess
import triton
from pathlib import Path
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton.runtime.build import compile_module_from_src

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]


@functools.lru_cache()
def _get_path_to_hip_runtime_dylib():
    """Find the HIP runtime library."""
    lib_name = "libamdhip64.so"

    if env_libhip_path := knobs.amd.libhip_path:
        if env_libhip_path.endswith(lib_name) and os.path.exists(env_libhip_path):
            return env_libhip_path
        raise RuntimeError(f"TRITON_LIBHIP_PATH '{env_libhip_path}' does not point to a valid {lib_name}")

    # Check LD_LIBRARY_PATH
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path:
        for d in env_ld_library_path.split(":"):
            f = os.path.join(d, lib_name)
            if os.path.exists(f):
                return f

    # Check HIP_PATH
    env_hip_path = os.getenv("HIP_PATH")
    if env_hip_path:
        hip_lib_path = os.path.join(env_hip_path, "lib", lib_name)
        if os.path.exists(hip_lib_path):
            return hip_lib_path

    # Check ROCM_PATH
    env_rocm_path = os.getenv("ROCM_PATH")
    if env_rocm_path:
        rocm_lib_path = os.path.join(env_rocm_path, "lib", lib_name)
        if os.path.exists(rocm_lib_path):
            return rocm_lib_path

    # Check common install path
    common_install_path = os.path.join('/opt/rocm/lib/', lib_name)
    if os.path.exists(common_install_path):
        return common_install_path

    # Try ldconfig
    try:
        libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
        locs = [line.split()[-1] for line in libs.splitlines() if line.strip().endswith(lib_name)]
        for loc in locs:
            if os.path.exists(loc):
                return loc
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(f"cannot locate {lib_name}")


class NanoUtils(object):
    """Utility class for Nano backend."""

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NanoUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        libhip_path = _get_path_to_hip_runtime_dylib()
        src = Path(os.path.join(dirname, "driver.c")).read_text()
        src = src.replace('/*py_libisa_search_path*/', libhip_path, 1)
        mod = compile_module_from_src(src=src, name="nano_utils", include_dirs=include_dirs)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata


def ty_to_cpp(ty):
    """Map Triton types to C++ types."""
    if ty.startswith('*'):
        return "hipDeviceptr_t"
    return {
        "i1": "int32_t",
        "i32": "int32_t",
        "u1": "uint32_t",
        "u32": "uint32_t",
        "fp32": "float",
        "f32": "float",
    }[ty]


def make_kernel_signature(signature):
    """Create signature metadata for kernel launch."""
    flat_signature = []
    for sig in signature:
        if isinstance(sig, tuple):
            flat_signature.extend(sig)
        elif sig != "constexpr":
            flat_signature.append(sig)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]
    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_signature)


class NanoLauncher(object):
    """Simplified kernel launcher for Nano backend."""

    def __init__(self, src, metadata):
        signature = {idx: value for idx, value in src.signature.items()}
        self.launch = triton.runtime.driver.active.utils.launch
        self.kernel_signature = make_kernel_signature(signature.values())
        self.warp_size = metadata.warp_size

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        # Extract num_warps and shared_memory from kernel_metadata tuple
        num_warps, num_ctas, shared_memory = kernel_metadata

        # Flatten args if needed
        flat_args = []
        for arg in args:
            if isinstance(arg, tuple):
                flat_args.extend(arg)
            else:
                flat_args.append(arg)

        # Call simplified launch
        self.launch(gridX, gridY, gridZ, stream, function, num_warps, shared_memory,
                    self.warp_size, self.kernel_signature, flat_args)


class NanoDriver(GPUDriver):
    """Minimal GPU driver for Nano backend."""

    def __init__(self):
        super().__init__()
        self.utils = NanoUtils()
        self.launcher_cls = NanoLauncher

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.cuda.is_available() and (torch.version.hip is not None)
        except ImportError:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = knobs.runtime.override_arch or device_properties['arch']
        warp_size = device_properties['warpSize']
        return GPUTarget("nano", arch.split(':')[0], warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
