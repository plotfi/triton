# Triton Nano Backend - Minimal backend for simple kernels like vector add
# Based on AMD backend, simplified for educational and prototyping purposes

import functools
import os
import subprocess
import re
import triton
from pathlib import Path
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton.runtime import _allocation
from triton.runtime.build import compile_module_from_src

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
PyTDMDescriptor = None
PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None


def _find_already_mmapped_dylib_on_linux(lib_name):
    import platform
    if platform.system() != 'Linux':
        return None

    import ctypes
    from ctypes import c_char, c_int, c_size_t, c_void_p, c_char_p, POINTER

    class DlPhdrInfo(ctypes.Structure):
        _fields_ = [
            ('dlpi_addr', c_void_p),
            ('dlpi_name', c_char_p),
        ]

    callback_t = ctypes.CFUNCTYPE(c_int, POINTER(DlPhdrInfo), POINTER(c_size_t), POINTER(c_char))

    try:
        dl_iterate_phdr = ctypes.CDLL('libc.so.6').dl_iterate_phdr
    except Exception:
        return None
    dl_iterate_phdr.argtypes = [callback_t, c_char_p]
    dl_iterate_phdr.restype = c_int

    max_path_length = 4096
    path = ctypes.create_string_buffer(max_path_length + 1)

    def callback(info, size, data):
        dlpi_name = info.contents.dlpi_name
        p = Path(os.fsdecode(dlpi_name))
        if lib_name in p.name:
            ctypes.memmove(data, dlpi_name, min(max_path_length, len(dlpi_name)))
            return 1
        return 0

    if dl_iterate_phdr(callback_t(callback), path):
        return os.fsdecode(ctypes.string_at(path))
    return None


@functools.lru_cache()
def _get_path_to_hip_runtime_dylib():
    """Find the HIP runtime library - reused from AMD backend."""
    lib_name = "libamdhip64.so"

    if env_libhip_path := knobs.amd.libhip_path:
        if env_libhip_path.endswith(lib_name) and os.path.exists(env_libhip_path):
            return env_libhip_path
        raise RuntimeError(f"TRITON_LIBHIP_PATH '{env_libhip_path}' does not point to a valid {lib_name}")

    mmapped_path = _find_already_mmapped_dylib_on_linux(lib_name)
    if mmapped_path:
        if os.path.exists(mmapped_path):
            return mmapped_path
        raise RuntimeError(f"memory mapped '{mmapped_path}' in process does not point to a valid {lib_name}")

    paths = []

    local_lib = os.path.join(os.path.dirname(__file__), "lib", lib_name)
    if os.path.exists(local_lib):
        return local_lib
    paths.append(local_lib)

    import site
    site_packages = site.getsitepackages()
    user_site = site.getusersitepackages()
    if site.ENABLE_USER_SITE:
        site_packages = [user_site] + site_packages
    for path in site_packages:
        path = os.path.join(path, "torch", "lib", lib_name)
        if os.path.exists(path):
            return path
        paths.append(path)

    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path:
        for d in env_ld_library_path.split(":"):
            f = os.path.join(d, lib_name)
            if os.path.exists(f):
                return f
            paths.append(f)

    env_hip_path = os.getenv("HIP_PATH")
    if env_hip_path:
        hip_lib_path = os.path.join(env_hip_path, "lib", lib_name)
        if os.path.exists(hip_lib_path):
            return hip_lib_path
        paths.append(hip_lib_path)

    try:
        hip_root = subprocess.check_output(["hipconfig", "--path"]).decode().strip()
        if hip_root:
            hip_lib_path = os.path.join(hip_root, "lib", lib_name)
            if os.path.exists(hip_lib_path):
                return hip_lib_path
            paths.append(hip_lib_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    env_rocm_path = os.getenv("ROCM_PATH")
    if env_rocm_path:
        rocm_lib_path = os.path.join(env_rocm_path, "lib", lib_name)
        if os.path.exists(rocm_lib_path):
            return rocm_lib_path
        paths.append(rocm_lib_path)

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    locs = [line.split()[-1] for line in libs.splitlines() if line.strip().endswith(lib_name)]
    for loc in locs:
        if os.path.exists(loc):
            return loc
        paths.append(loc)

    common_install_path = os.path.join('/opt/rocm/lib/', lib_name)
    if os.path.exists(common_install_path):
        return common_install_path
    paths.append(common_install_path)

    raise RuntimeError(f"cannot locate {lib_name} after attempted paths {paths}")


class NanoUtils(object):
    """Utility class for Nano backend - simplified from HIPUtils."""

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NanoUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        libhip_path = _get_path_to_hip_runtime_dylib()
        src = Path(os.path.join(dirname, "driver.c")).read_text()
        src = src.replace('/*py_libhip_search_path*/', libhip_path, 1)
        mod = compile_module_from_src(src=src, name="nano_utils", include_dirs=include_dirs)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.create_tdm_descriptor = mod.create_tdm_descriptor
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata
        global PyTDMDescriptor
        global PyKernelArg
        global ARG_CONSTEXPR
        global ARG_KERNEL
        global ARG_TUPLE
        PyTDMDescriptor = mod.PyTDMDescriptor
        PyKernelArg = mod.PyKernelArg
        ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        ARG_KERNEL = mod.ARG_KERNEL
        ARG_TUPLE = mod.ARG_TUPLE


# -------------------- Launcher ----------------------------
def ty_to_cpp(ty):
    if ty.startswith('*'):
        return "hipDeviceptr_t"
    if ty == "tensordesc":
        return "TDMDescriptor"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
    }[ty]


def expand_signature(signature, tensordesc_meta):
    output = []
    tensordesc_idx = 0
    for sig in signature:
        if isinstance(sig, str) and sig.startswith("tensordesc"):
            meta = tensordesc_meta[tensordesc_idx] if tensordesc_meta else None
            tensordesc_idx += 1

            match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", sig)
            dtype = match.group(1)
            shape = match.group(2)
            ndim = shape.count(",") + 1

            if meta is None:
                output.append("*" + dtype)
                for _ in range(2 * ndim):
                    output.append("i64")
                output.append("i1")
            else:
                output.append("tensordesc")

            for _ in range(ndim):
                output.append("i32")
            for _ in range(ndim):
                output.append("i64")
        else:
            output.append(sig)

    return output


def make_kernel_signature(signature):
    """Creates a kernel signature in C for efficient argument extraction."""

    def _flatten_signature(sig, output):
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    flat_signature = []
    for sig in signature:
        _flatten_signature(sig, flat_signature)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]

    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_signature)


def annotate_arguments(signature):
    """Annotate signature with C objects for efficient tuple flattening."""
    annotated_arguments = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated_arguments.append((PyKernelArg(nested_tuple=annotate_arguments(sig), type=ARG_TUPLE)))
        elif sig != "constexpr":
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated_arguments


def make_tensordesc_arg(arg, kernel_metadata, tensordesc_metadata):
    """Translate tensor descriptor argument into kernel arguments."""

    if tensordesc_metadata is None:
        return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", *arg.shape, *arg.strides]

    shape = arg.shape
    strides = arg.strides
    base = arg.base.data_ptr()

    assert "elem_bits" in tensordesc_metadata and "block_size" in tensordesc_metadata
    elem_bits = tensordesc_metadata["elem_bits"]
    block_size = tensordesc_metadata["block_size"]
    pad_interval, pad_amount = 0, 0
    interval_padding_pairs = tensordesc_metadata.get("interval_padding_pairs", [])
    if interval_padding_pairs:
        assert len(interval_padding_pairs) == 1 and len(interval_padding_pairs[0]) == 2
        pad_interval, pad_amount = interval_padding_pairs[0]
    num_warps = kernel_metadata[0]

    driver = triton.runtime.driver.active
    assert isinstance(driver, NanoDriver)

    desc = driver.utils.create_tdm_descriptor(elem_bits, block_size, num_warps, pad_interval, pad_amount, shape,
                                              strides, base)

    return [desc, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_metadata):
    """Wrap kernel launcher to handle tensor descriptor arguments."""

    has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    if not has_tensor_desc_arg:
        return launcher

    tensordesc_indices = set(
        [i for i, sig in enumerate(signature.values()) if isinstance(sig, str) and sig.startswith("tensordesc")])
    assert not tensordesc_metadata or len(tensordesc_metadata) == len(tensordesc_indices)
    if not tensordesc_metadata:
        tensordesc_metadata = [None] * len(tensordesc_indices)

    def inner(*args):
        base_args = args[:-1]
        kernel_metadata = base_args[7]
        kernel_args = args[-1]

        final_kernel_args = []
        tensordesc_idx = 0
        for i, arg in enumerate(kernel_args):
            if i in tensordesc_indices:
                final_kernel_args.extend(make_tensordesc_arg(arg, kernel_metadata, tensordesc_metadata[tensordesc_idx]))
                tensordesc_idx += 1
            else:
                final_kernel_args.append(arg)

        return launcher(*base_args, final_kernel_args)

    return inner


class NanoLauncher(object):
    """Simplified kernel launcher for Nano backend."""

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta)
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.warp_size = metadata.warp_size
        if self.launch_cooperative_grid:
            driver = triton.runtime.driver.active
            assert isinstance(driver, NanoDriver)
            device = driver.get_current_device()
            device_properties = driver.utils.get_device_properties(device)
            assert device_properties['cooperativeLaunch'], \
                "Cooperative launch requested but not supported by device"
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)

        self.launch(self.launch_cooperative_grid, gridX, gridY, gridZ, stream, function, profile_scratch,
                    kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, self.warp_size,
                    self.arg_annotations, self.kernel_signature, args)


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
        # Return nano as the backend name
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
