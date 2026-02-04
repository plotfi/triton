# Triton Nano Backend - ISA Support utilities

import functools
import os
import subprocess
from triton import knobs
from triton._C.libtriton import llvm, nano

@functools.lru_cache()
def get_path_to_isa_runtime_dylib():
    """Find the ISA runtime library (currently HIP for AMD)."""
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


def is_isa_backend_active():
    try:
        import torch
        return torch.cuda.is_available() and (torch.version.hip is not None)
    except ImportError:
        return False

class ISACompiler:
    """AMD ISA compiler abstraction for the Nano backend.

    This class wraps AMD-specific compiler functionality from the nano C++ module,
    providing a clean abstraction layer that can be extended for other ISAs.
    """

    def get_binary_extension(self) -> str:
        return "hsaco"

    def get_target_triple(self) -> str:
        return nano.TARGET_TRIPLE

    def get_calling_conv(self) -> int:
        return nano.CALLING_CONV_AMDGPU_KERNEL

    def get_warp_size(self, arch: str) -> int:
        gfx_major = int(arch[3:-2])
        return 32 if gfx_major >= 10 else 64

    def attach_target_triple(self, llvm_mod) -> None:
        nano.attach_target_triple(llvm_mod)

    def attach_datalayout(self, llvm_mod, arch: str, target_features: str = '') -> None:
        llvm.attach_datalayout(llvm_mod, nano.TARGET_TRIPLE, arch, target_features)

    def set_isa_version(self, llvm_mod, arch: str) -> None:
        nano.set_isa_version(llvm_mod, arch)

    def set_abi_version(self, llvm_mod, version: int) -> None:
        nano.set_abi_version(llvm_mod, version)

    def set_bool_control_constant(self, llvm_mod, name: str, value: bool) -> None:
        nano.set_bool_control_constant(llvm_mod, name, value)

    def set_wavefront_size(self, llvm_mod, warp_size: int) -> None:
        self.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", warp_size == 64)

    def set_kernel_attributes(self, fn, options, total_warps_num: int) -> None:
        fn.set_calling_conv(self.get_calling_conv())
        fn.add_fn_attr("amdgpu-flat-work-group-size", f"1,{total_warps_num*options.warp_size}")
        fn.add_fn_attr("uniform-work-group-size", "true")
        fn.add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}, {options.waves_per_eu}")
        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        fn.add_fn_attr("denormal-fp-math-f32", denormal_mode)

    def has_architected_sgprs(self, arch: str) -> bool:
        return nano.has_architected_sgprs(arch)

    def set_all_fn_arg_inreg(self, fn, arch: str) -> None:
        if arch != "gfx1250":
            nano.set_all_fn_arg_inreg(fn)

    def remove_workgroup_id_attrs(self, fn) -> None:
        fn.remove_fn_attr("amdgpu-no-workgroup-id-x")
        fn.remove_fn_attr("amdgpu-no-workgroup-id-y")
        fn.remove_fn_attr("amdgpu-no-workgroup-id-z")

    def need_extern_lib(self, llvm_mod, lib_name: str) -> bool:
        return nano.need_extern_lib(llvm_mod, lib_name)

    def cleanup_module_metadata(self, llvm_mod) -> None:
        nano.cleanup_bitcode_metadata(llvm_mod)

    def disable_print_inline(self, llvm_mod) -> None:
        nano.disable_print_inline(llvm_mod)

    def get_real_true16_feature(self, arch: str) -> str:
        return '-real-true16' if 'gfx11' in arch else ''

    def assemble_isa(self, assembly: str, arch: str, target_features: str = '') -> bytes:
        return nano.assemble_amdgcn(assembly, arch, target_features)

    def link_binary(self, in_path: str, out_path: str) -> None:
        nano.link_hsaco(in_path, out_path)

    def translate_to_asm(self, src: str, arch: str, features: str, flags: list,
                         enable_fp_fusion: bool) -> str:
        return llvm.translate_to_asm(src, self.get_target_triple(), arch, features, flags,
                                     enable_fp_fusion, False)
