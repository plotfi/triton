# Triton Nano Backend - ISA Support utilities

import functools
import os
import subprocess
from triton import knobs


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
    """Check if the ISA backend (currently HIP/AMD) is available."""
    try:
        import torch
        return torch.cuda.is_available() and (torch.version.hip is not None)
    except ImportError:
        return False
