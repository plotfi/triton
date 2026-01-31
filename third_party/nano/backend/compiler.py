# Triton Nano Backend - Minimal backend for simple kernels like vector add
# Based on AMD backend, simplified for educational and prototyping purposes

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, nano
from triton import knobs
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import os
import hashlib
import tempfile
import re
import functools
import warnings
from pathlib import Path


def get_min_dot_size(target: GPUTarget):
    # We fallback to use FMA and cast arguments if certain configurations is
    # not supported natively by matrix core units.
    return lambda lhs_type, rhs_type: (1, 1, 1)


@dataclass(frozen=True)
class NanoOptions:
    """Simplified options for the Nano backend - minimal configuration for simple kernels."""
    num_warps: int = 4
    waves_per_eu: int = 0
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = None
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5", "fp8e5b16", "fp8e4b8")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = 'nano'
    instrumentation_mode: str = ""
    schedule_hint: str = 'none'

    def __post_init__(self):
        gfx_major = int(self.arch[3:-2])  # Drop "gfx" prefix and minor/patch number
        warp_size = 32 if gfx_major >= 10 else 64
        object.__setattr__(self, 'warp_size', warp_size)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

        # No external libraries needed for basic kernels
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class NanoBackend(BaseBackend):
    """Minimal Triton backend for simple kernels like vector add."""
    instrumentation = None
    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'nano'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"

    def get_target_name(self, options) -> str:
        return f"nano:{options.arch}"

    def parse_options(self, opts) -> Any:
        args = {'arch': knobs.runtime.override_arch or self.target.arch}

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        args.update({k: opts[k] for k in NanoOptions.__dataclass_fields__.keys() if k in opts and opts[k] is not None})
        return NanoOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": get_min_dot_size(self.target)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        nano.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, options):
        """Convert Triton IR to optimized Triton IR - simplified for basic ops."""
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        """Convert Triton IR to Triton GPU IR - simplified pipeline for basic ops."""
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # Use nano backend target string
        passes.ttir.add_convert_to_ttgpuir(pm, f"nano:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
        pm.run(mod, 'make_ttgir_early')

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_canonicalizer(pm)

        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)

        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, 'make_ttgir')
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        """Convert Triton GPU IR to LLVM IR."""
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        passes.convert.add_index_to_llvmir(pm)

        # nano.passes.ttgpuir.add_allocate_shared_memory(pm)

        __NANO_FTZ = True
        nano.passes.ttgpuir.add_to_llvmir(pm, options.arch, __NANO_FTZ)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if not knobs.compilation.disable_line_info:
            passes.llvmir.add_di_scope(pm)

        pm.run(mod, 'make_llir')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        nano.attach_target_triple(llvm_mod)
        target_features = ''
        llvm.attach_datalayout(llvm_mod, nano.TARGET_TRIPLE, options.arch, target_features)

        # Set various control constants
        nano.set_isa_version(llvm_mod, options.arch)
        nano.set_abi_version(llvm_mod, 500)
        nano.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
        nano.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
        nano.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
        nano.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64)

        # Set kernel attributes
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        fns[0].set_calling_conv(nano.CALLING_CONV_AMDGPU_KERNEL)
        total_warps_num = options.num_warps
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            total_warps_num = total_num_warps
        fns[0].add_fn_attr("amdgpu-flat-work-group-size", f"1,{total_warps_num*options.warp_size}")
        fns[0].add_fn_attr("uniform-work-group-size", "true")
        fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}, {options.waves_per_eu}")
        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        fns[0].add_fn_attr("denormal-fp-math-f32", denormal_mode)

        if options.arch != "gfx1250":
            nano.set_all_fn_arg_inreg(fns[0])

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs if nano.need_extern_lib(llvm_mod, name)]
            if len(paths) > 0:
                llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)

        if nano.has_architected_sgprs(options.arch):
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-x")
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-y")
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-z")

        # Get metadata
        metadata["num_warps"] = total_warps_num
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1

        nano.cleanup_bitcode_metadata(llvm_mod)
        nano.disable_print_inline(llvm_mod)
        return str(llvm_mod)

    @staticmethod
    def make_amdgcn(src, metadata, options):
        """Convert LLVM IR to AMDGCN assembly."""
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]

        flags = []
        features = '-real-true16' if 'gfx11' in options.arch else ''
        amdgcn = llvm.translate_to_asm(src, nano.TARGET_TRIPLE, options.arch, features, flags,
                                       options.enable_fp_fusion, False)
        return amdgcn

    @staticmethod
    def make_hsaco(src, metadata, options):
        """Assemble AMDGCN to HSACO binary."""
        target_features = ''
        hsaco = nano.assemble_amdgcn(src, options.arch, target_features)
        with tempfile.NamedTemporaryFile() as tmp_out:
            with tempfile.NamedTemporaryFile() as tmp_in:
                with open(tmp_in.name, "wb") as fd_in:
                    fd_in.write(hsaco)
                nano.link_hsaco(tmp_in.name, tmp_out.name)
            with open(tmp_out.name, "rb") as fd_out:
                ret = fd_out.read()
        return ret

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f'{self.target}'
