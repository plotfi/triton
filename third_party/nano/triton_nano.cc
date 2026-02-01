// Triton Nano Backend - Minimal backend for simple kernels like vector add
// Based on AMD backend, simplified for educational and prototyping purposes

#include "TritonNANOGPUToLLVM/Passes.h"
#include "TritonNANOGPUToLLVM/TargetUtils.h"
#include "lib/TritonNANOGPUToLLVM/TargetInfo.h"
#include "lld/Common/Driver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/TargetParser.h"
#include <array>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace {
const char *const nanoTargetTriple = "amdgcn-amd-amdhsa";

void init_triton_nano_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, const std::string &arch, bool ftz) {
          pm.addPass(createConvertTritonNANOGPUToLLVMPass(arch, ftz));
        });
}

void addControlConstant(llvm::Module *module, const char *name,
                        uint32_t bitwidth, uint32_t value) {
  using llvm::GlobalVariable;

  llvm::IntegerType *type =
      llvm::IntegerType::getIntNTy(module->getContext(), bitwidth);
  auto *initializer = llvm::ConstantInt::get(type, value, /*isSigned=*/false);
  auto *constant = new llvm::GlobalVariable(
      *module, type, /*isConstant=*/true,
      GlobalVariable::LinkageTypes::LinkOnceODRLinkage, initializer, name,
      /*before=*/nullptr, GlobalVariable::ThreadLocalMode::NotThreadLocal,
      /*addressSpace=*/4);
  constant->setAlignment(llvm::MaybeAlign(bitwidth / 8));
  constant->setUnnamedAddr(GlobalVariable::UnnamedAddr::Local);
  constant->setVisibility(GlobalVariable::VisibilityTypes::ProtectedVisibility);
}

} // namespace

LLD_HAS_DRIVER(elf)

static std::optional<std::string> lldInvoke(const char *inPath,
                                            const char *outPath) {
  // Workaround: Disable parallelism to avoid hangs caused by LLVM's thread pool
  // when the following code is executed in a forked child process.
  std::array args{"ld.lld", "--threads=1", "-shared", inPath, "-o", outPath};
  std::string errString;
  llvm::raw_string_ostream errStream(errString);
  auto lldRes = lld::lldMain(args, llvm::outs(), llvm::errs(),
                             {{lld::Gnu, &lld::elf::link}});
  bool noErrors = (!lldRes.retCode && lldRes.canRunAgain);
  if (!noErrors) {
    errStream.flush();
    return errString;
  }
  return {};
}

void init_triton_nano(py::module &&m) {
  m.doc() = "Python bindings to the Triton Nano backend (minimal backend for simple kernels)";

  auto passes = m.def_submodule("passes");
  init_triton_nano_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  m.attr("TARGET_TRIPLE") = nanoTargetTriple;
  m.attr("CALLING_CONV_AMDGPU_KERNEL") =
      (unsigned)llvm::CallingConv::AMDGPU_KERNEL;

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    // TritonNANOGPU dialect removed - using standard TritonGPU dialect
    mlir::registerROCDLDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("attach_target_triple", [](llvm::Module *module) {
    module->setTargetTriple(llvm::Triple(nanoTargetTriple));
  });

  // Set target architecture ISA version
  m.def("set_isa_version", [](llvm::Module *module, const std::string &arch) {
    llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(arch);
    addControlConstant(module, "__oclc_ISA_version", /*bitwidth=*/32,
                       version.Major * 1000 + version.Minor * 100 +
                           version.Stepping);
  });

  // Set boolean control constant
  m.def("set_bool_control_constant",
        [](llvm::Module *module, const std::string &name, bool enable) {
          addControlConstant(module, name.c_str(), /*bitwidth=*/8, enable);
        });

  // Set code object ABI version
  m.def("set_abi_version", [](llvm::Module *module, int version) {
    llvm::Type *i32Ty = llvm::Type::getInt32Ty(module->getContext());
    llvm::GlobalVariable *abi = new llvm::GlobalVariable(
        *module, i32Ty, /*isConstant=*/true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(i32Ty, version), "__oclc_ABI_version", nullptr,
        llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    abi->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    abi->setAlignment(llvm::MaybeAlign(4));
    abi->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
    module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                          version);
  });

  m.def("cleanup_bitcode_metadata", [](llvm::Module *module) {
    if (auto *ident = module->getNamedMetadata("llvm.ident"))
      module->eraseNamedMetadata(ident);
    if (auto *openclVersion = module->getNamedMetadata("opencl.ocl.version"))
      module->eraseNamedMetadata(openclVersion);
  });

  m.def("disable_print_inline", [](llvm::Module *module) {
    // Printf not supported in minimal nano backend - no ockl functions to modify
  });

  m.def(
      "assemble_amdgcn",
      [](const std::string &assembly, const std::string &arch,
         const std::string &features) {
        std::string error;

        llvm::Triple triple(nanoTargetTriple);
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(triple, error);
        if (!target)
          throw std::runtime_error("target lookup error: " + error);

        llvm::SourceMgr srcMgr;
        srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(assembly),
                                  llvm::SMLoc());

        const llvm::MCTargetOptions mcOptions;
        std::unique_ptr<llvm::MCRegisterInfo> mri(
            target->createMCRegInfo(triple));
        std::unique_ptr<llvm::MCAsmInfo> mai(
            target->createMCAsmInfo(*mri, triple, mcOptions));
        std::unique_ptr<llvm::MCSubtargetInfo> sti(
            target->createMCSubtargetInfo(triple, arch, features));

        llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                            &mcOptions);
        std::unique_ptr<llvm::MCObjectFileInfo> mofi(
            target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                           /*LargeCodeModel=*/false));
        ctx.setObjectFileInfo(mofi.get());

        llvm::SmallString<128> cwd;
        if (!llvm::sys::fs::current_path(cwd))
          ctx.setCompilationDir(cwd);

        llvm::SmallVector<char, 0> result;
        llvm::raw_svector_ostream svos(result);

        std::unique_ptr<llvm::MCStreamer> mcStreamer;
        std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

        std::unique_ptr<llvm::MCCodeEmitter> ce(
            target->createMCCodeEmitter(*mcii, ctx));
        std::unique_ptr<llvm::MCAsmBackend> mab(
            target->createMCAsmBackend(*sti, *mri, mcOptions));
        std::unique_ptr<llvm::MCObjectWriter> ow(mab->createObjectWriter(svos));
        mcStreamer.reset(target->createMCObjectStreamer(
            triple, ctx, std::move(mab), std::move(ow), std::move(ce), *sti));

        std::unique_ptr<llvm::MCAsmParser> parser(
            createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
        std::unique_ptr<llvm::MCTargetAsmParser> tap(
            target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));
        if (!tap)
          throw std::runtime_error("assembler initialization error");

        parser->setTargetParser(*tap);
        parser->Run(/*NoInitialTextSection=*/false);

        return py::bytes(std::string(result.begin(), result.end()));
      },
      py::return_value_policy::take_ownership);

  m.def("has_architected_sgprs", [](const std::string &arch) {
    std::string error;
    llvm::Triple triple(nanoTargetTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target)
      throw std::runtime_error("target lookup error: " + error);
    std::unique_ptr<llvm::MCSubtargetInfo> sti(
        target->createMCSubtargetInfo(triple, arch, ""));
    return sti->checkFeatures("+architected-sgprs");
  });

  m.def("need_extern_lib", [](llvm::Module *module, const std::string &lib) {
    for (llvm::Function &f : module->functions()) {
      if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
        llvm::StringRef funcName = f.getName();
        if (funcName.contains(lib))
          return true;
        if (funcName.contains("__nv_")) {
          std::stringstream message;
          message << "Implicit conversion of CUDA " << funcName.str()
                  << " device function has been dropped; "
                  << "please use triton.language.extra.<op> instead";
          throw std::runtime_error(message.str());
        }
      }
    }
    return false;
  });

  m.def("set_all_fn_arg_inreg", [](llvm::Function *fn) {
    for (llvm::Argument &arg : fn->args()) {
      if (arg.hasByRefAttr() || arg.hasNestAttr())
        continue;
      arg.addAttr(llvm::Attribute::InReg);
    }
  });

  m.def("link_hsaco",
        [](const std::string &inPath, const std::string &outPath) {
          if (auto errString = lldInvoke(inPath.c_str(), outPath.c_str()))
            throw std::runtime_error("LLD failed to link hsaco source " +
                                     inPath + " into object file " + outPath +
                                     " because " + errString.value());
        });

  // m.def("add_scalarize_packed_fops_llvm_pass", [](llvm::Function *fn) {
  //   mlir::triton::NANO::runScalarizePackedFOpsPass(*fn);
  // });
}
