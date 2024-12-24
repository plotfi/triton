#include "LLVMPasses.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"

using namespace llvm;

static bool runOnFunction(Function &F) {
  SmallVector<CallInst *, 8> eraseInsts;

  bool Changed = false;
  for (BasicBlock &BB : F) {
    for (Instruction &inst : BB) {
      if (auto *callInst = dyn_cast<CallInst>(&inst)) {

        if (!callInst->getType()->isFloatingPointTy() &&
            !callInst->getType()->isIntegerTy())
          continue;

        if (callInst->getNumOperands() < 2)
          continue;

        bool hasPredicate = callInst->getNumOperands() == 4;
        auto value = dyn_cast<llvm::Constant>(callInst->getOperand(1));
        if (!value || !value->isZeroValue())
          continue;

        IRBuilder<> builder(callInst);

        if (auto *calleeInlineAsm =
                dyn_cast<llvm::InlineAsm>(callInst->getCalledOperand())) {
          StringRef asmString = calleeInlineAsm->getAsmString();

          StringRef movStr = "";

          if (asmString.starts_with("mov")) {
            auto movEnd = asmString.find(";") + 1;
            movStr = asmString.substr(0, movEnd);
            asmString = asmString.substr(movEnd).ltrim();
          }

          if (!asmString.contains("atom"))
            continue;

          if (!asmString.starts_with("atom") && !asmString.starts_with("@$"))
            continue;

          if (asmString.starts_with("@$")) {
            asmString = asmString.substr(strlen("@$")).ltrim();

            int skip = 0;
            for (char c : asmString) {
              if (!('0' <= c && c <= '9'))
                break;
              skip++;
            }
            asmString = asmString.substr(skip).ltrim();
          }

          int keep = 0;
          for (char c : asmString) {
            bool bail = false;
            switch (c) {
            default:
              keep++;
              break;
            case ' ':
            case '\t':
            case '\n':
            case '\v':
            case '\f':
            case '\r':
              bail = true;
              break;
            }

            if (bail)
              break;
          }

          StringRef asmMnemonic = asmString.substr(0, keep).trim();

          if (!asmMnemonic.contains("add") && !asmMnemonic.contains("fadd"))
            continue;

          SmallVector<StringRef, 8> mnemonics;
          asmMnemonic.split(mnemonics, '.');

          std::string ldMnemonic = "ld";
          for (auto mnemonic : mnemonics) {
            if (mnemonic == "global" || mnemonic == "shared" ||
                mnemonic == "gpu" || mnemonic == "cta" || mnemonic == "sys" ||
                mnemonic == "acquire" || mnemonic == "relaxed") {
              ldMnemonic += '.';
              ldMnemonic += mnemonic;
            }
          }

          ldMnemonic += '.';
          ldMnemonic += 'b';

          switch (callInst->getType()->getScalarSizeInBits()) {
          case 8:
            ldMnemonic += "8";
            break;
          case 16:
            ldMnemonic += "16";
            break;
          case 32:
            ldMnemonic += "32";
            break;
          default:
            continue;
          }

          std::string newInst = "";
          if (hasPredicate) {
            newInst += movStr.str();
            newInst += "\n\t";
            newInst += "@$2 ";
          }

          newInst += ldMnemonic;
          newInst += " $0, [ $1 + 0 ];";

          //// Do the replacement

          auto newFunctionType = FunctionType::get(
              calleeInlineAsm->getFunctionType()->getReturnType(),
              {
                  calleeInlineAsm->getFunctionType()->getParamType(0),
                  calleeInlineAsm->getFunctionType()->getParamType(2),
              },
              false);

          FunctionCallee newInlineAsm =
              InlineAsm::get(newFunctionType,
                             // calleeInlineAsm->getFunctionType(),
                             newInst, "=r,l,b",
                             // calleeInlineAsm->getConstraintString(),
                             calleeInlineAsm->hasSideEffects(),
                             calleeInlineAsm->isAlignStack());

          builder.SetInsertPoint(callInst);
          auto *newCallInst = builder.CreateCall(
              newInlineAsm.getFunctionType(), newInlineAsm.getCallee(),
              {callInst->getOperand(0), callInst->getOperand(2)});

          callInst->replaceAllUsesWith(newCallInst);
          eraseInsts.push_back(callInst);
          Changed |= true;
        }
      }
    }
  }

  for (CallInst *eraseInst : eraseInsts) {
    eraseInst->eraseFromParent();
  }

  return Changed;
}

PreservedAnalyses InlineAsmRewritePass::run(Function &F,
                                            FunctionAnalysisManager &AM) {

  bool b = runOnFunction(F);
  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
