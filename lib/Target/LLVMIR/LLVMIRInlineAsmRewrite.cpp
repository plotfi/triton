#include "LLVMPasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"
#include <algorithm>
#include <cctype>
#include <ios>

using namespace llvm;

static size_t countCharType(StringRef str, int (*fn)(int), bool check) {
  int count = 0;
  for (char c : str) {
    if ((check && fn(c)) || (!check && !fn(c)))
      break;
    count++;
  }
  return count;
}

static bool doAtomToLdOpChecks(const CallInst *callInst) {
  if (!callInst->getType()->isFloatingPointTy() &&
      !callInst->getType()->isIntegerTy())
    return false;

  if (callInst->getNumOperands() < 2)
    return false;

  // Make sure we are dealing with an atom that is accumulating zero
  auto value = dyn_cast<llvm::Constant>(callInst->getOperand(1));
  if (!value || !value->isZeroValue())
    return false;

  StringRef asmString =
      cast<llvm::InlineAsm>(callInst->getCalledOperand())->getAsmString();
  bool hasPredicate = callInst->getNumOperands() == 4;
  if (hasPredicate != asmString.starts_with("mov"))
    return false;

  return true;
}

static bool doAtomToLdMnemonicChecks(StringRef str) {
  // Make sure we are dealing with an atom that is acquiring
  if (!str.starts_with("atom") || !str.contains("acquire"))
    return false;
  // make sure we are dealing with an atom that is accumulating
  if (!str.contains("add") && !str.contains("fadd"))
    return false;
  return true;
}

static std::tuple<StringRef, StringRef>
parseAtomToLdAsmMovAndMnemonic(StringRef str) {
  StringRef movStr;
  StringRef asmString = str;

  // Chomp off the leading `mov` string (ie `mov.u32 $0, 0x0;`)
  if (asmString.starts_with("mov")) {
    auto movEnd = asmString.find(";") + 1;
    movStr = asmString.substr(0, movEnd);
    asmString = asmString.substr(movEnd).ltrim();
  }

  // Chomp off the leading predicate, ie  leading @$[0-9]+ in:
  // `@$3 atom.global.gpu.acquire.add.f32 $0, [ $1 + 0 ], $2;`
  //  ^^^^
  //  So that we can proceed with the Mnenomic handling
  if (asmString.starts_with("@$")) {
    asmString = asmString.substr(strlen("@$")).ltrim();
    const int digitSkips = countCharType(asmString, isdigit, false);
    asmString = asmString.substr(digitSkips).ltrim();
  }

  // Find the first space in:
  // `atom.global.gpu.acquire.add.f32 $0, [ $1 + 0 ], $2;`
  //                                 ^
  // This allows is to chomp off the Mnemonic
  const int spaceIndex = countCharType(asmString, isspace, true);
  StringRef asmMnemonic = asmString.substr(0, spaceIndex).trim();

  return {movStr, asmMnemonic};
}

std::string rewriteAtomToLdMnemonic(StringRef str) {
  SmallVector<StringRef, 8> mnemonics;
  str.split(mnemonics, '.');

  std::string newMnemonic;
  llvm::raw_string_ostream OS(newMnemonic);
  SmallVector<StringRef, 8> leftoverMnemonics;

  std::vector<StringRef> OneToOneMnemonics = {
      "global", "shared", "gpu", "cta", "sys", "acquire", "relaxed"};
  std::vector<StringRef> TypeMnemonicsB8 = {"s8", "u8", "b8"};
  std::vector<StringRef> TypeMnemonicsB16 = {"s16", "u16", "f16", "b16"};
  std::vector<StringRef> TypeMnemonicsB32 = {"s32", "u32", "f16x2", "f32",
                                             "b32"};
  std::vector<StringRef> TypeMnemonicsB64 = {"s64", "u64", "f64", "b64"};

  std::vector<std::pair<std::vector<StringRef>, unsigned>> typeMap = {
      {TypeMnemonicsB8, 8},   {TypeMnemonicsB16, 16}, {TypeMnemonicsB32, 32},
      {TypeMnemonicsB64, 64}, {{"b128"}, 128},
  };

  bool bail = false;
  for (auto mnemonic : mnemonics) {

    unsigned bitWidth = 0;
    for (auto entry : typeMap) {
      bitWidth +=
          llvm::any_of(entry.first,
                       [mnemonic](StringRef str) { return str == mnemonic; }) *
          entry.second;
    }

    if (bitWidth) {
      // NOTE: PTX errors when ld.acquire is used with non-b32 types
      if (bitWidth != 32) {
        bail = true;
        break;
      }
      OS << ".b" << bitWidth;
    } else if (llvm::any_of(OneToOneMnemonics, [mnemonic](StringRef str) {
                 return str == mnemonic;
               })) {
      OS << '.' << mnemonic;
    } else if (mnemonic == "atom" || mnemonic == "fadd" || mnemonic == "add") {
      continue;
    } else {
      leftoverMnemonics.push_back(mnemonic);
    }
  }

  // If we didn't process every sub-opcode in the Mnemonic then bail
  if (bail || leftoverMnemonics.size())
    return "";
  return OS.str();
}

static bool
runOnInlinePTXAtomInstruction(CallInst *I,
                              SmallVector<CallInst *, 8> &eraseInsts) {

  auto *calleeInlineAsm = cast<llvm::InlineAsm>(I->getCalledOperand());

  if (!doAtomToLdOpChecks(I))
    return false;

  StringRef movStr;
  StringRef asmMnemonic;
  std::tie(movStr, asmMnemonic) =
      parseAtomToLdAsmMovAndMnemonic(calleeInlineAsm->getAsmString());

  if (!doAtomToLdMnemonicChecks(asmMnemonic))
    return false;

  auto newMnemonic = rewriteAtomToLdMnemonic(asmMnemonic);
  if ("" == newMnemonic)
    return false;

  std::string newInlinePtxInst;
  llvm::raw_string_ostream OS(newInlinePtxInst);
  if (I->getNumOperands() == 4 /* hasPredicate */)
    OS << movStr.str() << "\n\t" << "@$2 ";
  OS << "ld" << newMnemonic << " $0, [ $1 + 0 ];";

  /// InlineAsm Instruction for PTX atom.acquire with a PTX ld.acquire
  FunctionCallee newInlineAsm = InlineAsm::get(
      FunctionType::get(calleeInlineAsm->getFunctionType()->getReturnType(),
                        {
                            calleeInlineAsm->getFunctionType()->getParamType(0),
                            calleeInlineAsm->getFunctionType()->getParamType(2),
                        },
                        false),
      // calleeInlineAsm->getFunctionType(),
      newInlinePtxInst, "=r,l,b",
      // calleeInlineAsm->getConstraintString(),
      calleeInlineAsm->hasSideEffects(), calleeInlineAsm->isAlignStack());

  // Replace the old CallInst with a new one to the new inline PTX
  IRBuilder<> builder(I);
  builder.SetInsertPoint(I);
  auto *newCallInst = builder.CreateCall(newInlineAsm.getFunctionType(),
                                         newInlineAsm.getCallee(),
                                         {I->getOperand(0), I->getOperand(2)});
  I->replaceAllUsesWith(newCallInst);
  eraseInsts.push_back(I);
  return true;
}

static bool runOnFunction(Function &F) {
  SmallVector<CallInst *, 8> eraseInsts;

  bool Changed = false;
  for (BasicBlock &BB : F) {
    for (Instruction &inst : BB) {
      auto *I = dyn_cast<CallInst>(&inst);
      if (!I || !isa<llvm::InlineAsm>(I->getCalledOperand()))
        continue;

      Changed |= runOnInlinePTXAtomInstruction(I, eraseInsts);
    }
  }

  for (CallInst *I : eraseInsts)
    I->eraseFromParent();
  return Changed;
}

PreservedAnalyses InlineAsmRewritePass::run(Function &F,
                                            FunctionAnalysisManager &AM) {

  bool b = runOnFunction(F);
  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
