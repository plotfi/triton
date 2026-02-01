#ifndef TRITON_ANALYSIS_AXISINFO_H
#define TRITON_ANALYSIS_AXISINFO_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"

#include <optional>

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

/// This lattice value represents known information on the axes of a lattice.
class AxisInfo {
public:
  typedef SmallVector<int64_t> DimVectorT;

public:
  AxisInfo() : AxisInfo({}, {}, {}) {}

  AxisInfo(ArrayRef<int64_t> contiguity, ArrayRef<int64_t> divisibility,
           ArrayRef<int64_t> constancy)
      : AxisInfo(contiguity, divisibility, constancy, std::nullopt) {}

  AxisInfo(ArrayRef<int64_t> contiguity, ArrayRef<int64_t> divisibility,
           ArrayRef<int64_t> constancy, std::optional<int64_t> constantValue)
      : contiguity(contiguity), divisibility(divisibility),
        constancy(constancy), constantValue(constantValue) {
    assert(divisibility.size() == contiguity.size());
    assert(constancy.size() == contiguity.size());
  }

  int64_t getContiguity(size_t dim) const { return contiguity[dim]; }
  const DimVectorT &getContiguity() const { return contiguity; }

  int64_t getDivisibility(size_t dim) const { return divisibility[dim]; }
  const DimVectorT &getDivisibility() const { return divisibility; }

  int64_t getConstancy(size_t dim) const { return constancy[dim]; }
  const DimVectorT &getConstancy() const { return constancy; }

  int getRank() const { return contiguity.size(); }

  std::optional<int64_t> getConstantValue() const { return constantValue; }

  static void initPessimisticStateFromFunc(int argNumber,
                                           FunctionOpInterface funcOp,
                                           DimVectorT *contiguity,
                                           DimVectorT *divisibility,
                                           DimVectorT *constancy);

  static void initDimVectorFromHint(Attribute attr, DimVectorT *vec);

  bool operator==(const AxisInfo &other) const {
    return contiguity == other.contiguity &&
           divisibility == other.divisibility && constancy == other.constancy &&
           constantValue == other.constantValue;
  }

  static AxisInfo getPessimisticValueState(Value value);

  // The gcd of both arguments for each dimension
  static AxisInfo join(const AxisInfo &lhs, const AxisInfo &rhs);

  void print(raw_ostream &os) const {
    auto print = [&](StringRef name, DimVectorT vec) {
      os << name << " = [";
      llvm::interleaveComma(vec, os);
      os << "]";
    };
    print("contiguity", contiguity);
    print(", divisibility", divisibility);
    print(", constancy", constancy);
    os << ", constant_value = ";
    if (constantValue)
      os << *constantValue;
    else
      os << "<none>";
  }

private:
  DimVectorT contiguity;
  DimVectorT divisibility;
  DimVectorT constancy;

  // The constant value of the lattice if we can infer it.
  std::optional<int64_t> constantValue;
};

class AxisInfoVisitor {
public:
  AxisInfoVisitor() = default;
  virtual ~AxisInfoVisitor() = default;

  bool isContiguousDim(const AxisInfo &info, ArrayRef<int64_t> shape, int dim) {
    return info.getContiguity(dim) == shape[dim];
  }

  bool isConstantDim(const AxisInfo &info, ArrayRef<int64_t> shape, int dim) {
    return info.getConstancy(dim) == shape[dim];
  }

  virtual AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) = 0;

  virtual bool match(Operation *op) = 0;
};

class AxisInfoVisitorList {
public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void append() {
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  AxisInfo apply(Operation *op,
                 ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) {
    for (auto &visitor : visitors)
      if (visitor->match(op))
        return visitor->getAxisInfo(op, operands);
    return AxisInfo();
  }

private:
  std::vector<std::unique_ptr<AxisInfoVisitor>> visitors;
};

namespace axisinfo {
using CallbackType = std::function<void(AxisInfoVisitorList &)>;
} // namespace axisinfo

using AxisInfoMapT = DenseMap<Value, AxisInfo>;
class ModuleAxisInfoAnalysis : public CallGraph<AxisInfoMapT> {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp,
                                  axisinfo::CallbackType callback = nullptr)
      : CallGraph<AxisInfoMapT>(moduleOp) {
    SmallVector<FunctionOpInterface> funcs;
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order edge walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order node walk callback
        [&](FunctionOpInterface funcOp) {
          funcs.push_back(funcOp);
          funcMap.try_emplace(funcOp, AxisInfoMapT{});
        });
    SetVector<FunctionOpInterface> sortedFuncs(funcs.begin(), funcs.end());
    SymbolTableCollection symbolTable;
    for (auto funcOp : llvm::reverse(sortedFuncs)) {
      initialize(funcOp, callback);
      funcOp.walk([&](CallOpInterface callOp) {
        auto callee = dyn_cast<FunctionOpInterface>(
            callOp.resolveCallableInTable(&symbolTable));
        update(callOp, callee);
      });
    }
  }

  AxisInfo *getAxisInfo(Value value) {
    auto funcOp =
        value.getParentRegion()->getParentOfType<FunctionOpInterface>();
    auto *axisInfoMap = getFuncData(funcOp);
    if (!axisInfoMap) {
      return nullptr;
    }
    auto it = axisInfoMap->find(value);
    if (it == axisInfoMap->end()) {
      return nullptr;
    }
    return &(it->second);
  }

  unsigned getContiguity(Value value);
  unsigned getAlignment(Value value);

  unsigned getContiguity(Value offsetsValue, unsigned elementBitWidth);
  unsigned getAlignment(Value offsetsValue, unsigned elementBitWidth);

  unsigned getMaskAlignment(Value mask);

private:
  void initialize(FunctionOpInterface funcOp,
                  axisinfo::CallbackType callback = nullptr);
  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
};
} // namespace mlir::triton

#endif
