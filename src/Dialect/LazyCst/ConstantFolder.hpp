/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <memory>
#include <vector>

namespace lazycst {

class ConstantFoldableAnalysis;

// Similar to ConversionPattern and Operation::fold().
// Every instance needs to implement both 'match' and 'fold'.
// Unlike ConversionPattern it's not enough to implement a combined
// 'matchAndFold' method because ConstantFoldableAnalysis needs to call
// 'match' without 'fold'.
class ConstantFolder {
public:
  using Fn =
      std::function<void(mlir::Operation *, llvm::ArrayRef<mlir::Attribute>,
          llvm::SmallVectorImpl<mlir::Attribute> &)>;

  virtual ~ConstantFolder() = default;

  virtual mlir::LogicalResult match(mlir::Operation *op) const = 0;
  virtual void fold(mlir::Operation *op,
      llvm::ArrayRef<mlir::Attribute> operands,
      llvm::SmallVectorImpl<mlir::Attribute> &results) const = 0;

  Fn fn() const {
    return [this](mlir::Operation *op, llvm::ArrayRef<mlir::Attribute> operands,
               llvm::SmallVectorImpl<mlir::Attribute> &results) {
      fold(op, operands, results);
    };
  }

  mlir::LogicalResult matchAndFold(mlir::Operation *op,
      llvm::ArrayRef<mlir::Attribute> operands,
      llvm::SmallVectorImpl<mlir::Attribute> &results) const {
    if (mlir::failed(match(op)))
      return mlir::failure();
    fold(op, operands, results);
    return mlir::success();
  }
};

template <typename OP>
class OpConstantFolder : public ConstantFolder {
public:
  using Op = OP;
  using FoldAdaptor = typename OP::FoldAdaptor;

  virtual ~OpConstantFolder() = default;

  virtual mlir::LogicalResult match(OP op) const { return mlir::success(); }
  mlir::LogicalResult match(mlir::Operation *op) const override final {
    return match(llvm::cast<OP>(op));
  }

  virtual mlir::Attribute fold(OP op, FoldAdaptor adaptor) const {
    llvm_unreachable("unimplemented");
  }
  virtual void fold(OP op, FoldAdaptor adaptor,
      llvm::SmallVectorImpl<mlir::Attribute> &results) const {
    results.emplace_back(fold(op, adaptor));
  }
  virtual void fold(mlir::Operation *op,
      llvm::ArrayRef<mlir::Attribute> operands,
      llvm::SmallVectorImpl<mlir::Attribute> &results) const override final {
    return fold(llvm::cast<OP>(op),
        FoldAdaptor(operands, op->getAttrDictionary(),
            op->getPropertiesStorage(), op->getRegions()),
        results);
  }
};

class ConstantFolders {
public:
  using PatternsGetter = std::function<void(
      ConstantFoldableAnalysis &, mlir::RewritePatternSet &)>;

  const ConstantFolder *lookup(llvm::StringRef opName) const {
    auto it = map.find(opName);
    return it == map.end() ? nullptr : it->second.get();
  }

  const ConstantFolder *lookup(mlir::OperationName opName) const {
    return lookup(opName.getIdentifier());
  }

  ConstantFolders &insert(
      llvm::StringRef opName, std::unique_ptr<ConstantFolder> constantFolder) {
    auto [_, inserted] = map.try_emplace(opName, std::move(constantFolder));
    assert(inserted);
    return *this;
  }

  template <class CONSTANT_FOLDER, typename... Args>
  ConstantFolders &insert(llvm::StringRef opName, Args &&... args) {
    return insert(
        opName, std::make_unique<CONSTANT_FOLDER>(std::forward<Args>(args)...));
  }

  template <class CONSTANT_FOLDER, typename... Args>
  ConstantFolders &insert(mlir::OperationName opName, Args &&... args) {
    return insert<CONSTANT_FOLDER, Args...>(
        opName.getIdentifier(), std::forward<Args>(args)...);
  }

  template <class OP_CONSTANT_FOLDER, typename... Args>
  ConstantFolders &insertOpConstantFolder(Args &&... args) {
    return insert<OP_CONSTANT_FOLDER, Args...>(
        OP_CONSTANT_FOLDER::Op::getOperationName(),
        std::forward<Args>(args)...);
  }

  void addPatternsGetter(PatternsGetter getter) {
    patternsGetters.emplace_back(std::move(getter));
  }

  void getPatterns(ConstantFoldableAnalysis &analysis,
      mlir::RewritePatternSet &results) const;

  mlir::LogicalResult match(mlir::Operation *op) const {
    if (const ConstantFolder *constantFolder = lookup(op->getName()))
      return constantFolder->match(op);
    return mlir::failure();
  }

private:
  // Keyed by operation name, i.e.
  // OP::getOperationName() or OperationName::getIdentifier().
  // TODO: consider using TypeID as key, i.e.
  //       TypeID::get<OP>() or  OperationName::getTypeID()
  llvm::StringMap<std::unique_ptr<ConstantFolder>> map;

  std::vector<PatternsGetter> patternsGetters;
};

} // namespace lazycst
