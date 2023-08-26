/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace lazycst {

// Similar to ConversionPattern and Operation::fold().
// Every instance needs to implement both 'match' and 'fold'.
// Unlike ConversionPattern it's not enough to implement a combined
// 'matchAndFold' method because ConstantFoldableAnalysis needs to call
// 'match' without 'fold'.
class LazyFolder {
public:
  virtual ~LazyFolder() = default;

  virtual mlir::LogicalResult match(mlir::Operation *op) const = 0;
  virtual void fold(mlir::Operation *op,
      llvm::ArrayRef<mlir::Attribute> operands,
      llvm::SmallVectorImpl<mlir::Attribute> &results) const = 0;

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
class OpLazyFolder : public LazyFolder {
public:
  using Op = OP;
  using FoldAdaptor = typename OP::FoldAdaptor;

  virtual ~OpLazyFolder() = default;

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

class LazyFolders {
public:
  mlir::LogicalResult match(mlir::Operation *op) const {
    if (const LazyFolder *lazyFolder = lookup(op->getName()))
      return lazyFolder->match(op);
    return mlir::failure();
  }

  const LazyFolder *lookup(llvm::StringRef opName) const {
    auto it = map.find(opName);
    return it == map.end() ? nullptr : it->second.get();
  }

  const LazyFolder *lookup(mlir::OperationName opName) const {
    return lookup(opName.getIdentifier());
  }

  LazyFolders &insert(
      llvm::StringRef opName, std::unique_ptr<LazyFolder> lazyFolder) {
    auto [_, inserted] = map.try_emplace(opName, std::move(lazyFolder));
    assert(inserted);
    return *this;
  }

  template <class LAZY_FOLDER, typename... Args>
  LazyFolders &insert(llvm::StringRef opName, Args &&... args) {
    return insert(
        opName, std::make_unique<LAZY_FOLDER>(std::forward<Args>(args)...));
  }

  template <class LAZY_FOLDER, typename... Args>
  LazyFolders &insert(mlir::OperationName opName, Args &&... args) {
    return insert<LAZY_FOLDER, Args...>(
        opName.getIdentifier(), std::forward<Args>(args)...);
  }

  template <class OP_LAZY_FOLDER, typename... Args>
  LazyFolders &insertOpLazyFolder(Args &&... args) {
    return insert<OP_LAZY_FOLDER, Args...>(
        OP_LAZY_FOLDER::Op::getOperationName(), std::forward<Args>(args)...);
  }

private:
  // Keyed by operation name, i.e.
  // OP::getOperationName() or OperationName::getIdentifier().
  // TODO: consider using TypeID as key, i.e.
  //       TypeID::get<OP>() or  OperationName::getTypeID()
  llvm::StringMap<std::unique_ptr<LazyFolder>> map;
};

} // namespace lazycst