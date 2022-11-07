/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DisposablePool.hpp ------------------------===//
//
// DisposablePool manages instances of DisposableElementsAttr.
// It creates them, maintains a record of them until they are deemed
// unreachable, and it can be called to garbage collect those that are
// unreachable, and to replace them with DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"

#include <unordered_set>

namespace onnx_mlir {

class DisposablePool : public mlir::DialectInterface::Base<DisposablePool> {
public:
  static DisposablePool &create(mlir::MLIRContext *context);

  static DisposablePool *get(mlir::MLIRContext *context);

  DisposablePool(mlir::Dialect *dialect, mlir::MLIRContext *context);
  ~DisposablePool();

  template <typename... Args>
  mlir::DisposableElementsAttr createElementsAttr(Args &&...args) {
    auto d = mlir::DisposableElementsAttr::get(std::forward<Args>(args)...);
    insert(d);
    return d;
  }

  void garbageCollectUnreachable(mlir::ModuleOp moduleOp);

  void scrub(mlir::ModuleOp moduleOp);

  void close() {
    assert(pool.empty() && "pool must be scrubbed before close ");
    active = false;
  }

  bool isActive() const { return active; }

private:
  using Pool = std::unordered_set<mlir::DisposableElementsAttributeStorage *>;

  void insert(mlir::DisposableElementsAttr d);
  void eraseUnreachable(const Pool &reachable);

  Pool pool;
  bool active = true;
};

} // namespace onnx_mlir
