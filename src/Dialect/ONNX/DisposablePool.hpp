/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- DisposablePool.hpp ------------------------===//
//
// DisposablePool manages instances of DisposableElementsAttr.
// It creates them, maintains a record of them (in a "pool") until they are
// deemed unreachable, and it can be called to garbage collect those that are
// unreachable, and to "scrub" all occurrences in a module by replacing each
// with a DenseElementsAttr.
//
// Garbage collected and scrubbed DisposableElementsAttrs are removed from the
// pool and their reference to the underlying MemoryBuffer is cleared,
// decrementing the shared_ptr reference count.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"

#include <unordered_map>
#include <unordered_set>

namespace onnx_mlir {

class DisposablePool : public mlir::DialectInterface::Base<DisposablePool> {
  friend class ElementsAttrBuilder; // allow access to insert()
public:
  static DisposablePool &create(mlir::MLIRContext *context);

  static DisposablePool *get(mlir::MLIRContext *context);

  // Disposes every DisposableElementsAttr and in moduleOp replaces each with a
  // DenseElementsAttr.
  static void scrub(mlir::ModuleOp moduleOp, DisposablePool *disposablepool);

  DisposablePool(mlir::Dialect *dialect, mlir::MLIRContext *context);
  ~DisposablePool();

  mlir::DisposableElementsAttr lookup(size_t id) const {
    auto found = map.find(id);
    if (found == map.end())
      return nullptr;
    return found->second;
  }

  // Disposes every DisposableElementsAttr in the pool which is unreachable
  // (doesn't appear in moduleOp).
  void garbageCollectUnreachable(mlir::ModuleOp moduleOp);

  void close() {
    assert(pool.empty() && "pool must be scrubbed before close");
    active = false;
  }

  bool isActive() const { return active; }

private:
  using Item = mlir::DisposableElementsAttributeStorage *;
  using Pool = std::unordered_set<Item>;
  using Scrubbed = std::unordered_map<Item, mlir::DenseElementsAttr>;

  using Map = std::unordered_map<size_t, mlir::DisposableElementsAttr>;

  void insert(mlir::DisposableElementsAttr disposable);

  static Scrubbed doScrub(mlir::ModuleOp moduleOp);

  void flushAfterScrub(const Scrubbed &scrubbed);

  void eraseUnreachable(const Pool &reachable);

  Pool pool; // TODO: remove pool and just use map for everything
  Map map;
  bool active = true;
};

} // namespace onnx_mlir
