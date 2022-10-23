/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXAttributes.hpp - ONNX Attributes ----------------===//
//
// This file defines attributes in the ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/ONNXDialect.hpp"

#include "src/Dialect/Mlir/DType.hpp"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"

#include <memory>

namespace onnx_mlir {

// The buffer can be mmap'ed to a file or be heap allocated or point into
// a DenseElementsAttr buffer.
// (Potential problem: MemoryBuffer doesn't guarantee alignment.)
class DisposableElementsImpl {
public:
  // ElementIterator<T>, Bool/Int/FloatElementIterator, like DenseElementsAttr
  using DataType = DTYPE::DataType;
  using DisposableElements = std::shared_ptr<DisposableElementsImpl>;

  static llvm::SmallVector<int64_t, 4> calculateStrides(llvm::ArrayRef<int64_t> shape);
  static size_t sizeOfDataType(DataType);

  // move to DType.h
  union RawDatum {
    bool b;
    int8_t i8;
    uint8_t u8;
    int16_t i16;
    uint16_t u16; // represents UINT16, FLOAT16, BFLOAT16
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
    float f32;
    double f64;
  };
  using Transform = std::function<RawDatum(llvm::ArrayRef<char>, size_t)>;

  DisposableElementsImpl(mlir::ShapedType type, DTYPE::DataType dtype,
      const std::shared_ptr<llvm::MemoryBuffer> &buffer, llvm::ArrayRef<int64_t> strides,
      Transform transformFn = nullptr);
  ~DisposableElementsImpl();

  DisposableElements transpose(mlir::ShapedType type, DataType dtype, llvm::ArrayRef<int64_t> permutation) const;
  DisposableElements reshape(mlir::ShapedType type, DataType dtype) const;
  DisposableElements broadcast(mlir::ShapedType type, DataType dtype) const;
  DisposableElements transform(mlir::ShapedType type, DataType dtype, Transform transformFn) const;

  mlir::ShapedType getType() const { return type; }
  llvm::ArrayRef<int64_t> getShape() const { return type.getShape(); }
  int64_t getNumElements() const { return type.getNumElements(); }
  llvm::ArrayRef<int64_t> getStrides() const { return strides; }

  // NOTE: isSplat() may return false even if all elements are the same, namely if transform
  //       maps all underlying buffer entries to the same element 
  bool isSplat() const { return transformFn == nullptr && buffer->getBufferSize() == sizeOfDataType(dtype); }
  bool isContiguous() const {
    auto defaultStrides = calculateStrides(getShape());
    return strides == defaultStrides;
  }
  llvm::Optional<llvm::ArrayRef<char>> getRawData() const {
    if (transformFn) return llvm::None; // meaningless to look at underlying buffer if it requires transform
    llvm::StringRef str = buffer->getBuffer();
    return llvm::makeArrayRef(str.data(), str.size());
  }

private:
  mlir::ShapedType type; // ranked type of the corresponding DisposableElementsAttr
  llvm::SmallVector<int64_t, 4> strides; // empty/truncated strides mean broadcast or do we guarantee strides.size()==type.getRank()?
  DataType dtype;
  std::shared_ptr<llvm::MemoryBuffer> buffer;
  Transform transformFn; // is null if isSplat, can be null if dtype matches buffer contents
};

using DisposableElements = std::shared_ptr<DisposableElementsImpl>;

// Similar to the Adaptor class for an op, but with attribute operands like
// fold().
class OpFoldAdaptor {
public:
  OpFoldAdaptor() = default;
  // OpFoldAdaptor(OperationName opName, ArrayRef<Attribute> operands,
  //     mlir::DictionaryAttr attrs = nullptr, mlir::RegionRange regions = {});
  OpFoldAdaptor(mlir::Operation *op, llvm::ArrayRef<mlir::Attribute> operands);
private:
  llvm::Optional<mlir::OperationName> opName;
  llvm::SmallVector<mlir::Attribute, 2> operands;
  mlir::DictionaryAttr attributes;
  // No regions. Don't use OpFoldAdaptor for If, Loop, Scan.
};

// Represents the results of an operation with constant inputs,
// lazily evaluated.
class DisposableExpression {
public:
  DisposableExpression(DisposableElements result);
  DisposableElements getResult(size_t i);
  void force() { // should this be private?
    llvm::call_once(forceOnceFlag, [this]{ this->doForce(); });
  }
  void markReachable(size_t gcCycle); // marks all decendants reachable
  void dispose();
  bool isDisposed() const { return disposed; }
private:
  void doForce(); // populates results if not already forced
  bool disposed = false;
  llvm::once_flag forceOnceFlag;
  llvm::SmallVector<DisposableElements, 1> results; // only populated if forced
  llvm::Optional<OpFoldAdaptor> opFoldAdaptor; // ignored if forced
  size_t reachableMarking = 0;
};
using DisposableResultsHandle = DisposableExpression *;

}

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.hpp.inc"

namespace mlir {

inline ::onnx_mlir::DisposableElements DisposableElementsAttr::getElements() const {
  return getResultsHandle()->getResult(getResultIndex());
}

}