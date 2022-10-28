/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXConstProp.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to constprop an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the constpropd operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"
#include "src/Support/DType.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

#include <math.h>

using namespace mlir;
using namespace onnx_mlir;

namespace {

//===----------------------------------------------------------------------===//
// Instructions to add a constant operation.
//===----------------------------------------------------------------------===//
// There is currently support for adding constant propagation for unary and
// binary arithmetic ops (binary ops support broadcast). To add an operation,
// you simply have to add a templated method on how to compute the result in
// terms of one or two inputs.
//
// The methods are:
//
// ElementWiseBinaryOpImpl and ElementWiseUnaryOpImpl
// and they need to be templated with an ONNX Operation (presuably).
//
// Then you need to add rules on how to transform the patterns; look into
// ConstProp.td for example.
//

const StringRef BUFFER_ID_ATTR = "buffer_id";

/// Buffers will be allocated to store intermediate constants during the const
/// propagation. The use of buffers is to avoid creating dense attributes which
/// are immortal by design in MLIR, leading to small memory footprint.
///
/// There are three helper functions to use when working with buffers:
/// 1) getArrayFromAttributeOrBuffer(PatternRewriter &rewriter, Operation *op)
///    - create a buffer from a dense attribute at the first time we reach the
///      const 'op' and add the buffer to the buffer pool, or
///    - get the buffer from the buffer pool if it was created.
/// 2) createConstantOpAndStoreBufferPtr(..., char *buffer)
///    - create a new ONNXConstantOp using the given buffer, and
///    - add the buffer to the buffer pool.
/// 3) allocateBufferFor(Value value, bool useMaxSize = false)
///    - create a new buffer whose size is obtained from the type of 'value'.
///
/// Note that:
///   - The buffers in the buffer pool will be automatically freed. Users don't
///     need to take care about that.
///   - If we create a buffer and do not put it on the buffer pool, please
///     make sure that it is correctly freed.
///
/// Buffer pool to store buffer pointers.
SmallVector<char *, 4> bufferPtrs;

template <typename S>
struct CastIntsOrFPs {
  template <typename DstDTy, typename... Args>
  struct Cast {
    using D = typename DstDTy::type;
    static void eval(ArrayRef<char> src, char *dst) {
      D *rs = reinterpret_cast<D *>(dst);
      ArrayRef<S> vs = castArrayRef<S>(src);
      std::transform(
          vs.begin(), vs.end(), rs, [](S v) { return static_cast<D>(v); });
    }
  };
};

ArrayRef<char> getDenseIntOrFPRawDataFromConstOp(
    ONNXConstantOp constOp, ShapedType type) {
  Attribute bufferIDAttr =
      constOp->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR);
  if (bufferIDAttr) {
    unsigned bufferId = bufferIDAttr.cast<IntegerAttr>().getUInt();
    int64_t maxSize = getMaxSizeInBytes(type);
    ArrayRef<char> src = llvm::makeArrayRef(bufferPtrs[bufferId], maxSize);
    int64_t size = getSizeInBytes(type);
    if (size == maxSize)
      return src;
    // TODO: redo all the following
    char *res = allocateBufferFor(type, /*useMaxSize=*/false);
    bufferPtrs.push_back(res);
    Type elementType = type.getElementType();
    if (elementType.isa<FloatType>()) {
      dispatchFP<CastIntsOrFPs<double>::template Cast, void>::eval(
          elementType, src, res);
    } else if (elementType.isa<IntegerType>()) {
      dispatchInt<CastIntsOrFPs<int64_t>::template Cast, void>::eval(
          elementType, src, res);
    } else {
      llvm_unreachable("Unknown data type");
    }
    return llvm::makeArrayRef(res, size);
  }
  ElementsAttr elements = constOp.valueAttr().cast<ElementsAttr>();
  return getDenseIntOrFPRawData(elements);
}

ArrayRef<char> getDenseIntOrFPRawDataFromConstValue(Value constValue) {
  ONNXConstantOp constOp = getONNXConstantOp(constValue);
  return getDenseIntOrFPRawDataFromConstOp(constOp, constValue.getType());
}

/// Get a data array from a given ONNXConstantOp. If data were stored in memory,
/// get from memory. Otherwise, get from the dense attribute.
char *getArrayFromAttributeOrBuffer(PatternRewriter &rewriter, Operation *op) {
  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  assert(constOp && "Not a constant operation");
  char *res = nullptr;

  Attribute bufferIDAttr = op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR);
  if (bufferIDAttr) {
    unsigned bufferId = bufferIDAttr.cast<IntegerAttr>().getUInt();
    res = bufferPtrs[bufferId];
  } else {
    ElementsAttr dataAttr = op->getAttrOfType<::mlir::Attribute>("value")
                                .cast<mlir::ElementsAttr>();
    res = createArrayFromDenseElementsAttr(dataAttr);
    bufferPtrs.emplace_back(res);
    unsigned bufferId = bufferPtrs.size() - 1;
    // Add an attribute to store the buffer id.
    op->setAttr(BUFFER_ID_ATTR,
        IntegerAttr::get(
            rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false),
            bufferId));
  }
  return res;
}

/// Get array with the exact data type for the final ONNXConstantOp.
void getArrayForFinalOutput(Operation *op, char *res) {
  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  assert(constOp && "Not a constant operation");

  Attribute bufferIDAttr = op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR);
  if (bufferIDAttr) {
    unsigned bufferId = bufferIDAttr.cast<IntegerAttr>().getUInt();
    char *resArr = bufferPtrs[bufferId];
    convertDoubleInt64ToExactType(constOp.getResult().getType(), resArr, res);
  } else {
    llvm_unreachable("Could not find the input buffer");
  }
}

/// A helper function to construct a RankedTensorType from a ShapedType.
ATTRIBUTE(unused) RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

/// A helper function to check whether a value is produced by a dense
/// ONNXConstantOp.
bool isFromDenseONNXConstantOp(Value result, bool trueONNXConstant = false) {
  Operation *op = result.getDefiningOp();

  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  // If the dense attribute is null, there must be buffer_id
  // attribute.
  if (!(op->getAttrOfType<::mlir::Attribute>("value"))) {
    if (trueONNXConstant)
      return false;
    if (!(op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR)))
      return false;
  }
  // The other attributes must be null.
  if (op->getAttrOfType<::mlir::Attribute>("sparse_value"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_float"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_floats"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_int"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_ints"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_string"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_strings"))
    return false;

  return true;
}

/// A helper function to check whether a variadic value is produced by dense
/// ONNXConstantOps.
bool isVariadicOperandFromDenseONNXConstantOp(ValueRange operands) {
  return llvm::all_of(
      operands, [](Value v) { return isFromDenseONNXConstantOp(v); });
}

/// A helper function to create an ONNXConstantOp for a given data array.
/// This ONNXConstantOp is only used internally.
ONNXConstantOp createConstantOpAndStoreBufferPtr(
    PatternRewriter &rewriter, Value replacingValue, char *vt) {
  Location loc = replacingValue.getLoc();
  // int64_t maxSizeInBytes = getMaxSizeInBytes(replacingValue.getType());

  ONNXConstantOp constOp = rewriter.create<ONNXConstantOp>(loc,
      replacingValue.getType(), Attribute(), Attribute(), FloatAttr(),
      ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());

  // Store the buffer pointer.
  unsigned bufferId = (unsigned)-1;
  for (unsigned i = 0; i < bufferPtrs.size(); ++i) {
    if (bufferPtrs[i] == vt) {
      bufferId = i;
      break;
    }
  }

  if (bufferId == (unsigned)-1) {
    bufferPtrs.emplace_back(vt);
    bufferId = bufferPtrs.size() - 1;
  }
  // Store the buffer id.
  // llvm::errs() << "BUFFER_ID_ATTR: " << replacingValue.getType() << "\n";
  // replacingValue.dump();
  // llvm::errs() << "\n";
  constOp.getOperation()->setAttr(BUFFER_ID_ATTR,
      IntegerAttr::get(
          rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false), bufferId));

  return constOp;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for binary in presence of broadcast.
//===----------------------------------------------------------------------===//

// Template to generate binary operation results. It takes as input the element
// type as well as the two element attributes for the operation, and return the
// result of the operation.

template <typename OP, typename U, class Enable = void>
struct ElementWiseBinaryOpImpl {
  static U impl(U lhs, U rhs) { llvm_unreachable("unknown operation"); }
};

template <typename U>
struct ElementWiseBinaryOpImpl<ONNXAddOp, U, notBool<U>> {
  static U impl(U lhs, U rhs) { return lhs + rhs; }
};

template <typename U>
struct ElementWiseBinaryOpImpl<ONNXSubOp, U, notBool<U>> {
  static U impl(U lhs, U rhs) { return lhs - rhs; }
};

template <typename U>
struct ElementWiseBinaryOpImpl<ONNXMulOp, U, notBool<U>> {
  static U impl(U lhs, U rhs) { return lhs * rhs; }
};

template <typename U>
struct ElementWiseBinaryOpImpl<ONNXDivOp, U, notBool<U>> {
  static U impl(U lhs, U rhs) { return lhs / rhs; }
};

std::vector<int64_t> unbroadcast(
    ArrayRef<int64_t> dstIndices, ArrayRef<int64_t> srcShape) {
  assert(dstIndices.size() >= srcShape.size());
  size_t offset = dstIndices.size() - srcShape.size();
  std::vector<int64_t> srcIndices;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    assert(srcShape[i] == 1 || dstIndices[i + offset] < srcShape[i]);
    srcIndices.push_back(srcShape[i] == 1 ? 0 : dstIndices[i + offset]);
  }
  return srcIndices;
}

template <typename OP>
struct ElementwiseBinary {
  template <typename DTy, typename... Ts>
  struct Compute {
    using S = typename DTy::type;
    using U = typename DTy::unpacked_type;
    static S fn(S x, S y) {
      return static_cast<S>(ElementWiseBinaryOpImpl<OP, U>::impl(
          static_cast<U>(x), static_cast<U>(y)));
    }
    static void eval(ArrayRef<int64_t> lhsShape, ArrayRef<char> lhs,
        ArrayRef<int64_t> rhsShape, ArrayRef<char> rhs,
        ArrayRef<int64_t> dstShape, MutableArrayRef<char> dst) {
      ArrayRef<S> xs = castArrayRef<S>(lhs);
      ArrayRef<S> ys = castArrayRef<S>(rhs);
      MutableArrayRef<S> zs = castMutableArrayRef<S>(dst);
      std::vector<int64_t> lhsStrides;
      bool lhsBroadcast = lhsShape != dstShape;
      if (lhsBroadcast)
        lhsStrides = getStrides(lhsShape);
      std::vector<int64_t> rhsStrides;
      bool rhsBroadcast = rhsShape != dstShape;
      if (rhsBroadcast)
        rhsStrides = getStrides(rhsShape);
      std::vector<int64_t> dstStrides;
      bool broadcast = lhsBroadcast || rhsBroadcast;
      if (broadcast)
        dstStrides = getStrides(dstShape);
      for (size_t k = 0; k < zs.size(); ++k) {
        size_t i = k;
        size_t j = k;
        if (broadcast) {
          std::vector<int64_t> dstIndices = getAccessIndex(i, dstStrides);
          if (lhsBroadcast) {
            std::vector<int64_t> lhsIndices = unbroadcast(dstIndices, lhsShape);
            i = getLinearAccessIndex(lhsIndices, lhsStrides);
          }
          if (rhsBroadcast) {
            std::vector<int64_t> rhsIndices = unbroadcast(dstIndices, rhsShape);
            j = getLinearAccessIndex(rhsIndices, rhsStrides);
          }
        }
        zs[k] = fn(xs[i], ys[j]);
      }
    }
  };
};

/// Do element-wise binary calculation of 'lhs' and 'rhs' values and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseBinaryOp>
Value ConstPropElementwiseBinary(PatternRewriter &rewriter,
    Value replacingValue, Value lhsValue, Value rhsValue) {
  ShapedType lhsType = lhsValue.getType().cast<ShapedType>();
  ShapedType rhsType = rhsValue.getType().cast<ShapedType>();
  ShapedType type = replacingValue.getType().cast<ShapedType>();
  size_t eltSizeInBytes = getEltSizeInBytes(type);
  Type elementType = type.getElementType();
  assert(
      lhsType.getElementType() == elementType && "lhs element type mismatch");
  assert(
      rhsType.getElementType() == elementType && "rhs element type mismatch");
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();

  ArrayRef<int64_t> splatShape = {};
  ArrayRef<char> lhs = getDenseIntOrFPRawDataFromConstValue(lhsValue);
  if (lhs.size() == eltSizeInBytes) {
    lhsShape = splatShape;
  }
  ArrayRef<char> rhs = getDenseIntOrFPRawDataFromConstValue(rhsValue);
  if (rhs.size() == eltSizeInBytes) {
    rhsShape = splatShape;
  }

  // TODO: make single element splat dst buffer if both lhs and rhs are splat

  ElementsAttr elements = makeDenseIntOrFPElementsAttrWithRawBuffer(
      type, [&](MutableArrayRef<char> dst) {
        dispatchFPOrInt<
            ElementwiseBinary<ElementwiseBinaryOp>::template Compute,
            void>::eval(elementType, lhsShape, lhs, rhsShape, rhs,
            type.getShape(), dst);
      });

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res = createONNXConstantOpWithDenseAttr(
      rewriter, replacingValue.getLoc(), elements);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
//// Code to perform constant propagation for unary operation.
//===----------------------------------------------------------------------===//

template <typename OP, typename U, class Enable = void>
struct ElementWiseUnaryOpImpl {
  static U impl(U val) { llvm_unreachable("unknown operation"); }
};

template <typename U>
struct ElementWiseUnaryOpImpl<ONNXSqrtOp, U, onlyFP<U>> {
  static U impl(U val) { return sqrt(val); }
};

template <typename U>
struct ElementWiseUnaryOpImpl<ONNXNegOp, U, notBool<U>> {
  static U impl(U val) { return -val; }
};

template <typename U>
struct ElementWiseUnaryOpImpl<ONNXReluOp, U, notBool<U>> {
  static U impl(U val) { return val < 0 ? 0 : val; }
};

template <typename OP>
struct ElementwiseUnary {
  template <typename DTy, typename... Ts>
  struct Compute {
    using S = typename DTy::type;
    using U = typename DTy::unpacked_type;
    static void eval(ArrayRef<char> src, MutableArrayRef<char> dst) {
      fillOrTransform(
          castArrayRef<S>(src), castMutableArrayRef<S>(dst), [](S v) {
            return static_cast<S>(
                ElementWiseUnaryOpImpl<OP, U>::impl(static_cast<U>(v)));
          });
    }
  };
};

/// Do element-wise unary calculation of 'input' value and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseUnaryOp>
Value ConstPropElementwiseUnary(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ShapedType srcType = constValue.getType().cast<ShapedType>();
  ShapedType replacingType = replacingValue.getType().cast<ShapedType>();
  assert(srcType.getNumElements() == replacingType.getNumElements() &&
         "types must have the equally many elements");

  Type elementType = replacingType.getElementType();

  ArrayRef<char> src = getDenseIntOrFPRawDataFromConstValue(constValue);

  // TODO: make single element splat dst buffer if src isSplat

  ElementsAttr elements = makeDenseIntOrFPElementsAttrWithRawBuffer(
      replacingType, [&](MutableArrayRef<char> dst) {
        dispatchFPOrInt<ElementwiseUnary<ElementwiseUnaryOp>::template Compute,
            void>::eval(elementType, src, dst);
      });

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res = createONNXConstantOpWithDenseAttr(
      rewriter, replacingValue.getLoc(), elements);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for transpose.
//===----------------------------------------------------------------------===//

Value ConstPropTranspose(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ArrayRef<int64_t> replacingShape =
      replacingValue.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> constShape =
      constValue.getType().cast<ShapedType>().getShape();
  Type elementType =
      replacingValue.getType().cast<ShapedType>().getElementType();

  // Get perm attribute.
  SmallVector<uint64_t, 4> perm;
  Attribute permAttr =
      replacingValue.getDefiningOp()->getAttrOfType<::mlir::Attribute>("perm");
  assert(permAttr && "permute attribute expected to be defined here");
  for (auto permVal : permAttr.cast<ArrayAttr>().getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());

  // Get the const value.
  char *constArray =
      getArrayFromAttributeOrBuffer(rewriter, constValue.getDefiningOp());

  // Do calculation.
  // Use maximum size (double or int64_t) to avoid the precision loss.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);
  ConstPropTransposeImpl(
      elementType, constArray, constShape, perm, replacingShape, resArray);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unsqueeze.
//===----------------------------------------------------------------------===//

Value ConstPropUnsqueeze(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  Operation *inputOp = input.getDefiningOp();

  char *resArray = getArrayFromAttributeOrBuffer(rewriter, inputOp);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for Squeeze.
//===----------------------------------------------------------------------===//

Value ConstPropSqueeze(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  Operation *inputOp = input.getDefiningOp();

  char *resArray = getArrayFromAttributeOrBuffer(rewriter, inputOp);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//

template <typename Op>
LogicalResult ConstPropSplitPatternCommon(Op splitOp, PatternRewriter &rewriter,
    llvm::Optional<ArrayAttr> splitAttr) {
  // Basic info.
  unsigned numOfResults = splitOp.getNumResults();
  Value input = splitOp.input();
  if (!isFromDenseONNXConstantOp(input))
    return failure();
  ShapedType inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  Type elementType = inputType.getElementType();

  // Split axis.
  uint64_t splitAxis = splitOp.axis();
  // Compute split offsets.
  SmallVector<int64_t, 4> splitOffsets;
  {
    if (!splitAttr.has_value())
      // If split attribute is not specified, split size is equally divided.
      assert(inputShape[splitAxis] % numOfResults == 0 &&
             "The dimension at the split axis is expected to be divisible by "
             "the number of results");
    int64_t offset = 0;
    for (unsigned int i = 0; i < numOfResults; ++i) {
      splitOffsets.emplace_back(offset);
      if (splitAttr.has_value())
        offset += splitAttr.value()[i].cast<IntegerAttr>().getInt();
      else
        offset += inputShape[splitAxis] / numOfResults;
    }
  }

  // Get the constant input value.
  char *inputArray =
      getArrayFromAttributeOrBuffer(rewriter, input.getDefiningOp());

  SmallVector<Value, 4> replacingValues;
  SmallVector<Type, 4> replacingTypes;
  for (unsigned int i = 0; i < numOfResults; ++i) {
    replacingValues.emplace_back(splitOp.getResults()[i]);
    replacingTypes.emplace_back(splitOp.getResults()[i].getType());
  }

  // Do splitting.
  std::vector<char *> resBuffers;
  ConstPropSplitImpl(elementType, inputArray, inputShape, splitAxis,
      splitOffsets, replacingTypes, resBuffers);

  // Construct result values.
  std::vector<Value> resValues;
  for (unsigned int i = 0; i < numOfResults; ++i) {
    ONNXConstantOp res = createConstantOpAndStoreBufferPtr(
        rewriter, replacingValues[i], resBuffers[i]);
    resValues.emplace_back(res.getResult());
  }

  rewriter.replaceOp(splitOp, resValues);
  return success();
}

class ConstPropSplitPattern : public OpRewritePattern<ONNXSplitOp> {
public:
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitOp splitOp, PatternRewriter &rewriter) const override {

    auto split = splitOp.split();
    auto builder = mlir::Builder(splitOp.getContext());

    llvm::Optional<ArrayAttr> optionalAttr;
    if (auto splitConstOp = getONNXConstantOp(split)) {
      // Checking value of split parameter.
      auto splitAttribute =
          createArrayAttrFromConstantOp(builder, splitConstOp);
      optionalAttr.emplace(splitAttribute);
    } else if (!split.getType().isa<NoneType>()) {
      llvm_unreachable("dynamic split not yet supported");
    }

    return ConstPropSplitPatternCommon(splitOp, rewriter, optionalAttr);
  }
};

class ConstPropSplitV11Pattern : public OpRewritePattern<ONNXSplitV11Op> {
public:
  using OpRewritePattern<ONNXSplitV11Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitV11Op splitOp, PatternRewriter &rewriter) const override {
    return ConstPropSplitPatternCommon(splitOp, rewriter, splitOp.split());
  }
};

// https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterND-13
/*
 * output = np.copy(data)
 * update_indices = indices.shape[:-1]
 * for idx in np.ndindex(update_indices):
 *     output[indices[idx]] = updates[idx]
 */
template <typename T>
LogicalResult ScatterNDImpl(
    PatternRewriter &rewriter, ONNXScatterNDOp scatterNdOp, char *raw_buffer) {

  char *data_value = getArrayFromAttributeOrBuffer(
      rewriter, scatterNdOp.data().getDefiningOp());
  char *indices_value = getArrayFromAttributeOrBuffer(
      rewriter, scatterNdOp.indices().getDefiningOp());
  char *updates_value = getArrayFromAttributeOrBuffer(
      rewriter, scatterNdOp.updates().getDefiningOp());

  auto data_shape = scatterNdOp.data().getType().cast<ShapedType>().getShape();
  auto indices_shape =
      scatterNdOp.indices().getType().cast<ShapedType>().getShape();
  auto updates_shape =
      scatterNdOp.updates().getType().cast<ShapedType>().getShape();

  // the output shape keep same with data, so fill with input data temporarily
  T *output_data = reinterpret_cast<T *>(data_value);
  int64_t *indices_data = reinterpret_cast<int64_t *>(indices_value);
  T *updates_data = reinterpret_cast<T *>(updates_value);

  int64_t n_slices = 1;
  int64_t slice_size = 1;

  int64_t outer_dims = indices_shape.size() - 1;
  int64_t indices_nd = indices_shape[outer_dims];
  int64_t updates_dims = updates_shape.size();

  for (int64_t i = 0; i < outer_dims; i++) {
    n_slices *= indices_shape[i];
  }

  for (int64_t i = outer_dims; i < updates_dims; i++) {
    slice_size *= updates_shape[i];
  }

  int64_t output_flat_size = ShapedType::getNumElements(data_shape);
  int64_t remain_flat_size = output_flat_size;
  std::vector<int64_t> dims_to_count(indices_nd, 0);

  for (int64_t i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / data_shape[i];
    remain_flat_size = dims_to_count[i];
  }

  for (int64_t i = 0; i < n_slices; ++i) {
    int64_t to_pos = 0;
    for (int64_t j = 0; j < indices_nd; ++j) {
      int64_t idx = indices_data[i * indices_nd + j];
      // assert(0 <= idx && idx < data_shape[j]);
      to_pos += idx * dims_to_count[j];
    }
    for (int64_t j = 0; j < slice_size; j++) {
      output_data[to_pos + j] = updates_data[i * slice_size + j];
    }
  }

  std::memcpy(raw_buffer, data_value, output_flat_size * 8);
  return success();
}

class ConstPropScatterNDPattern : public OpRewritePattern<ONNXScatterNDOp> {
public:
  using OpRewritePattern<ONNXScatterNDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXScatterNDOp scatterNdOp, PatternRewriter &rewriter) const override {
    // Match
    if (!scatterNdOp.getResult()
             .getType()
             .template dyn_cast_or_null<RankedTensorType>())
      return failure();

    if (!isFromDenseONNXConstantOp(scatterNdOp.data()))
      return failure();

    if (!isFromDenseONNXConstantOp(scatterNdOp.indices()))
      return failure();

    if (!isFromDenseONNXConstantOp(scatterNdOp.updates()))
      return failure();

    char *result_raw_data =
        allocateBufferFor(scatterNdOp.data().getType(), /*useMaxSize=*/true);

    mlir::ShapedType shaped_type =
        scatterNdOp.data().getType().cast<ShapedType>();

    if (shaped_type.getElementType().isa<FloatType>()) {
      if (mlir::failed(
              ScatterNDImpl<double>(rewriter, scatterNdOp, result_raw_data)))
        return failure();
    } else if (shaped_type.getElementType().isa<IntegerType>()) {
      if (mlir::failed(
              ScatterNDImpl<int64_t>(rewriter, scatterNdOp, result_raw_data)))
        return failure();
    } else {
      llvm_unreachable("type not yet supported");
    }

    // Construct result values.
    ONNXConstantOp gen_const_op = createConstantOpAndStoreBufferPtr(
        rewriter, scatterNdOp.data(), result_raw_data);

    SmallVector<Value, 1> op_repl_values(1, gen_const_op.getResult());
    rewriter.replaceOp(scatterNdOp, op_repl_values);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for CastOp.
//===----------------------------------------------------------------------===//

template <typename SrcDTy, typename... Args>
struct SrcDstCast {
  using S = typename SrcDTy::type;

  template <typename DstDTy, typename... InnerArgs>
  struct DstCast {
    using D = typename DstDTy::type;
    static void eval(ArrayRef<S> src, MutableArrayRef<char> dst) {
      fillOrTransform(src, castMutableArrayRef<D>(dst),
          [](S v) { return static_cast<D>(v); });
    }
  };

  static void eval(
      Type dstType, ArrayRef<char> src, MutableArrayRef<char> dst) {
    dispatchFPOrInt<DstCast, void>::eval(dstType, castArrayRef<S>(src), dst);
  }
};

Value ConstPropCast(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ShapedType srcType = constValue.getType().cast<ShapedType>();
  ShapedType dstType = replacingValue.getType().cast<ShapedType>();
  assert(srcType.getNumElements() == dstType.getNumElements() &&
         "types must have the equally many elements");

  Type srcElemType = srcType.getElementType();
  Type dstElemType = dstType.getElementType();

  ArrayRef<char> src = getDenseIntOrFPRawDataFromConstValue(constValue);

  // TODO: make single element splat dst buffer if src isSplat

  ElementsAttr elements = makeDenseIntOrFPElementsAttrWithRawBuffer(
      dstType, [&](MutableArrayRef<char> dst) {
        dispatchFPOrInt<SrcDstCast, void>::eval(
            srcElemType, dstElemType, src, dst);
      });

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res = createONNXConstantOpWithDenseAttr(
      rewriter, replacingValue.getLoc(), elements);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for SliceOp.
//===----------------------------------------------------------------------===//

Value ConstPropSlice(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  Operation *op = replacingValue.getDefiningOp();
  ONNXSliceOp sliceOp = cast<ONNXSliceOp>(op);

  ArrayRef<int64_t> inputShape = getShape(constValue.getType());
  std::vector<int64_t> inputStrides = getStrides(inputShape);
  ArrayRef<int64_t> outputShape = getShape(replacingValue.getType());
  std::vector<int64_t> outputStrides = getStrides(outputShape);

  // Get the const value using the maximum precision e.g. double, int64_t.
  char *constArray =
      getArrayFromAttributeOrBuffer(rewriter, constValue.getDefiningOp());

  // Create the result buffer using the maximum precision e.g. double, int64_t.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);

  // Get starts, ends, axes and steps via ShapeHelper.
  ONNXSliceOpShapeHelper shapeHelper(&sliceOp);
  ONNXSliceOpAdaptor operandAdaptor(sliceOp);
  if (failed(shapeHelper.computeShape(operandAdaptor))) {
    sliceOp.emitError("Failed to scan " + ONNXSliceOp::getOperationName() +
                      " parameters successfully");
    return nullptr;
  }

  // Iterate over the output index space.
  for (int64_t i = 0; i < ShapedType::getNumElements(outputShape); ++i) {
    // Input index: "ii * step + start" for all dim.
    // Output index: "ii" for all dims.
    // where `ii` is a tensor index.
    std::vector<int64_t> outputIndices = getAccessIndex(i, outputStrides);
    SmallVector<int64_t, 4> inputIndices;
    for (unsigned k = 0; k < outputIndices.size(); ++k) {
      int64_t ii = outputIndices[k];
      inputIndices.emplace_back(ii * shapeHelper.steps[k].getLiteral() +
                                shapeHelper.starts[k].getLiteral());
    }
    int64_t inputOffset = getLinearAccessIndex(inputIndices, inputStrides);
    int64_t typeSize = 8; // both double and int64_t have size of 8 bytes.
    memcpy(
        resArray + i * typeSize, constArray + inputOffset * typeSize, typeSize);
  }

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ConcatOp.
//===----------------------------------------------------------------------===//

Value ConstPropConcat(PatternRewriter &rewriter, Value replacingValue,
    ValueRange operands, IntegerAttr axisAttr) {
  // Get the const values using the maximum precision e.g. double, int64_t.
  SmallVector<char *, 4> inputArrays;
  for (uint64_t i = 0; i < operands.size(); ++i) {
    char *array =
        getArrayFromAttributeOrBuffer(rewriter, operands[i].getDefiningOp());
    inputArrays.emplace_back(array);
  }
  // Create the result buffer using the maximum precision e.g. double, int64_t.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);

  ArrayRef<int64_t> outputShape = getShape(replacingValue.getType());
  std::vector<int64_t> outputStrides = getStrides(outputShape);
  int64_t axis = axisAttr.getValue().getSExtValue();
  if (axis < 0)
    axis += outputShape.size();

  // If concatenation is on the outermost dimension, do memcpy for better
  // performance. Otherwise, copy elements one-by-one.
  if (axis == 0) {
    int64_t offset = 0;
    for (uint64_t i = 0; i < operands.size(); ++i) {
      int64_t sizeInBytes = getMaxSizeInBytes(operands[i].getType());
      memcpy(resArray + offset, inputArrays[i], sizeInBytes);
      offset += sizeInBytes;
    }
  } else {
    int64_t dimAtAxis = 0;
    for (uint64_t i = 0; i < operands.size(); ++i) {
      ArrayRef<int64_t> inputShape = getShape(operands[i].getType());
      std::vector<int64_t> inputStrides = getStrides(inputShape);
      for (int64_t k = 0; k < ShapedType::getNumElements(inputShape); ++k) {
        std::vector<int64_t> inputIndices = getAccessIndex(k, inputStrides);
        std::vector<int64_t> outputIndices(inputIndices);
        outputIndices[axis] += dimAtAxis;
        int64_t outputOffset =
            getLinearAccessIndex(outputIndices, outputStrides);
        int64_t typeSize = 8; // both double and int64_t have size of 8 bytes.
        memcpy(resArray + outputOffset * typeSize,
            inputArrays[i] + k * typeSize, typeSize);
      }
      dimAtAxis += inputShape[axis];
    }
  }

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ExpandOp.
//===----------------------------------------------------------------------===//

Value ConstPropExpand(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  // Get the const value using the maximum precision e.g. double, int64_t.
  char *inputArray =
      getArrayFromAttributeOrBuffer(rewriter, constValue.getDefiningOp());
  // Create the result buffer using the maximum precision e.g. double, int64_t.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);

  ArrayRef<int64_t> inputShape = getShape(constValue.getType());
  std::vector<int64_t> inputStrides = getStrides(inputShape);
  ArrayRef<int64_t> outputShape = getShape(replacingValue.getType());
  std::vector<int64_t> outputStrides = getStrides(outputShape);
  int64_t inputRank = inputShape.size();
  int64_t outputRank = outputShape.size();

  for (int64_t i = 0; i < ShapedType::getNumElements(outputShape); ++i) {
    // Compute indices to access the output.
    std::vector<int64_t> outputIndices = getAccessIndex(i, outputStrides);
    // Compute indices to access the input.
    SmallVector<int64_t, 4> inputIndices;
    if (inputRank == 0) {
      inputIndices.emplace_back(0);
    } else {
      for (int inputAxis = 0; inputAxis < inputRank; ++inputAxis) {
        if (inputShape[inputAxis] == 1) {
          // broadcast
          inputIndices.emplace_back(0);
        } else {
          int outputIndex = (outputRank - inputRank) + inputAxis;
          inputIndices.emplace_back(outputIndices[outputIndex]);
        }
      }
    }

    // Calculate the final result.
    int64_t inputOffset = getLinearAccessIndex(inputIndices, inputStrides);
    int64_t outputOffset = getLinearAccessIndex(outputIndices, outputStrides);
    int64_t typeSize = 8; // both double and int64_t have size of 8 bytes.
    memcpy(resArray + outputOffset * typeSize,
        inputArray + inputOffset * typeSize, typeSize);
  }

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for GatherOp.
//===----------------------------------------------------------------------===//

Value ConstPropGather(PatternRewriter &rewriter, Value replacingValue,
    Value inputValue, Value indicesValue) {
  Operation *op = replacingValue.getDefiningOp();
  ONNXGatherOp gatherOp = cast<ONNXGatherOp>(op);

  ArrayRef<int64_t> inputShape = getShape(inputValue.getType());
  ArrayRef<int64_t> indicesShape = getShape(indicesValue.getType());
  ArrayRef<int64_t> outputShape = getShape(replacingValue.getType());
  std::vector<int64_t> inputStrides = getStrides(inputShape);
  std::vector<int64_t> indicesStrides = getStrides(indicesShape);
  std::vector<int64_t> outputStrides = getStrides(outputShape);
  int64_t inputRank = inputShape.size();
  int64_t indicesRank = indicesShape.size();

  int64_t axis = gatherOp.axis();
  if (axis < 0)
    axis += inputRank;
  int64_t axisDim = inputShape[axis];

  // Get the input value using the maximum precision e.g. double, int64_t.
  char *inputArray =
      getArrayFromAttributeOrBuffer(rewriter, inputValue.getDefiningOp());

  // Get the indices value using the maximum precision. Index is integer.
  int64_t *indicesArray = (int64_t *)getArrayFromAttributeOrBuffer(
      rewriter, indicesValue.getDefiningOp());

  // Create the result buffer using the maximum precision e.g. double, int64_t.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);

  // Iterate over the output index space.
  for (int64_t ii = 0; ii < ShapedType::getNumElements(outputShape); ++ii) {
    std::vector<int64_t> outputIndices = getAccessIndex(ii, outputStrides);
    SmallVector<int64_t, 4> inputIndices, indicesIndices;
    // Compute tensor access indices for indices: indices[jj].
    for (int j = 0; j < indicesRank; ++j)
      indicesIndices.emplace_back(outputIndices[axis + j]);
    int64_t indicesOffset =
        getLinearAccessIndex(indicesIndices, indicesStrides);
    // Get indices.
    int64_t axisIndex = *(indicesArray + indicesOffset);
    if (axisIndex < 0)
      axisIndex += axisDim;

    // Compute tensor access indices for input: input[ii + (indices[jj],) + kk]
    // First add indices ii
    for (int i = 0; i < axis; ++i)
      inputIndices.emplace_back(outputIndices[i]);
    // Then add indices[jj] at axis.
    inputIndices.emplace_back(axisIndex);
    // Then add kk.
    for (int k = axis + 1; k < inputRank; ++k)
      inputIndices.emplace_back(outputIndices[indicesRank - 1 + k]);

    // Copy values.
    int64_t inputOffset = getLinearAccessIndex(inputIndices, inputStrides);
    int64_t typeSize = 8; // both double and int64_t have size of 8 bytes.
    memcpy(resArray + ii * typeSize, inputArray + inputOffset * typeSize,
        typeSize);
  }

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
}

//===----------------------------------------------------------------------===//
// Pattern definition.
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/ONNXConstProp.inc"

//===----------------------------------------------------------------------===//
// Code to manage the pass.
//===----------------------------------------------------------------------===//

struct ConstPropONNXToONNXPass
    : public PassWrapper<ConstPropONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstPropONNXToONNXPass)

  StringRef getArgument() const override { return "constprop-onnx"; }

  StringRef getDescription() const override {
    return "ConstProp ONNX operations into composition of "
           "other ONNX operations.";
  }

  void runOnOperation() final;
};
} // end anonymous namespace.

void ConstPropONNXToONNXPass::runOnOperation() {
  auto function = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.insert<ConstPropSplitPattern>(&getContext());
  patterns.insert<ConstPropSplitV11Pattern>(&getContext());
  patterns.insert<ConstPropScatterNDPattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
    signalPassFailure();

  // Create DenseElementsAttr and clean up helper attributes.
  function.walk([&](ONNXConstantOp constOp) {
    Operation *op = constOp.getOperation();
    if (op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR)) {
      ShapedType type = constOp.getResult().getType().cast<ShapedType>();
      char *arr = allocateBufferFor(type, /*useMaxSize=*/false);
      getArrayForFinalOutput(op, arr);
      DenseElementsAttr denseAttr =
          createDenseElementsAttrFromRawBuffer(type, arr);
      op->setAttr("value", denseAttr);
      op->removeAttr(BUFFER_ID_ATTR);
      free(arr);
    } else if (auto elements =
                   constOp.valueAttr().dyn_cast<DenseResourceElementsAttr>()) {
      auto *r = elements.getRawHandle().getResource();
      ArrayRef<char> a = r->getBlob()->getData();
      constOp.valueAttr(
          DenseElementsAttr::getFromRawBuffer(elements.getType(), a));
      r->setBlob({}); // Free blob.
    }
  });

  // Remove temporary buffers.
  for (char *ptr : bufferPtrs) {
    free(ptr);
  }
  bufferPtrs.clear();

} // end anonymous namespace

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
