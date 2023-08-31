/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/ONNX/ONNXConstantFolders.hpp"
#include "src/Dialect/LazyCst/ACConstantFoldableOpInterface.hpp"
#include "src/Dialect/LazyCst/ConstantFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/ConstantFolder.hpp"
#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

// HELPERS

// Extracts number from a scalar elements attribute.
WideNum getScalarNum(ElementsAttr elements) {
  Type elementType = elements.getElementType();
  if (isa<FloatType>(elementType)) {
    APFloat f = *elements.value_begin<APFloat>();
    return WideNum::fromAPFloat(f);
  } else if (auto itype = dyn_cast<IntegerType>(elementType)) {
    APInt i = *elements.value_begin<APInt>();
    return WideNum::fromAPInt(i, !itype.isUnsigned());
  } else {
    llvm_unreachable("Only integer and float types are supported");
  }
}

// Restricts specialization to non-bool types.
template <typename T>
using EnableNotBool = std::enable_if_t<!std::is_same_v<T, bool>>;

template <typename UnaryOpType, typename T, class Enable = void>
struct ElementwiseUnaryOpImpl {
  static T eval(T val) { llvm_unreachable("unsupported op or type"); }
};

template <typename T>
struct ElementwiseUnaryOpImpl<ONNXNegOp, T, EnableNotBool<T>> {
  static T eval(T val) { return -val; }
};

template <>
struct ElementwiseUnaryOpImpl<ONNXSqrtOp, double> {
  static double eval(double val) { return sqrt(val); }
};

template <typename T>
struct ElementwiseUnaryOpImpl<ONNXReluOp, T, EnableNotBool<T>> {
  static T eval(T val) { return (val < 0) ? 0 : val; }
};

template <typename UnaryOpType>
auto elementwiseUnaryOpFunction(Type elemType) {
  return getWideNumWrappedTemplateFunction<ElementwiseUnaryOpImpl, UnaryOpType>(
      elemType);
}

template <typename BinaryOpType, typename T, class Enable = void>
struct ElementwiseBinaryOpImpl {
  static T eval(T lhs, T rhs) { llvm_unreachable("unsupported op or type"); }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXAddOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs + rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXSubOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs - rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXMulOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs * rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXDivOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs / rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXMinOp, T> {
  static T eval(T lhs, T rhs) { return std::min<T>(lhs, rhs); }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXMaxOp, T> {
  static T eval(T lhs, T rhs) { return std::max<T>(lhs, rhs); }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXEqualOp, T> {
  static bool eval(T lhs, T rhs) { return lhs == rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXLessOp, T, EnableNotBool<T>> {
  static bool eval(T lhs, T rhs) { return lhs < rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXGreaterOp, T, EnableNotBool<T>> {
  static bool eval(T lhs, T rhs) { return lhs > rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXLessOrEqualOp, T, EnableNotBool<T>> {
  static bool eval(T lhs, T rhs) { return lhs <= rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXGreaterOrEqualOp, T, EnableNotBool<T>> {
  static bool eval(T lhs, T rhs) { return lhs >= rhs; }
};

template <typename T>
struct ElementwiseBinaryOpImpl<ONNXSumOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs + rhs; }
};

template <typename BinaryOpType>
constexpr auto elementwiseBinaryOpCombiner(Type elemType) {
  return getWideNumWrappedTemplateFunction<ElementwiseBinaryOpImpl,
      BinaryOpType>(elemType);
}

[[maybe_unused]] constexpr auto addCombiner(Type elemType) {
  return elementwiseBinaryOpCombiner<ONNXAddOp>(elemType);
}

[[maybe_unused]] constexpr auto subCombiner(Type elemType) {
  return elementwiseBinaryOpCombiner<ONNXSubOp>(elemType);
}

// CONSTANT FOLDERS

template <typename UnaryOpType>
class ONNXElementwiseUnaryOpConstantFolder
    : public lazycst::OpConstantFolder<UnaryOpType> {
public:
  using FoldAdaptor = typename UnaryOpType::FoldAdaptor;
  virtual Attribute fold(UnaryOpType op, FoldAdaptor adaptor) const override {
    ElementsAttr operand = cast<ElementsAttr>(adaptor.getOperands()[0]);
    Type elementType = operand.getElementType();
    OnnxElementsAttrBuilder eb(op.getContext());
    return eb.transform(operand, elementType,
        elementwiseUnaryOpFunction<UnaryOpType>(elementType));
  }
};

template <typename BinaryOpType>
class ONNXElementwiseBinaryOpConstantFolder
    : public lazycst::OpConstantFolder<BinaryOpType> {
public:
  using FoldAdaptor = typename BinaryOpType::FoldAdaptor;
  virtual Attribute fold(BinaryOpType op, FoldAdaptor adaptor) const override {
    ElementsAttr lhs = cast<ElementsAttr>(adaptor.getOperands()[0]);
    ElementsAttr rhs = cast<ElementsAttr>(adaptor.getOperands()[1]);
    ShapedType type = cast<ShapedType>(op.getType());
    OnnxElementsAttrBuilder eb(op.getContext());
    return eb.combine(lhs, rhs, type,
        elementwiseBinaryOpCombiner<BinaryOpType>(type.getElementType()));
  }
};

class ONNXRangeOpConstantFolder
    : public lazycst::OpConstantFolder<ONNXRangeOp> {
public:
  virtual Attribute fold(ONNXRangeOp op, FoldAdaptor adaptor) const override {
    ElementsAttr start = cast<ElementsAttr>(adaptor.getStart());
    ElementsAttr delta = cast<ElementsAttr>(adaptor.getDelta());
    ShapedType type = cast<ShapedType>(op.getType());
    OnnxElementsAttrBuilder eb(op.getContext());
    return eb.range(type, getScalarNum(start), getScalarNum(delta));
  }
};

// CONSTANT-FOLDABLE PROPAGATION PATTERNS

struct SubConstToNegPattern : public OpRewritePattern<ONNXSubOp> {
  using Base = OpRewritePattern<ONNXSubOp>;

  SubConstToNegPattern(
      lazycst::ConstantFoldableAnalysis &analysis, MLIRContext *ctx)
      : Base(ctx), analysis(analysis) {}

  LogicalResult matchAndRewrite(
      ONNXSubOp subOp, PatternRewriter &rewriter) const override {
    if (analysis.isConstantFoldableOp(subOp))
      return failure();
    Value rhs = subOp.getB();
    if (!analysis.isConstantFoldable(rhs))
      return failure();
    // onnx.Neg doesn't work on unsigned types. In principle we could negate
    // with onnx.Sub(dense<0>,_) but "negative" unsigned numbers are tricky
    // so we just bail out.
    if (getElementType(rhs.getType()).isUnsignedInteger())
      return failure();

    ONNXNegOp negOp =
        rewriter.create<ONNXNegOp>(rhs.getLoc(), rhs.getType(), rhs);
    analysis.insertConstantFoldableOp(negOp);
    ONNXAddOp addOp = rewriter.create<ONNXAddOp>(
        subOp.getLoc(), subOp.getType(), subOp.getA(), negOp);
    rewriter.replaceOp(subOp, addOp);
    return success();
  }

  lazycst::ConstantFoldableAnalysis &analysis;
};

} // namespace

void populateONNXConstantFolders(
    MLIRContext *ctx, lazycst::LazyCstDialect *lazycstDialect, ONNXDialect *) {
  lazycstDialect->constantFolders
      .insertOpConstantFolder<ONNXElementwiseUnaryOpConstantFolder<ONNXNegOp>>()
      .insertOpConstantFolder<
          ONNXElementwiseBinaryOpConstantFolder<ONNXAddOp>>()
      .insertOpConstantFolder<
          ONNXElementwiseBinaryOpConstantFolder<ONNXMulOp>>()
      .insertOpConstantFolder<ONNXRangeOpConstantFolder>();

  ONNXAddOp::attachInterface<lazycst::ACConstantFoldableOpInterface>(*ctx);
  ONNXMulOp::attachInterface<lazycst::ACConstantFoldableOpInterface>(*ctx);
  ONNXXorOp::attachInterface<lazycst::ACConstantFoldableOpInterface>(*ctx);
  ONNXBitwiseXorOp::attachInterface<lazycst::ACConstantFoldableOpInterface>(
      *ctx);

  lazycstDialect->constantFolders.addPatternsGetter(
      [](lazycst::ConstantFoldableAnalysis &analysis,
          mlir::RewritePatternSet &results) {
        results.add<SubConstToNegPattern>(analysis, results.getContext());
      });
}

} // namespace onnx_mlir
