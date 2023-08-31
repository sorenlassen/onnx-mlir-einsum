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

template <typename OpType>
class ONNXElementwiseBinaryOpConstantFolder
    : public lazycst::OpConstantFolder<OpType> {
public:
  using FoldAdaptor = typename OpType::FoldAdaptor;
  virtual Attribute fold(OpType op, FoldAdaptor adaptor) const override {
    llvm_unreachable("TODO: implement this");
  }
};

template <typename OpType>
class ONNXElementwiseUnaryOpConstantFolder
    : public lazycst::OpConstantFolder<OpType> {
public:
  using FoldAdaptor = typename OpType::FoldAdaptor;
  virtual Attribute fold(OpType op, FoldAdaptor adaptor) const override {
    llvm_unreachable("TODO: implement this");
  }
};

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
