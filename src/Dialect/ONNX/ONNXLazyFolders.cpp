/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/ONNX/ONNXLazyFolders.hpp"
#include "src/Dialect/LazyCst/ACLazyFoldableOpInterface.hpp"
#include "src/Dialect/LazyCst/LazyCst.hpp"
#include "src/Dialect/LazyCst/LazyFoldableAnalysis.hpp"
#include "src/Dialect/LazyCst/LazyFolder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

template <typename OpType>
class ONNXBinaryOpLazyFolder : public lazycst::OpLazyFolder<OpType> {
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

class ONNXRangeOpLazyFolder : public lazycst::OpLazyFolder<ONNXRangeOp> {
public:
  virtual Attribute fold(ONNXRangeOp op, FoldAdaptor adaptor) const override {
    ElementsAttr start = cast<ElementsAttr>(adaptor.getStart());
    ElementsAttr delta = cast<ElementsAttr>(adaptor.getDelta());
    ShapedType type = cast<ShapedType>(op.getType());
    OnnxElementsAttrBuilder eb(op.getContext());
    return eb.range(type, getScalarNum(start), getScalarNum(delta));
  }
};

} // namespace

void populateONNXLazyFolders(
    MLIRContext *ctx, lazycst::LazyCstDialect *lazycstDialect, ONNXDialect *) {
  lazycstDialect->lazyFolders
      .insertOpLazyFolder<ONNXBinaryOpLazyFolder<ONNXAddOp>>()
      .insertOpLazyFolder<ONNXBinaryOpLazyFolder<ONNXMulOp>>()
      .insertOpLazyFolder<ONNXRangeOpLazyFolder>();

  ONNXAddOp::attachInterface<lazycst::ACLazyFoldableOpInterface>(*ctx);
  ONNXMulOp::attachInterface<lazycst::ACLazyFoldableOpInterface>(*ctx);
  ONNXXorOp::attachInterface<lazycst::ACLazyFoldableOpInterface>(*ctx);
  ONNXBitwiseXorOp::attachInterface<lazycst::ACLazyFoldableOpInterface>(*ctx);
}

} // namespace onnx_mlir
