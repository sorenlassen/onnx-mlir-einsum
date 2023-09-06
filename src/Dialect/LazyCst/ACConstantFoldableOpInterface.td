// SPDX-License-Identifier: Apache-2.0

#ifndef AC_CONSTANT_FOLDABLE_OP_INTERFACE
#define AC_CONSTANT_FOLDABLE_OP_INTERFACE

include "mlir/IR/OpBase.td"


def ACConstantFoldableOpInterface: OpInterface<"ACConstantFoldableOpInterface"> {
  let cppNamespace = "lazycst";
  let description = [{
    Classifies ops as Associative, Commutative, and Constant-Foldable.
    To classify an op, e.g. ONNXAddOp, call
    ONNXAddOp::attachInterface<ACConstantFoldableOpInterface>(ctx).
  }];
}

#endif // AC_CONSTANT_FOLDABLE_OP_INTERFACE
