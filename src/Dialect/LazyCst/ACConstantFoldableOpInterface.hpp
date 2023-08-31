/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/TypeID.h"

namespace lazycst {

// Classifies ops as Associative, Commutative, and Constant-Foldable.
// To classify an Op, call
// Op::attachInterface<ACConstantFoldableOpInterface>(ctx).
//
// TODO: Move declaration to TableGen.
class ACConstantFoldableOpInterface;

namespace detail {

struct ACConstantFoldableOpInterfaceTraits {
  struct Concept {
    virtual ~Concept() = default;
  };
  template <typename ConcreteOp>
  struct Model : public Concept {
    using Interface = ACConstantFoldableOpInterface;
  };
  template <typename ConcreteOp>
  struct FallbackModel : public Concept {
    using Interface = ACConstantFoldableOpInterface;
  };
  template <typename ConcreteModel, typename ConcreteOp>
  struct ExternalModel : public FallbackModel<ConcreteModel> {};
};

template <typename ConcreteOp>
struct ACConstantFoldableOpInterfaceTrait;

} // namespace detail

class ACConstantFoldableOpInterface
    : public mlir::OpInterface<ACConstantFoldableOpInterface,
          detail::ACConstantFoldableOpInterfaceTraits> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ACConstantFoldableOpInterface);

  /// Inherit the base class constructor to support LLVM-style casting.
  using mlir::OpInterface<ACConstantFoldableOpInterface,
      detail::ACConstantFoldableOpInterfaceTraits>::OpInterface;

  template <typename ConcreteOp>
  struct Trait : public detail::ACConstantFoldableOpInterfaceTrait<ConcreteOp> {
  };
};

namespace detail {

template <typename ConcreteOp>
struct ACConstantFoldableOpInterfaceTrait
    : public mlir::OpInterface<ACConstantFoldableOpInterface,
          detail::ACConstantFoldableOpInterfaceTraits>::Trait<ConcreteOp> {};

} // namespace detail

} // namespace lazycst
