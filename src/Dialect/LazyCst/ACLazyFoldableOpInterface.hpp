/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/TypeID.h"

namespace lazycst {

// Classifies ops as Associative, Commutative, and Lazy-Foldable.
// To classify an Op, call Op::attachInterface<ACLazyFoldableOpInterface>(ctx).
//
// TODO: Move declaration to TableGen.
class ACLazyFoldableOpInterface;

namespace detail {

struct ACLazyFoldableOpInterfaceTraits {
  struct Concept {
    virtual ~Concept() = default;
  };
  template <typename ConcreteOp>
  struct Model : public Concept {
    using Interface = ACLazyFoldableOpInterface;
  };
  template <typename ConcreteOp>
  struct FallbackModel : public Concept {
    using Interface = ACLazyFoldableOpInterface;
  };
  template <typename ConcreteModel, typename ConcreteOp>
  struct ExternalModel : public FallbackModel<ConcreteModel> {};
};

template <typename ConcreteOp>
struct ACLazyFoldableOpInterfaceTrait;

} // namespace detail

class ACLazyFoldableOpInterface
    : public mlir::OpInterface<ACLazyFoldableOpInterface,
          detail::ACLazyFoldableOpInterfaceTraits> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ACLazyFoldableOpInterface);

  /// Inherit the base class constructor to support LLVM-style casting.
  using mlir::OpInterface<ACLazyFoldableOpInterface,
      detail::ACLazyFoldableOpInterfaceTraits>::OpInterface;

  template <typename ConcreteOp>
  struct Trait : public detail::ACLazyFoldableOpInterfaceTrait<ConcreteOp> {};
};

namespace detail {

template <typename ConcreteOp>
struct ACLazyFoldableOpInterfaceTrait
    : public mlir::OpInterface<ACLazyFoldableOpInterface,
          detail::ACLazyFoldableOpInterfaceTraits>::Trait<ConcreteOp> {};

} // namespace detail

} // namespace lazycst
