//===-------------------------- Strides.hpp.inc ---------------------------===//
//
//===----------------------------------------------------------------------===//
// Strides template implementations
//===----------------------------------------------------------------------===//

template <typename Iterator, typename Arg0, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Arg0> src0, Action &&act) {
  Iterator dst = begin;
  for (StridesIterator<1> it(shape, {src0.strides}), end(shape); it != end;
       ++it, ++dst) {
    act(dst, src0.begin() + it->pos[0]);
  }
  return dst;
}

template <typename Iterator, typename Arg0, typename Arg1, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act) {
  // The implementation below effectively inlines this iterator implementation
  // (manually inlined below to get 1.5x speedup):
  //
  //   Iterator dst = begin;
  //   for (StridesIterator<2> it(shape, {src0.strides, src1.strides}),
  //          end(shape); it != end; ++it, ++dst) {
  //     act(dst, src0.begin() + it->pos[0], src1.begin() + it->pos[1]);
  //   }
  //   return dst;
  //
  size_t pos0 = 0, pos1 = 0;
  llvm::SmallVector<uint64_t, 6> index(shape.size(), 0);
  auto end = begin + mlir::ShapedType::getNumElements(shape);
  for (auto d = begin; d != end; ++d) {
    act(d, src0.begin() + pos0, src1.begin() + pos1);
    for (auto axis = shape.size();;) {
      if (axis == 0) {
        assert(d + 1 == end && "index resets on last iteration");
        break;
      }
      --axis;
      uint64_t dim = shape[axis];
      pos0 += src0.strides[axis];
      pos1 += src1.strides[axis];
      if (++(index[axis]) < dim)
        break;
      pos0 -= dim * src0.strides[axis];
      pos1 -= dim * src1.strides[axis];
      index[axis] = 0;
    }
  }
  return end;
}

template <typename Res, typename Arg0, typename Action>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Arg0> src0, Action &&act) {
  Res *end = traverseStrides<Res *, Arg0>(shape, dst.begin(), src0,
      [&act](Res *res, const Arg0 *arg0) { *res = act(*arg0); });
  assert(end == dst.end() && "traverses every dst element");
}

template <typename Res, typename Arg0, typename Arg1, typename Action>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act) {
  Res *end = traverseStrides<Res *, Arg0, Arg1>(shape, dst.begin(), src0, src1,
      [&act](Res *res, const Arg0 *arg0, const Arg1 *arg1) {
        *res = act(*arg0, *arg1);
      });
  assert(end == dst.end() && "traverses every dst element");
}
