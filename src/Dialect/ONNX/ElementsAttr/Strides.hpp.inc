//===-------------------------- Strides.hpp.inc ---------------------------===//
//
//===----------------------------------------------------------------------===//
// Strides template implementations
//===----------------------------------------------------------------------===//

template <typename Iter, size_t N>
struct StridedIterator
    : public llvm::iterator_facade_base<StridedIterator<Iter, N>,
          std::forward_iterator_tag, std::pair<Iter, std::array<size_t, N>>> {
  using value_type = std::pair<Iter, std::array<size_t, N>>;

  const llvm::ArrayRef<int64_t> shape;
  const std::array<llvm::ArrayRef<int64_t>, N> strides;
  value_type locations;
  llvm::SmallVector<uint64_t, 6> idx;

public:
  StridedIterator(llvm::ArrayRef<int64_t> shape,
      std::array<llvm::ArrayRef<int64_t>, N> strides, Iter iter)
      : shape(shape), strides(strides), locations(iter, {}),
        idx(shape.size(), 0) {}

  StridedIterator(const StridedIterator &) = default;

  StridedIterator &operator=(const StridedIterator &) = default;

  bool operator==(const StridedIterator &other) const {
    return locations.first == other.locations.first;
  }

  const value_type &operator*() const { return locations; }

  const value_type *operator->() const { return &locations; }

  StridedIterator &operator++() {
    ++(locations.first);
    for (auto axis = idx.size();;) {
      if (axis == 0)
        break;
      --axis;
      uint64_t dim = shape[axis];
      for (unsigned i = 0; i < N; ++i)
        locations.second[i] += strides[i][axis];
      if (++(idx[axis]) < dim)
        break;
      for (unsigned i = 0; i < N; ++i)
        locations.second[i] -= dim * strides[i][axis];
      idx[axis] = 0;
    }
    return *this;
  }
};

template <typename Iterator, typename Arg0, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Arg0> src0, Action &&act) {
  Iterator end = begin + mlir::ShapedType::getNumElements(shape);
  for (StridedIterator<Iterator, 1> it(shape, {src0.strides}, begin);
       it->first != end; ++it) {
    act(it->first, src0.begin() + it->second[0]);
  }
  return end;
}

template <typename Iterator, typename Arg0, typename Arg1, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act) {
  Iterator end = begin + mlir::ShapedType::getNumElements(shape);
  for (StridedIterator<Iterator, 2> it(
           shape, {src0.strides, src1.strides}, begin);
       it->first != end; ++it) {
    act(it->first, src0.begin() + it->second[0], src1.begin() + it->second[1]);
  }
  return end;
}

template <typename Res, typename Arg0, typename Action>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Arg0> src0, Action &&act) {
  for (StridedIterator<Res *, 1> it(shape, {src0.strides}, dst.begin());
       it->first != dst.end(); ++it) {
    *(it->first) = act(src0.begin()[it->second[0]]);
  }
}

template <typename Res, typename Arg0, typename Arg1, typename Action>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act) {
  // The implementation below effectively inlines this iterator implementation
  // (manually inlined below to get 1.5x speedup):
  //
  //   for (StridesIterator<2> it(shape, {src0.strides, src1.strides}),
  //          end(shape); it != end; ++it)
  //     dst[it->flattenedIndex] = act(src0[it->pos[0]], src1[it->pos[1]]);
  //
  size_t pos0 = 0, pos1 = 0;
  llvm::SmallVector<uint64_t, 6> index(shape.size(), 0);
  for (auto d = dst.begin(); d != dst.end(); ++d) {
    *d = act(src0[pos0], src1[pos1]);
    for (auto axis = shape.size();;) {
      if (axis == 0) {
        assert(d + 1 == dst.end() && "index resets on last iteration");
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
}
