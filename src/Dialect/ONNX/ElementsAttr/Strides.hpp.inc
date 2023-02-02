//===-------------------------- Strides.hpp.inc ---------------------------===//
//
//===----------------------------------------------------------------------===//
// Strides template implementations
//===----------------------------------------------------------------------===//

#if 1 // defined(_MSC_VER)

// MSVC cannot figure out the parameter packs in the variadic
// implementations of traverseStrides and mapStrides, so here we
// implement them manually for unary and binary cases.

template <typename Iter, size_t N>
struct StridedIterator
    : public llvm::iterator_facade_base<StridedIterator<Iter, Arg>, std::forward_iterator_tag, std::pair<Iter, std::array<size_t, N>>> {
  const llvm::ArrayRef<int64_t> shape;
  const std::array<llvm::ArrayRef<int64_t>, N> strides;
  value_type locations;
  llvm::SmallVector<uint64_t, 6> idx;

public:
  StridedIterator(llvm::ArrayRef<int64_t> shape, Iter iter, std::array<llvm::ArrayRef<int64_t>> strides)
  : shape(shape), strides(strides), tuple(iter, {}), idx(shape.size(), 0) {}

  StridedIterator(const StridedIterator &) = default;

  StridedIterator &operator=(const StridedIterator &) = default;

  bool operator==(const StridedIterator &other) const {
    return locations.first == other.locations.first;
  }

  const value_type &operator*() const { return tuple; }

  StridedIterator &operator++() {
    ++(locations.first);
#if 1
    for (auto axis = idx.size();;) {
      if (axis == 0)
        break;
      --axis;
      int64_t dim = shape[axis];
      for (unsigned i = 0; i < N; ++i)
        locations.second[i] += strides[i][axis];
      if (++(idx[axis]) < dim)
        break;
      for (unsigned i = 0; i < N; ++i)
        locations.second[i] -= dim * strides[i][axis];
      idx[axis] = 0;
    }
    return *this;
#else
    for (int a = static_cast<int>(axes.size()) - 1; a >= 0; --a) {
      int64_t dim = shape[a];
      int64_t stride = strides[a];
      ptr += stride;
      if (++(axes[a]) < dim) {
        return *this;
      }
      ptr -= dim * stride;
      axes[a] = 0;
    }
    return *this;
#endif
  }
};

template <typename Iterator, typename Arg0, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Arg0> src0, Action &&act) {
#if 1
  Iterator end = begin + mlir::ShapedType::getNumElements(shape);
  for (StridedIterator<Iterator, 1> it(shape, begin, {src0.strides}), stop({}, end, {{}}); it != stop; ++it) {
    auto locations = *it;
    act(it.first, it.second[0]);
  }
  return end;
#elif 1 // about 5% slower
  llvm::ArrayRef<int64_t> strides = src0.strides;
  llvm::SmallVector<int64_t, 6> axes(shape.size(), 0);
  Iterator it = begin;
  for (const Arg0 *ptr = src0.begin(); ptr != nullptr; ++it) {
    act(it, ptr);
    for (unsigned a = axes.size(); ; ) {
      if (a == 0) {
        ptr = nullptr;
        break;
      }
      --a;
      int64_t dim = shape[a];
      int64_t stride = strides[a];
      ptr += stride;
      if (++(axes[a]) < dim) {
        break;
      }
      ptr -= dim * stride;
      axes[a] = 0;
    }
  }
  return it;
#else
  auto traverse = [=](const auto &recurse, unsigned axis, Iterator it,
                      const Arg0 *elm0) -> Iterator {
    if (axis == shape.size()) {
      act(it, elm0);
      it++;
    } else {
      for (int64_t i = 0; i < shape[axis]; ++i) {
        it = recurse(recurse, axis + 1, it, elm0);
        elm0 += src0.strides[axis];
      }
    }
    return it;
  };
  return traverse(traverse, /*axis=*/0, begin, src0.begin());
#endif
}

template <typename Iterator, typename Arg0, typename Arg1, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Arg0> src0, StridedArrayRef<Arg1> src1, Action &&act) {
  auto traverse = [=](const auto &recurse, unsigned axis, Iterator it,
                      const Arg0 *elm0, const Arg1 *elm1) -> Iterator {
    if (axis == shape.size()) {
      act(it, elm0, elm1);
      it++;
    } else {
      for (int64_t i = 0; i < shape[axis]; ++i) {
        it = recurse(recurse, axis + 1, it, elm0, elm1);
        elm0 += src0.strides[axis];
        elm1 += src1.strides[axis];
      }
    }
    return it;
  };
  return traverse(traverse, /*axis=*/0, begin, src0.begin(), src1.begin());
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

#else

template <typename Iterator, typename... Args, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Args>... src, Action &&act) {
  // Iterator it = begin;
  // for (StridedIterator<Args...> srcs(shape, src...); srcs != nullptr; ++srcs, ++it)
  //   std::apply(act, std::tuple_cat(std::make_tuple(it), *srcs));
  // return it;
#if 0
  llvm::SmallVector<int64_t, 6> axes(shape.size(), 0);
#else
  auto traverse = [=](const auto &recurse, unsigned axis, Iterator it,
                      const Args *... elms) -> Iterator {
    if (axis == shape.size()) {
      act(it, elms...);
      it++;
    } else {
      for (int64_t i = 0; i < shape[axis]; ++i) {
        it = recurse(recurse, axis + 1, it, elms...);
        ((elms += src.strides[axis]), ...);
      }
    }
    return it;
  };
  return traverse(traverse, /*axis=*/0, begin, src.begin()...);
#endif
}

template <typename Res, typename... Args, typename Action>
void mapStrides(llvm::ArrayRef<int64_t> shape, llvm::MutableArrayRef<Res> dst,
    StridedArrayRef<Args>... src, Action &&act) {
  Res *end = traverseStrides<Res *, Args...>(shape, dst.begin(), src...,
      [&act](Res *res, const Args *... args) { *res = act(*args...); });
  assert(end == dst.end() && "traverses every dst element");
}

#endif