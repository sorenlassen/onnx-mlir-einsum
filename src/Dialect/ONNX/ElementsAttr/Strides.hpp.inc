//===-------------------------- Strides.hpp.inc ---------------------------===//
//
//===----------------------------------------------------------------------===//
// Strides template implementations
//===----------------------------------------------------------------------===//

#if 1 // defined(_MSC_VER)

// MSVC cannot figure out the parameter packs in the variadic
// implementations of traverseStrides and mapStrides, so here we
// implement them manually for unary and binary cases.

template <typename T>
struct StridedIterator
    : public llvm::iterator_facade_base<StridedIterator<T>, std::forward_iterator_tag, T> {
  const llvm::ArrayRef<int64_t> shape;
  const llvm::ArrayRef<int64_t> strides;
  T *ptr;
  llvm::SmallVector<int64_t, 6> axes;

public:
  StridedIterator(llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides, T *ptr)
  : shape(shape), strides(strides), ptr(ptr), axes(shape.size(), 0) {}

  StridedIterator(std::nullptr_t)
  : ptr(nullptr) {}

  StridedIterator(const StridedIterator &) = default;

  StridedIterator &operator=(const StridedIterator &) = default;

  bool operator==(const StridedIterator &other) const {
    return ptr == other.ptr;
  }

  bool operator==(std::nullptr_t) const {
    return ptr == nullptr;
  }

  const T &operator*() const { return *ptr; }

  T &operator*() { return *ptr; }

  StridedIterator &operator++() {
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
    ptr = nullptr;
    return *this;
  }
};

#if 0
template <typename T>
struct StridedIterator
    : public llvm::iterator_facade_base<StridedIterator, std::forward_iterator_tag, T> {
  struct Axis {
    int64_t dim;
    int64_t stride;
    int64_t pos;
  };
  
  T *ptr;
  std::vector<Axis> axes;

public:
  StridedIterator(llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides, pointer ptr)
  : ptr(ptr) {
    assert(shape.size() == strides.size());
    axes.reserve(shape.size());
    for (unsigned i = 0; i < shape.size(); ++i)
      axes.emplace_back(shape[i], strides[i], 0);
  }

  StridedIterator(const StridedIterator &) = default;

  StridedIterator &operator=(const StridedIterator &) = default;

  bool operator==(const StridedIterator &other) const {
    return ptr == other.ptr;
  }

  const T &operator*() const { return *ptr; }

  T &operator*() { return *ptr; }

  StridedIterator &operator++() {
    auto next = ptr;
    auto it = axes.rbegin(), end = axes.rend();
    while (it != end) {
      next += it->str;
      ++(it->pos);
      if (it->pos < it->dim) {
        ptr = next;
        return *this;
      }
      next -= it->pos * it->str;
      it->pos = 0;
    }
    ptr++;
    return *this;
  }
};
#endif

template <typename Iterator, typename Arg0, typename Action>
Iterator traverseStrides(llvm::ArrayRef<int64_t> shape, Iterator begin,
    StridedArrayRef<Arg0> src0, Action &&act) {
#if 1
  Iterator it = begin;
  for (StridedIterator<const Arg0> si(shape, src0.strides, src0.begin()); si != nullptr; ++si, ++it)
    act(it, &*si);
  return it;
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