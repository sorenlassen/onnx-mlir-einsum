/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.cpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
//
// NOTE: This source file is to compiled separately and linked by CMake,
// instead it's included by ONNXOps.cpp, because it needs to see the
// complete definition of DisposableElementsAttributeStorage in order to
// add DisposableElementsAttr to the ONNX dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp" // ONNXConstantOp

#include "mlir/IR/BuiltinDialect.h"

using namespace onnx_mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

struct DisposableElementsAttributeStorage : public AttributeStorage {
  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Properties = DisposableElementsAttributeProperties;
  using Reader = DisposableElementsAttributeReader;
  using KeyTy = std::tuple<ShapedType, Strides, Properties>;

  // Constructs only type and strides while the caller sets buffer and reader
  // after construction to minimize copying.
  DisposableElementsAttributeStorage(
      ShapedType type, Strides strides, Properties properties)
      : type(type), strides(strides), properties(properties) {}

  // Equality and hashKey are engineered to defeat the storage uniquer.
  // We don't want uniqueing because we can't compare readers for equality
  // and we could be in a sitation later where we have the same data or the
  // same buffer address but there is an undetectable mismatch because the
  // buffer and reader were disposed by garbage collection.
  bool operator==(const KeyTy &key) const { return false; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    // Generates a unique number each time it is called to defeat the storage
    // uniquer.
    static std::atomic<size_t> counter{0};
    return ++counter;
  }

  static DisposableElementsAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    ShapedType type = std::get<0>(key);
    Strides strides = std::get<1>(key);
    Properties properties = std::get<2>(key);
    return new (allocator.allocate<DisposableElementsAttributeStorage>())
        DisposableElementsAttributeStorage(
            type, allocator.copyInto(strides), properties);
  }

  // The tensor shape and element type that this object represents.
  // The template type T (a Cpp type bool, float, int8_t, etc) may not match
  // the element type and the caller must cast T to the element type to read
  // the underlying data.
  ShapedType type;

  // Specifies how to map positions expressed in type's shape to the flat
  // indices in buffer. strides can express that buffer is not in the default
  // row-major order (maybe as a result of a transpose) or requires broadcast
  // to fill in type's shape. A special case is when the buffer holds a single
  // splat value that broadcasts to shape's size with all-zero strides.
  Strides strides;

  Properties properties;

  // shared_ptr to an underlying MemoryBuffer which can be either heap allocated
  // or a mmap'ed file or point to the raw data of a DenseElementsAttr.
  //
  // The buffer elements' data type may not match T, namely when the transform
  // function transforms the buffer data type to another data type.
  // The buffer elements' data type is not knowable, but you can compute the
  // number of elements from strides and type's shape and then deduce the
  // data type bytewidth from the buffer's size in bytes.
  //
  // Garbage collection clears the buffer when the DisposableElementsAttr is
  // disposed.
  //
  // Multiple DisposableElementsAttr can point to the same MemoryBuffer.
  // The MemoryBuffer is destroyed (and heap allocated data freed or mmap'ed
  // file closed) when no one points to it anymore.
  Buffer buffer;

  // Reads the buffer elements to WideNums corresponding to type's
  // element type. Is set to the identity reader function if data is not
  // transformed, namely when properties.isTransformed is false.
  //
  // Garbage collection clears the reader when the DisposableElementsAttr is
  // disposed.
  Reader reader;
};

namespace {

template <DType DTYPE>
void identityReader(StringRef s, MutableArrayRef<WideNum> dst) {
  using X = CppType<DTYPE>;
  auto src = asArrayRef<X>(s);
  assert(src.size() == dst.size());
  std::transform(src.begin(), src.end(), dst.begin(),
      [](X x) { return WideNum::from<X>(DTYPE, x); });
}

DisposableElementsAttributeReader getIdentityReader(onnx_mlir::DType dtype) {
  return dispatchByDType(
      dtype, [](auto staticDType) { return identityReader<staticDType>; });
}

DisposableElementsAttributeReader getSplatReader(
    onnx_mlir::DType dtype, StringRef rawBytes) {
  unsigned bytewidth = bytewidthOfDType(dtype);
  ArrayRef<char> memory = asArrayRef(rawBytes.take_front(bytewidth));
  WideNum splatValue = WideNum::load(dtype, memory);
  return [=](StringRef s, MutableArrayRef<WideNum> dst) {
    assert(s.size() == bytewidth);
    assert(dst.size() == 1);
    *dst.begin() = splatValue;
  };
}

} // namespace

/*static*/
DisposableElementsAttr DisposableElementsAttr::create(ShapedType type,
    Strides strides, Properties properties, const Buffer &buffer,
    Reader reader) {
  DisposableElementsAttr a =
      Base::get(type.getContext(), type, strides, properties);
  Storage &s = *a.getImpl();
  s.buffer = buffer;
  if (reader) {
    s.reader = std::move(reader);
  } else {
    assert(wideDTypeOfDType(properties.bufferDType) ==
               wideDTypeOfDType(properties.dtype) &&
           "buffer wide type mismatch requires transforming reader");
    if (properties.isBufferSplat) {
      s.reader = getSplatReader(properties.bufferDType, buffer->getBuffer());
    }
    s.reader = getIdentityReader(properties.bufferDType);
  }
  return a;
}

bool DisposableElementsAttr::isDisposed() const {
  //  TODO: Decide if a splat value can be represented with a constant
  //        reader with no buffer; in that case isDisposed should
  //        only return true if both buffer and reader are null.
  return !getImpl()->buffer;
}

ShapedType DisposableElementsAttr::getType() const { return getImpl()->type; }

auto DisposableElementsAttr::getStrides() const -> Strides {
  return getImpl()->strides;
}

auto DisposableElementsAttr::getProperties() const -> const Properties & {
  return getImpl()->properties;
}

auto DisposableElementsAttr::getBuffer() const -> const Buffer & {
  assert(!isDisposed());
  return getImpl()->buffer;
}

auto DisposableElementsAttr::getReader() const -> const Reader & {
  assert(!isDisposed());
  return getImpl()->reader;
}

void DisposableElementsAttr::printWithoutType(raw_ostream &os) const {
  printIntOrFPElementsAttrAsDenseWithoutType(*this, os);
}

/*static*/
DisposablePool &DisposablePool::create(MLIRContext *context) {
  return context->getLoadedDialect<BuiltinDialect>()
      ->addInterface<DisposablePool>(context);
}

/*static*/
DisposablePool *DisposablePool::get(MLIRContext *context) {
  return context->getLoadedDialect<BuiltinDialect>()
      ->getRegisteredInterface<DisposablePool>();
}

DisposablePool::DisposablePool(Dialect *dialect, MLIRContext *context)
    : Base(dialect), pool() {}
DisposablePool::~DisposablePool() {}

void DisposablePool::insert(DisposableElementsAttr d) {
  auto insertion = pool.insert(d.getImpl());
  if (!insertion.second)
    llvm_unreachable("cannot insert existing DisposableElementsAttr");
}

void DisposablePool::garbageCollectUnreachable(ModuleOp moduleOp) {
  Pool reachable;
  moduleOp.walk([&reachable, this](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto elements = attr->dyn_cast<DisposableElementsAttr>()) {
        assert(this->pool.count(elements.getImpl()) == 1 &&
               "reachable disposables must be in the pool");
        reachable.insert(elements.getImpl());
      }
  });
  eraseUnreachable(reachable);
}

void DisposablePool::scrub(ModuleOp moduleOp) {
  moduleOp.walk([&](ONNXConstantOp constOp) {
    if (auto attr = constOp.value())
      if (auto elements = attr->dyn_cast<DisposableElementsAttr>()) {
        assert(this->pool.count(elements.getImpl()) == 1 &&
               "reachable disposables must be in the pool");
        ArrayBuffer<char> rawBuffer = getElementsRawBytes(elements);
        constOp.valueAttr(DenseElementsAttr::getFromRawBuffer(
            elements.getType(), rawBuffer.get()));
      }
  });
  eraseUnreachable({});
}

void DisposablePool::eraseUnreachable(const Pool &reachable) {
  for (Pool::iterator it = pool.begin(); it != pool.end();) {
    DisposableElementsAttributeStorage *p = *it;
    if (pool.count(p) == 0) {
      // p is unreachable, so we reset the buffer payload shared_ptr
      // which decreases the reference count and, if it reached zero,
      // frees or closes the underlying MemoryBuffer's heap allocation or file.
      p->buffer.reset();
      p->reader = nullptr;
      it = pool.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace mlir