/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- FrontendDialectHelper.cpp ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for handling input ONNX models.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SwapByteOrder.h"

#include "src/Builder/FrontendDialectHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/DType.hpp"

// Enable DenseElementsAttr to operate on float_16, bfloat_16 data types.
// TODO: move to AttributesHelper.hpp
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::float_16> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::bfloat_16> {
  static constexpr bool value = true;
};

namespace {

// Parses unsigned number.
size_t parseOffsetOrLength(const std::string &value) {
  char *end = nullptr;
  size_t offsetOrLength = strtoull(value.c_str(), &end, 0);
  assert(end != value.c_str() && "failed to parse offset or length");
  return offsetOrLength;
}

// Reads external data from file location specified in tensor proto.
// See https://github.com/onnx/onnx/blob/main/docs/ExternalData.md
std::unique_ptr<llvm::MemoryBuffer> readExternalData(
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  std::string location;
  uint64_t offset = 0;
  uint64_t length = -1; // MemoryBuffer uses -1 to mean infinity
  for (const onnx::StringStringEntryProto &entry : tp.external_data()) {
    assert(entry.has_key() && "external_data entry must have key");
    assert(entry.has_value() && "external_data entry must have value");
    if (entry.key() == "location") {
      location = entry.value();
    } else if (entry.key() == "offset") {
      offset = parseOffsetOrLength(entry.value());
    } else if (entry.key() == "length") {
      length = parseOffsetOrLength(entry.value());
    }
  }
  assert(!location.empty() && "missing external data location");
  llvm::SmallVector<char> path(externalDataDir.begin(), externalDataDir.end());
  llvm::sys::path::append(path, location);
  auto bufferOrError = llvm::MemoryBuffer::getFileSlice(
      path, length, offset, /*IsVolatile=*/false);
  if (std::error_code ec = bufferOrError.getError()) {
    std::string pathStr(path.data(), path.size());
    llvm::errs() << "Error " << ec.message() << " reading from file " << pathStr
                 << ", offset=" << offset << ", length=" << length << "\n";
    llvm_unreachable("llvm::MemoryBuffer::getFileSlice failed");
  }
  return std::move(bufferOrError.get());
}

template <typename T>
struct TransformValueToONNXData {
  static const google::protobuf::RepeatedField<int32_t> &data(
      const onnx::TensorProto &tp) {
    // int32_data is used for int32, uint8, int8, uint16, int16, bool
    return tp.int32_data();
  }
};

template <>
struct TransformValueToONNXData<double> {
  static const google::protobuf::RepeatedField<double> &data(
      const onnx::TensorProto &tp) {
    return tp.double_data();
  }
};

template <>
struct TransformValueToONNXData<float> {
  static const google::protobuf::RepeatedField<float> &data(
      const onnx::TensorProto &tp) {
    return tp.float_data();
  }
};

template <>
struct TransformValueToONNXData<int64_t> {
  static const google::protobuf::RepeatedField<int64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.int64_data();
  }
};

template <>
struct TransformValueToONNXData<uint32_t> {
  static const google::protobuf::RepeatedField<uint64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.uint64_data();
  }
};

template <>
struct TransformValueToONNXData<uint64_t> {
  static const google::protobuf::RepeatedField<uint64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.uint64_data();
  }
};

template <typename To, typename From>
std::enable_if_t<!onnx_mlir::isFP16Type<To>, To> DeserializeDatum(From from) {
  return from;
}
template <typename To, typename From>
std::enable_if_t<onnx_mlir::isFP16Type<To>, To> DeserializeDatum(From from) {
  return To::bitcastFromU16(from);
}

// When the protobuf repeated field has a type of the same size as T,
// access the data directly via ArrayRef.
template <typename T, typename U>
std::enable_if_t<std::is_same_v<T, U>, mlir::DenseElementsAttr>
createDenseElmAttrFromProtoData(const google::protobuf::RepeatedField<U> &data,
    mlir::RankedTensorType tensorType) {
  return mlir::DenseElementsAttr::get(
      tensorType, llvm::makeArrayRef(data.data(), data.size()));
}

// When the protobuf repeated field has a type larger than T,
// copy the data into correctly typed SmallVector because
// DenseElementsAttr needs argument type of the correct bitwidth.
template <typename T, typename U>
std::enable_if_t<!std::is_same_v<T, U>, mlir::DenseElementsAttr>
createDenseElmAttrFromProtoData(const google::protobuf::RepeatedField<U> &data,
    mlir::RankedTensorType tensorType) {
  llvm::SmallVector<T> copy;
  copy.resize_for_overwrite(data.size());
  std::transform(data.begin(), data.end(), copy.data(), DeserializeDatum<T, U>);
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(copy));
}

template <typename T>
std::enable_if_t<!std::is_same_v<T, bool> && !onnx_mlir::isFP16Type<T>, T>
swappedBytes(T x) {
  return llvm::sys::getSwappedBytes(x);
}
template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, T> swappedBytes(T x) {
  assert(sizeof(x) == 1 && "bool should fit into 1 byte");
  return x;
}
template <typename T>
std::enable_if_t<onnx_mlir::isFP16Type<T>, T> swappedBytes(T x) {
  return T::bitcastFromU16(llvm::sys::getSwappedBytes(x.bitcastToU16()));
}

// Returns DenseElementsAttr with tp's data.
template <typename T>
mlir::DenseElementsAttr createDenseElmAttr(const std::string &externalDataDir,
    const onnx::TensorProto &tp, mlir::RankedTensorType tensorType) {
  std::unique_ptr<llvm::MemoryBuffer> externalData =
      (tp.has_data_location() &&
          tp.data_location() == onnx::TensorProto::EXTERNAL)
          ? readExternalData(externalDataDir, tp)
          : nullptr;
  if (externalData || tp.has_raw_data()) {
    llvm::StringRef buffer = externalData ? externalData->getBuffer()
                                          : llvm::StringRef(tp.raw_data());
    size_t size = buffer.size() / sizeof(T);
    llvm::ArrayRef<T> arrayRef(
        reinterpret_cast<T const *>(buffer.data()), size);
    // Perform byte swap if system endianness is BE.
    // ONNX tensor content raw data is always in LE.
    if (sizeof(T) > 1 && llvm::support::endian::system_endianness() !=
                             llvm::support::endianness::little) {
      llvm::SmallVector<T> vector;
      vector.reserve(size);
      for (T x : arrayRef) {
        vector.push_back(swappedBytes(x));
      }
      return mlir::DenseElementsAttr::get(
          tensorType, llvm::makeArrayRef(vector));
    } else {
      // No need to take care of endianness.
      return mlir::DenseElementsAttr::get(tensorType, arrayRef);
    }
  } else {
    // Not raw, no need to take care of endianness.
    const auto &data = TransformValueToONNXData<T>::data(tp);
    return createDenseElmAttrFromProtoData<T>(data, tensorType);
  }
}

} // namespace

namespace onnx_mlir {

mlir::Value EmitInitializerForInputTensor(mlir::Location loc,
    mlir::OpBuilder &builder, const std::string &externalDataDir,
    const onnx::TensorProto &initializer) {
  // Return none if the initializer is an empty tensor, e.g tensor<0xf32>.
  llvm::ArrayRef<int64_t> tensorDims(
      initializer.dims().data(), initializer.dims().size());
  if (tensorDims.size() == 1 && tensorDims[0] == 0)
    return builder.create<mlir::ONNXNoneOp>(
        loc, builder.getNoneType(), builder.getUnitAttr());

  mlir::DenseElementsAttr denseElmAttr =
      onnxTensorProtoToDenseElmAttr(builder, externalDataDir, initializer);
  return builder.create<mlir::ONNXConstantOp>(loc, nullptr, denseElmAttr);
}

mlir::DenseElementsAttr onnxTensorProtoToDenseElmAttr(mlir::OpBuilder &builder,
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  // Tensor dimensions.
  DType dtype = dtypeOfOnnxDataType(tp.data_type());
  mlir::Type elmType = mlirTypeOfDType(dtype, builder.getContext());
  llvm::ArrayRef<int64_t> tensorDims(tp.dims().data(), tp.dims().size());
  auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
  return dispatch(elmType, [&](auto typeToken) {
    using CppType = decltype(typeToken);
    return createDenseElmAttr<CppType>(externalDataDir, tp, tensorType);
  });
}

} // namespace onnx_mlir
