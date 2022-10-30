/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ raw_line_ostream.hpp ------------------------===//
//
// Output stream that forwards the data line by line to a sink.
// This can be used to process the output of the mlir assembly printer.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/raw_ostream.h"

#include <deque>
#include <functional>
#include <string>

namespace onnx_mlir {

class raw_line_ostream : public llvm::raw_ostream {
public:
  using LineSink = std::function<void(std::string)>;

  explicit raw_line_ostream(LineSink sink);
  ~raw_line_ostream() override;

private:
  void write_impl(const char *ptr, size_t size) override;

  uint64_t current_pos() const override { return pos; }

  LineSink sink;
  std::deque<char> deq;
  uint64_t pos = 0;
};

} // namespace onnx_mlir