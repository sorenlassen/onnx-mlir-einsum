/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ raw_line_ostream.cpp ------------------------===//
//
// Output stream that forwards the data line by line to a sink.
// This can be used to process the output of the mlir assembly printer.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/raw_line_ostream.hpp"

#include <algorithm>

namespace onnx_mlir {

raw_line_ostream::raw_line_ostream(LineSink sink) : sink(std::move(sink)) {
  SetUnbuffered();
}

raw_line_ostream::~raw_line_ostream() {
  if (!buffer.empty())
    sink(buffer);
}

void raw_line_ostream::write_impl(const char *ptr, size_t size) {
  pos += size;
  const char *end = ptr + size;
  const char *eol = std::find(ptr, end, '\n');
  if (eol != end) {
    buffer.append(ptr, eol + 1);
    sink(buffer);
    buffer.clear();
    ptr = eol + 1;
    while ((eol = std::find(ptr, end, '\n')) != end) {
      sink(llvm::StringRef(ptr, end - (eol + 1)));
      ptr = eol + 1;
    }
  }
  buffer.append(ptr, end);
}

} // namespace onnx_mlir