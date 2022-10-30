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

namespace onnx_mlir {

raw_line_ostream::raw_line_ostream(LineSink sink) : sink(std::move(sink)) {
  SetUnbuffered();
}

raw_line_ostream::~raw_line_ostream() {
  if (!deq.empty())
    sink(std::string(deq.begin(), deq.end()));
}

void raw_line_ostream::write_impl(const char *ptr, size_t size) {
  pos += size;
  deq.insert(deq.end(), ptr, ptr + size);
  auto eol = std::find(deq.end() - size, deq.end(), '\n');
  while (eol != deq.end()) {
    sink(std::string(deq.begin(), eol + 1));
    deq.erase(deq.begin(), eol + 1);
    eol = std::find(deq.begin(), deq.end(), '\n');
  }
}

} // namespace onnx_mlir