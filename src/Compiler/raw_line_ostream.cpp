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

LineForwardingRawOstream::LineForwardingRawOstream(
    llvm::raw_ostream &out, LineForwarder fwd)
    : out(out), fwd(std::move(fwd)) {
  SetUnbuffered();
}

LineForwardingRawOstream::~LineForwardingRawOstream() {
  if (!buffer.empty())
    fwd(buffer, out);
}

void LineForwardingRawOstream::write_impl(const char *ptr, size_t size) {
  pos += size;
  const char *end = ptr + size;
  const char *eol = std::find(ptr, end, '\n');
  if (eol != end) {
    buffer.append(ptr, eol + 1);
    fwd(buffer, out);
    buffer.clear();
    ptr = eol + 1;
    while ((eol = std::find(ptr, end, '\n')) != end) {
      fwd(llvm::StringRef(ptr, end - (eol + 1)), out);
      ptr = eol + 1;
    }
  }
  buffer.append(ptr, end);
}

} // namespace onnx_mlir