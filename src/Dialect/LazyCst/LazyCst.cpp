/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/LazyCst.hpp"

#include "src/Support/Arrays.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <string.h>

void lazycst::LazyCstDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/LazyCst/LazyCstAttributes.cpp.inc"
      >();
}

#include "src/Dialect/LazyCst/LazyCstDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/LazyCst/LazyCstAttributes.cpp.inc"

using namespace mlir;

namespace lazycst {

namespace {
#define LETTERS "ABCDEFGHIJKLMNOPQRSTUVWZYZabcdefghijklmnopqrstuvwxyz"
#define DIGITS "0123456789"
// constexpr char validLeadingIdentifierChars[] = "_" LETTERS;
constexpr char validIdentifierChars[] = "_$." DIGITS LETTERS;
} // namespace

std::string escapeIdentifier(StringRef unescapedIdentifier) {
  SmallString<64> escaped;
  escaped.reserve(unescapedIdentifier.size());
  for (unsigned char c : unescapedIdentifier) {
    if (c == '_' || strchr(validIdentifierChars, c) == nullptr) {
      escaped.push_back('_');
      escaped.push_back(llvm::hexdigit(c / 16));
      escaped.push_back(llvm::hexdigit(c & 0xf));
    } else {
      escaped.push_back(c);
    }
  }
  return std::string(escaped);
}

std::string unescapeIdentifier(StringRef escapedIdentifier) {
  SmallString<64> unescaped;
  unescaped.reserve(escapedIdentifier.size());
  for (size_t i = 0; i < escapedIdentifier.size();) {
    char c = escapedIdentifier[i];
    ++i;
    if (c == '_') {
      assert(i + 2 <= escapedIdentifier.size());
      unsigned high = llvm::hexDigitValue(escapedIdentifier[i]);
      assert(high < 16);
      unsigned low = llvm::hexDigitValue(escapedIdentifier[i + 1]);
      assert(low < 16);
      i += 2;
      c = high * 16 + low;
    }
    unescaped.push_back(c);
  }
  return std::string(unescaped);
}

// template <class C>
// StringAttr BufferElementsAttr<C>::getPath() const {
//   // using T = typename C::ContiguousIterableTypesT;
//   return static_cast<typename C::ImplType *>(impl)->path;
// }

llvm::ArrayRef<char> FileDataElementsAttr::getRawBytesImpl() const {
  LazyCstDialect *lazyElementsDialect =
      getContext()->getLoadedDialect<LazyCstDialect>();
  StringRef buffer = lazyElementsDialect->fileDataManager.readFile(getPath());
  return onnx_mlir::asArrayRef(buffer);
}

} // namespace lazycst