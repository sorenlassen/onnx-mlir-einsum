// SPDX-License-Identifier: Apache-2.0

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// The Lazy Elements Dialect.
//===----------------------------------------------------------------------===//

def LazyElements_Dialect : Dialect {
  let name = "lazyelms";
  let description = [{
    The Lazy Elements dialect.
  }];
  let cppNamespace = "::lazy_elements";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    FileDataManager fileDataManager;
  }];
}

//===----------------------------------------------------------------------===//
// Lazy Elements Attributes.
//===----------------------------------------------------------------------===//

class LazyElements_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LazyElements_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def LazyElements_FileDataAttr : LazyElements_Attr<"FileData", "file_data", [
    ElementsAttrInterface
  ]> {
  let summary = "Attribute for external data: file with raw bytes.";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    "mlir::StringAttr":$path,
    OptionalParameter<"uint64_t">:$offset
  );
  let assemblyFormat = "`<` $path ( `,` struct($offset)^ )? `>`";
}

def LazyElements_LazyElementsAttr : LazyElements_Attr<"LazyElements", "lazy", [
    ElementsAttrInterface
  ]> {
  let summary = "Lazily evaluated elements attribute.";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
    "mlir::StringAttr":$operation,
    ArrayRefParameter<"mlir::ElementsAttr", "">:$operands,
    OptionalParameter<"mlir::DictionaryAttr">:$attributes
  );
  let assemblyFormat = "`<` $operation `(` $operands `)` ($attributes^)? `>`";
  // TODO: add fold / materialization methods
}
