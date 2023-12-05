// SPDX-License-Identifier: Apache-2.0

include "mlir/IR/DialectBase.td"

def LazyCst_Dialect : Dialect {
  let name = "lazycst";
  let description = [{
    The Lazy Constants dialect provides attributes and operations to delay
    loading and transformations of large constants to reduce compile time
    and memory usage.
  }];
  let cppNamespace = "::lazycst";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
  private:
    // Called from LazyCstDialect::initialize() to register dialect attributes.
    void registerAttributes();

    std::atomic<unsigned> counter;
  public:
    FileDataManager fileDataManager;
    LazyCstExprManager lazyCstExprManager;
    ConstantFolders constantFolders;

    // Generate a new lazycst.expr name that is unused in symbolTable.
    mlir::StringAttr nextExprName(const mlir::SymbolTable &symbolTable);
  }];
}
