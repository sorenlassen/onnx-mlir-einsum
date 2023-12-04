// SPDX-License-Identifier: Apache-2.0

include "mlir/IR/DialectBase.td"

def LazyCst_Dialect : Dialect {
  let name = "lazycst";
  let description = [{
    The Lazy Constants dialect.
  }];
  let cppNamespace = "::lazycst";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
  private:
    void registerAttributes(); // Register the dialect attributes.
    std::atomic<unsigned> counter;
  public:
    FileDataManager fileDataManager;
    LazyCstExprManager lazyCstExprManager;
    ConstantFolders constantFolders;
    mlir::StringAttr nextExprName();
  }];
}
