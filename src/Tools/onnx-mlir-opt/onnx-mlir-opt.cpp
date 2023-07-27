/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- onnx-mlir-opt.cpp - Optimization Driver ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Main function for onnx-mlir-opt.
// Implements main for onnx-mlir-opt driver.
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "RegisterPasses.hpp"
#include "src/Compiler/CompilerDialects.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Version/Version.hpp"

#include <memory>
#include <string>

using namespace mlir;
using namespace onnx_mlir;

// Options for onnx-mlir-opt only.
static llvm::cl::OptionCategory OnnxMlirOptOptions(
    "ONNX-MLIR-OPT Options", "These are opt frontend options.");

static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
    llvm::cl::desc("<input file>"), llvm::cl::init("-"),
    llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<std::string> output_filename("o",
    llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"), llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool> split_input_file("split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool> verify_diagnostics("verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool> verify_passes("verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptOptions));

void scanAndSetOptLevel(int argc, char **argv) {
  // In decreasing order, so we pick the last one if there are many.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    int num = -1;
    if (currStr.find("--O") == 0)
      num = atoi(&argv[i][3]); // Get the number starting 3 char down.
    else if (currStr.find("-O") == 0)
      num = atoi(&argv[i][2]); // Get the number starting 2 char down.
    // Silently ignore out of bound opt levels.
    if (num >= 0 && num <= 3) {
      OptimizationLevel = (OptLevel)num;
      return;
    }
  }
}

void scanAndSetMAccel(int argc, char **argv) {
  // Scan accelerators and add them to the maccel option.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    if (currStr.find("--maccel=") == 0) {
      std::string accelKind(
          &argv[i][9]); // Get the string starting 9 chars down.
      setTargetAccel(accelKind);
      break;
    }
    if (currStr.find("-maccel=") == 0) {
      std::string accelKind(
          &argv[i][8]); // Get the string starting 8 chars down.
      setTargetAccel(accelKind);
      break;
    }
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Scan Opt Level manually now as it is needed to register passes
  // before command line options are parsed.
  scanAndSetOptLevel(argc, argv);

  // Scan maccel manually now as it is needed to initialize accelerators
  // before ParseCommandLineOptions() is called.
  scanAndSetMAccel(argc, argv);

  // Hide unrelated options except common ones and the onnx-mlir-opt options
  // defined above.
  llvm::cl::HideUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptOptions});

  DialectRegistry registry = registerDialects(maccel);
  registry.insert<tosa::TosaDialect>();

  // Registered passes can be expressed as command line flags, so they must
  // must be registered before command line options are parsed.
  registerPasses(OptimizationLevel);

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();

  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  if (!parseCustomEnvFlagsCommandLineOption(argc, argv, &llvm::errs()) ||
      !llvm::cl::ParseCommandLineOptions(argc, argv,
          getVendorName() + " - A modular optimizer driver\n", &llvm::errs(),
          customEnvFlags.c_str())) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  // Set up the input file.
  std::string error_message;
  auto file = openInputFile(input_filename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to open file; " << error_message << "\n";
    return 1;
  }

  auto output = openOutputFile(output_filename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to compile file; " << error_message << "\n";
    return 1;
  }

  auto passManagerSetupFn = [&](PassManager &pm) {
    MLIRContext *ctx = pm.getContext();
    // MlirOptMain constructed ctx with our registry so we just load all our
    // already registered dialects.
    ctx->loadAllAvailableDialects();

    // Passes are configured with command line options so they must be
    // configured after command line parsing but before any passes are run.
    configurePasses(pm);

    auto errorHandler = [ctx](const Twine &msg) {
      emitError(UnknownLoc::get(ctx)) << msg;
      return failure();
    };
    return passPipeline.addToPipeline(pm, errorHandler);
  };

  MlirOptMainConfig config;
  config.setPassPipelineSetupFn(passManagerSetupFn)
      .splitInputFile(split_input_file)
      .verifyDiagnostics(verify_diagnostics)
      .verifyPasses(verify_passes)
      .allowUnregisteredDialects(allowUnregisteredDialects)
      .emitBytecode(false)
      .useExplicitModule(false);

  if (failed(MlirOptMain(output->os(), std::move(file), registry, config)))
    return 1;

  output->keep();
  return 0;
}
