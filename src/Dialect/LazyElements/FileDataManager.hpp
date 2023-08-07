/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace lazy_elements {

class FileDataManager {
public:
  struct Config {
    std::vector<std::filesystem::path> readDirectoryPaths;
    std::filesystem::path writePathPrefix;
    std::string writePathSuffix;
  };

  void configure(const Config &config) { this->config = config; };

  llvm::StringRef readFile(llvm::StringRef filepath);

  void writeFile(size_t size,
      const std::function<void(llvm::MutableArrayRef<char>)> writer);

private:
  using FileBuffer = std::unique_ptr<llvm::MemoryBuffer>;

  class File {
  public:
    File(FileBuffer fileBuffer) : buf(fileBuffer.release()) {}
    ~File() { delete buf; }
    llvm::StringRef getBuffer();
    void set(FileBuffer fileBuffer);

  private:
    // TODO: replace mux and cv with simpler std::latch in C++20
    // TODO: use atomic<unique_ptr> for buf in C++20
    std::mutex mux;
    std::condition_variable cv;
    std::atomic<llvm::MemoryBuffer *> buf;
  };

  Config config;

  std::atomic<uint64_t> nextFileNumber = 0;

  std::mutex filesMux;
  llvm::StringMap<File> files;
};

} // namespace lazy_elements
