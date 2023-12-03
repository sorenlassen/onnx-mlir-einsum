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

namespace lazycst {

// Memory-maps files with MemoryBuffer. Tracks the memory mapping of every read
// of written file in a map keyed by filepath and reuses the existing memory
// mapping if a file is read after it was already read or written.
class FileDataManager {
public:
  struct Config {
    std::vector<std::filesystem::path> readDirectoryPaths;
    std::filesystem::path writeDirectoryPath;
    std::string writePathPrefix;
    std::string writePathSuffix;
  };

  void configure(const Config &config) {
    // TODO: decide if we should enforce that write path is in read paths
    this->config = config;
  };

  // Returns the contents of the file at filepath as a StringRef,
  // like llvm::MemoryBuffer::getBuffer(), where filepath can be a relative
  // path which is searched in all configured readDirectoryPaths.
  // Crashes with assertion failure if filepath is not found.
  //
  // TODO: consider returning ArrayRef<char> instead of StringRef.
  llvm::StringRef readFile(llvm::StringRef filepath);

  // Writes a file in writeDirectoryPath/writePathPrefix{N}writePathSuffix
  // where {N} is a non-negative number with `size` bytes produced by
  // invoking `writer`.
  // Returns the relative file path without writeDirectoryPath.
  // The written file is memory-mapped and can be read with readFile()
  // if writeDirectoryPath is in readDirectoryPaths.
  std::string writeFile(size_t size,
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

  std::mutex filesMutex;
  llvm::StringMap<File> files;
};

} // namespace lazycst
