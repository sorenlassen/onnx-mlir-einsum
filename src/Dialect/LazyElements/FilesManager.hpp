/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace lazy_elements {

// TODO:
// * configure read base paths
// * configure write base path, prefix, suffix
//
// Outside this class also track which files are read.
//
//
class FilesManager {
public:
  using FileBuffer = std::unique_ptr<llvm::MemoryBuffer>;

  struct Config {
    std::vector<std::string> readDirectoryPaths;
    std::string writePathPrefix;
    std::string writePathSuffix;
  };

  void configure(const Config &config) { this->config = config; };

  llvm::StringRef readFile(llvm::StringRef filepath) {
    File *file = nullptr;
    {
      std::lock_guard<std::mutex> lock(filesMux);
      auto [iter, inserted] = files.try_emplace(filepath, nullptr);
      if (inserted) {
        // TODO: read file and call iter->second.set(..)
      } else {
        file = &iter->second;
      }
    }
    return file->buffer();
  }

  void writeFile(size_t size,
      const std::function<void(llvm::MutableArrayRef<char>)> writer) {
    uint64_t fileNumber = writeCounter++;
    std::string filePath = config.writePathPrefix + std::to_string(fileNumber) +
                           config.writePathSuffix;
    // TODO:
    // * construct a file at filePath with length size
    //   (seek to end and write a zero byte)
    // * make a WriteThroughMemoryBuffer
    // * call writer to fill it up
    // * take filesMux and put it in files
  }

private:
  class File {
  public:
    File(llvm::MemoryBuffer *buf) : buf(buf) {}
    ~File() { delete buf; }
    llvm::StringRef buffer() {
      if (buf == nullptr) {
        std::unique_lock<std::mutex> ulock(mux);
        cv.wait(ulock, [this] { return buf != nullptr; });
      }
      return buf.load()->getBuffer();
    }
    void set(FileBuffer fileBuffer) {
      {
        const std::lock_guard<std::mutex> lock(mux);
        assert(buf != nullptr && "only set once");
        buf = fileBuffer.release();
      }
      cv.notify_all();
    }

  private:
    // TODO: replace mux and cv with simpler std::latch in C++20
    // TODO: use atomic<unique_ptr> for buf in C++20
    std::mutex mux;
    std::condition_variable cv;
    std::atomic<llvm::MemoryBuffer *> buf;
  };

  Config config;

  std::atomic<uint64_t> writeCounter = 0;

  std::mutex filesMux;
  llvm::StringMap<File> files;
};

} // namespace lazy_elements
