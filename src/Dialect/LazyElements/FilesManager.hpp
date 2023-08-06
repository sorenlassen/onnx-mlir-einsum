/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <atomic>
#include <filesystem>
#include <fstream>
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
    std::vector<std::filesystem::path> readDirectoryPaths;
    std::filesystem::path writePathPrefix;
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
    uint64_t fileNumber = nextFileNumber++;
    std::filesystem::path filepath = config.writePathPrefix;
    filepath += std::to_string(fileNumber) + config.writePathSuffix;
    std::ofstream outFile(filepath, std::ofstream::app);
    {
      std::ofstream create(filepath, std::ios::binary | std::ios::trunc);
      assert(!create.fail() && "failed to create data file");
    }
    std::error_code ec;
    std::filesystem::resize_file(filepath, size, ec);
    assert(!ec && "failed to write data file");
    auto errorOrFilebuf =
        llvm::WriteThroughMemoryBuffer::getFile(filepath.native(), size);
    assert(errorOrFilebuf && "failed to mmap data file");
    std::unique_ptr<llvm::WriteThroughMemoryBuffer> filebuf =
        std::move(errorOrFilebuf.get());
    writer(filebuf->getBuffer());
    {
      std::lock_guard<std::mutex> lock(filesMux);
      auto [iter, inserted] =
          files.try_emplace(filepath.native(), std::move(filebuf));
      assert(inserted && "written data file cannot already exist");
    }
  }

private:
  class File {
  public:
    File(FileBuffer fileBuffer) : buf(fileBuffer.release()) {}
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

  std::atomic<uint64_t> nextFileNumber = 0;

  std::mutex filesMux;
  llvm::StringMap<File> files;
};

} // namespace lazy_elements
