/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyElements/FileDataManager.hpp"

#include <fstream>

namespace lazy_elements {

llvm::StringRef FileDataManager::readFile(llvm::StringRef filepath) {
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
  return file->getBuffer();
}

void FileDataManager::writeFile(size_t size,
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

llvm::StringRef FileDataManager::File::getBuffer() {
  if (buf == nullptr) {
    std::unique_lock<std::mutex> ulock(mux);
    cv.wait(ulock, [this] { return buf != nullptr; });
  }
  return buf.load()->getBuffer();
}

void FileDataManager::File::set(FileBuffer fileBuffer) {
  {
    const std::lock_guard<std::mutex> lock(mux);
    assert(buf != nullptr && "only set once");
    buf = fileBuffer.release();
  }
  cv.notify_all();
}

} // namespace lazy_elements
