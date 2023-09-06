/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Dialect/LazyCst/FileDataManager.hpp"

#include <fstream>

namespace lazycst {

llvm::StringRef FileDataManager::readFile(llvm::StringRef filepath) {
  File *file = nullptr;
  bool exists = false;
  {
    std::lock_guard<std::mutex> lock(filesMux);
    auto [iter, inserted] = files.try_emplace(filepath, nullptr);
    exists = !inserted;
    file = &iter->second;
  }
  if (exists) {
    return file->getBuffer();
  } else {
    std::filesystem::path fullpath;
    for (const std::filesystem::path &dir : config.readDirectoryPaths) {
      fullpath = dir / std::filesystem::path(filepath.begin(), filepath.end());
      std::error_code ec;
      bool exists = std::filesystem::exists(fullpath, ec);
      assert(!ec && "failed to test read file status");
      if (exists)
        break;
      fullpath.clear();
    }
    assert(!fullpath.empty() && "failed to find file to read");

    auto errorOrFilebuf = llvm::MemoryBuffer::getFile(
        fullpath.string(), /*IsText=*/false, /*RequiresNullTerminator=*/false);
    assert(errorOrFilebuf && "failed to read data file");
    FileBuffer filebuf = std::move(errorOrFilebuf.get());
    llvm::StringRef buffer = filebuf->getBuffer();
    file->set(std::move(filebuf));
    return buffer;
  }
}

std::string FileDataManager::writeFile(size_t size,
    const std::function<void(llvm::MutableArrayRef<char>)> writer) {
  uint64_t fileNumber = nextFileNumber++;
  std::string filepath = config.writePathPrefix + std::to_string(fileNumber) +
                         config.writePathSuffix;
  std::filesystem::path fullpath = config.writeDirectoryPath;
  fullpath /= filepath;
  {
    std::ofstream create(fullpath, std::ios::binary | std::ios::trunc);
    assert(!create.fail() && "failed to create data file");
  }
  std::error_code ec;
  std::filesystem::resize_file(fullpath, size, ec);
  assert(!ec && "failed to write data file");
  auto errorOrFilebuf =
      llvm::WriteThroughMemoryBuffer::getFile(fullpath.native(), size);
  assert(errorOrFilebuf && "failed to mmap data file");
  std::unique_ptr<llvm::WriteThroughMemoryBuffer> filebuf =
      std::move(errorOrFilebuf.get());
  writer(filebuf->getBuffer());
  {
    std::lock_guard<std::mutex> lock(filesMux);
    auto [iter, inserted] = files.try_emplace(filepath, std::move(filebuf));
    assert(inserted && "written data file cannot already exist");
  }
  return filepath;
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
    assert(buf == nullptr && "only set once");
    buf = fileBuffer.release();
  }
  cv.notify_all();
}

} // namespace lazycst
