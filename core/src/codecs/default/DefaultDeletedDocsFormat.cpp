// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// under the License.
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "codecs/default/DefaultDeletedDocsFormat.h"

#include <fcntl.h>
#include <unistd.h>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <memory>
#include <string>
#include <vector>

#include "segment/Types.h"
#include "utils/Exception.h"
#include "utils/Log.h"

namespace milvus {
namespace codec {

void
DefaultDeletedDocsFormat::read(const storage::FSHandlerPtr& fs_ptr, segment::DeletedDocsPtr& deleted_docs) {
    const std::lock_guard<std::mutex> lock(mutex_);

    std::string dir_path = fs_ptr->operation_ptr_->GetDirectory();
    const std::string del_file_path = dir_path + "/" + deleted_docs_filename_;

    if (!fs_ptr->reader_ptr_->open(del_file_path)) {
        std::string err_msg = "Fail to open file: " + del_file_path + ", error: " + std::strerror(errno);
        LOG_ENGINE_ERROR_ << err_msg;
        throw Exception(SERVER_CANNOT_CREATE_FILE, err_msg);
    }

    int64_t pos = 0;
    fs_ptr->reader_ptr_->seekg(0);

    size_t num_bytes;
    fs_ptr->reader_ptr_->read(&num_bytes, sizeof(num_bytes));
    pos += sizeof(num_bytes);
    fs_ptr->reader_ptr_->seekg(pos);

    auto deleted_docs_size = num_bytes / sizeof(segment::offset_t);
    std::vector<segment::offset_t> deleted_docs_list;
    deleted_docs_list.resize(deleted_docs_size);

    fs_ptr->reader_ptr_->read(deleted_docs_list.data(), num_bytes);

    deleted_docs = std::make_shared<segment::DeletedDocs>(deleted_docs_list);

    fs_ptr->reader_ptr_->close();
}

void
DefaultDeletedDocsFormat::write(const storage::FSHandlerPtr& fs_ptr, const segment::DeletedDocsPtr& deleted_docs) {
    const std::lock_guard<std::mutex> lock(mutex_);

    std::string dir_path = fs_ptr->operation_ptr_->GetDirectory();
    const std::string del_file_path = dir_path + "/" + deleted_docs_filename_;

    bool old_del_file_exist = false;
    size_t old_num_bytes;
    std::vector<segment::offset_t> old_deleted_docs_list;

    if (fs_ptr->reader_ptr_->open(del_file_path)) {
        old_del_file_exist = true;
        int64_t pos = 0;
        fs_ptr->reader_ptr_->seekg(0);

        fs_ptr->reader_ptr_->read(&old_num_bytes, sizeof(old_num_bytes));
        pos += sizeof(old_num_bytes);
        fs_ptr->reader_ptr_->seekg(pos);

        auto old_deleted_docs_size = old_num_bytes / sizeof(segment::offset_t);
        old_deleted_docs_list.resize(old_deleted_docs_size);
        fs_ptr->reader_ptr_->read(old_deleted_docs_list.data(), old_num_bytes);

        fs_ptr->reader_ptr_->close();

        fs_ptr->operation_ptr_->DeleteFile(del_file_path);  // remove old del_file
    }

    if (!fs_ptr->writer_ptr_->open(del_file_path)) {
        std::string err_msg = "Fail to open file: " + del_file_path + ", error: " + std::strerror(errno);
        LOG_ENGINE_ERROR_ << err_msg;
        throw Exception(SERVER_WRITE_ERROR, err_msg);
    }

    auto deleted_docs_list = deleted_docs->GetDeletedDocs();
    size_t new_num_bytes;

    if (old_del_file_exist) {
        new_num_bytes = old_num_bytes + sizeof(segment::offset_t) * deleted_docs->GetSize();
        fs_ptr->writer_ptr_->write(&new_num_bytes, sizeof(size_t));
        fs_ptr->writer_ptr_->write(old_deleted_docs_list.data(), old_num_bytes);
    } else {
        new_num_bytes = sizeof(segment::offset_t) * deleted_docs->GetSize();
        fs_ptr->writer_ptr_->write(&new_num_bytes, sizeof(size_t));
    }

    fs_ptr->writer_ptr_->write(deleted_docs_list.data(), sizeof(segment::offset_t) * deleted_docs->GetSize());

    fs_ptr->writer_ptr_->close();
}

void
DefaultDeletedDocsFormat::readSize(const storage::FSHandlerPtr& fs_ptr, size_t& size) {
    const std::lock_guard<std::mutex> lock(mutex_);

    std::string dir_path = fs_ptr->operation_ptr_->GetDirectory();
    const std::string del_file_path = dir_path + "/" + deleted_docs_filename_;

    if (!fs_ptr->reader_ptr_->open(del_file_path)) {
        std::string err_msg = "Fail to open file: " + del_file_path + ", error: " + std::strerror(errno);
        LOG_ENGINE_ERROR_ << err_msg;
        throw Exception(SERVER_CANNOT_CREATE_FILE, err_msg);
    }

    fs_ptr->reader_ptr_->seekg(0);

    size_t num_bytes;
    fs_ptr->reader_ptr_->read(&num_bytes, sizeof(num_bytes));

    size = num_bytes / sizeof(segment::offset_t);

    fs_ptr->reader_ptr_->close();
}

}  // namespace codec
}  // namespace milvus
