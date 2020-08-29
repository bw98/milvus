// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
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

#include "codecs/default/DefaultVectorsFormat.h"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

#include <boost/filesystem.hpp>

#include "utils/Exception.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

namespace milvus {
namespace codec {

void
DefaultVectorsFormat::read_vectors_internal(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                                            off_t offset, size_t num, std::vector<uint8_t>& raw_vectors) {
    if (!fs_ptr->reader_ptr_->open(file_path.c_str())) {
        std::string err_msg = "Failed to open file: " + file_path + ", error: " + std::strerror(errno);
        LOG_ENGINE_ERROR_ << err_msg;
        throw Exception(SERVER_CANNOT_OPEN_FILE, err_msg);
    }

    size_t num_bytes;
    fs_ptr->reader_ptr_->read(&num_bytes, sizeof(size_t));

    num = std::min(num, num_bytes - offset);

    offset += sizeof(size_t);  // Beginning of file is num_bytes
    fs_ptr->reader_ptr_->seekg(offset);

    raw_vectors.resize(num / sizeof(uint8_t));
    fs_ptr->reader_ptr_->read(raw_vectors.data(), num);

    fs_ptr->reader_ptr_->close();
}

void
DefaultVectorsFormat::read_uids_internal(const storage::FSHandlerPtr& fs_ptr, const std::string& file_path,
                                         std::vector<segment::doc_id_t>& uids) {
    if (!fs_ptr->reader_ptr_->open(file_path.c_str())) {
        std::string err_msg = "Failed to open file: " + file_path + ", error: " + std::strerror(errno);
        LOG_ENGINE_ERROR_ << err_msg;
        throw Exception(SERVER_CANNOT_OPEN_FILE, err_msg);
    }

    int64_t pos = 0;
    fs_ptr->reader_ptr_->seekg(0);

    size_t num_bytes;
    fs_ptr->reader_ptr_->read(&num_bytes, sizeof(size_t));
    pos += sizeof(num_bytes);
    fs_ptr->reader_ptr_->seekg(pos);

    uids.resize(num_bytes / sizeof(segment::doc_id_t));
    fs_ptr->reader_ptr_->read(uids.data(), num_bytes);

    fs_ptr->reader_ptr_->close();
}

void
DefaultVectorsFormat::read(const storage::FSHandlerPtr& fs_ptr, segment::VectorsPtr& vectors_read) {
    const std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> file_paths;
    fs_ptr->operation_ptr_->ListDirectory(file_paths);

    for (auto& file_path : file_paths) {
        auto extension_start_pos = file_path.find_last_of('.');
        if (extension_start_pos != file_path.npos) {
            // The file has an extension
            std::string extension = file_path.substr(extension_start_pos);
            if (extension == raw_vector_extension_) {
                auto& vector_list = vectors_read->GetMutableData();
                read_vectors_internal(fs_ptr, file_path, 0, INT64_MAX, vector_list);

                auto slash_pos = file_path.find_last_of('/');
                if (slash_pos != file_path.npos) {
                    std::string filename = file_path.substr(file_path.find_last_of('/') + 1);
                    vectors_read->SetName(filename.substr(0, filename.find_last_of('.')));
                } else {
                    vectors_read->SetName(file_path.substr(0, file_path.find_last_of('.')));
                }
            } else if (extension == user_id_extension_) {
                auto& uids = vectors_read->GetMutableUids();
                read_uids_internal(fs_ptr, file_path, uids);
            }
        }
    }
}

void
DefaultVectorsFormat::write(const storage::FSHandlerPtr& fs_ptr, const segment::VectorsPtr& vectors) {
    const std::lock_guard<std::mutex> lock(mutex_);

    std::string dir_path = fs_ptr->operation_ptr_->GetDirectory();

    const std::string rv_file_path = dir_path + "/" + vectors->GetName() + raw_vector_extension_;
    const std::string uid_file_path = dir_path + "/" + vectors->GetName() + user_id_extension_;

    TimeRecorder rc("write vectors");

    if (!fs_ptr->writer_ptr_->open(rv_file_path.c_str())) {
        std::string err_msg = "Failed to open file: " + rv_file_path + ", error: " + std::strerror(errno);
        LOG_ENGINE_ERROR_ << err_msg;
        throw Exception(SERVER_CANNOT_CREATE_FILE, err_msg);
    }

    size_t rv_num_bytes = vectors->GetData().size() * sizeof(uint8_t);
    fs_ptr->writer_ptr_->write(&rv_num_bytes, sizeof(size_t));
    fs_ptr->writer_ptr_->write((void*)vectors->GetData().data(), rv_num_bytes);
    fs_ptr->writer_ptr_->close();

    rc.RecordSection("write rv done");

    if (!fs_ptr->writer_ptr_->open(uid_file_path.c_str())) {
        std::string err_msg = "Failed to open file: " + uid_file_path + ", error: " + std::strerror(errno);
        LOG_ENGINE_ERROR_ << err_msg;
        throw Exception(SERVER_CANNOT_CREATE_FILE, err_msg);
    }
    size_t uid_num_bytes = vectors->GetUids().size() * sizeof(segment::doc_id_t);
    fs_ptr->writer_ptr_->write(&uid_num_bytes, sizeof(size_t));
    fs_ptr->writer_ptr_->write((void*)vectors->GetUids().data(), uid_num_bytes);
    fs_ptr->writer_ptr_->close();

    rc.RecordSection("write uids done");
}

void
DefaultVectorsFormat::read_uids(const storage::FSHandlerPtr& fs_ptr, std::vector<segment::doc_id_t>& uids) {
    const std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> file_paths;
    fs_ptr->operation_ptr_->ListDirectory(file_paths);

    for (auto& file_path : file_paths) {
        auto extension_start_pos = file_path.find_last_of('.');
        if (extension_start_pos != file_path.npos) {
            // The file has an extension
            std::string extension = file_path.substr(extension_start_pos);
            if (extension == user_id_extension_) {
                read_uids_internal(fs_ptr, file_path, uids);
            }
        }
    }
}

void
DefaultVectorsFormat::read_vectors(const storage::FSHandlerPtr& fs_ptr, off_t offset, size_t num_bytes,
                                   std::vector<uint8_t>& raw_vectors) {
    const std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> file_paths;
    fs_ptr->operation_ptr_->ListDirectory(file_paths);

    for (auto& file_path : file_paths) {
        auto extension_start_pos = file_path.find_last_of('.');
        if (extension_start_pos != file_path.npos) {
            // the file has an extension
            std::string extension = file_path.substr(extension_start_pos);
            if (extension == raw_vector_extension_) {
                read_vectors_internal(fs_ptr, file_path, offset, num_bytes, raw_vectors);
            }
        }
    }
}

}  // namespace codec
}  // namespace milvus
