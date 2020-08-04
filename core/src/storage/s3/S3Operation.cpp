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

#include "storage/s3/S3Operation.h"
#include "storage/s3/S3ClientWrapper.h"
#include "utils/Status.h"
#include "utils/Exception.h"
#include "utils/Log.h"

namespace milvus {
namespace storage {

S3Operation::S3Operation(const std::string& dir_path) : dir_path_(dir_path) {
}

void
S3Operation::CreateDirectory() {
    // use dir_path_ as the prefix of files, instead of creating a real directory
}

const std::string&
S3Operation::GetDirectory() const {
    return dir_path_;
}

void
S3Operation::ListDirectory(std::vector<std::string>& file_paths) {
    // regard dir_path_ as prefix, and get paths of files which have the prefix
    auto status = S3ClientWrapper::GetInstance().ListObjects(file_paths, dir_path_);
    if (!status.ok()) {
        std::string err_msg = "Failed to list S3 directory: " + dir_path_;
        LOG_ENGINE_ERROR_ << err_msg;
        //throw Exception(SERVER_CANNOT_LIST_S3_FOLDER, err_msg);
    }
}

bool
S3Operation::DeleteFile(const std::string& file_path) {
    return (S3ClientWrapper::GetInstance().DeleteObject(file_path).ok());
}

}  // namespace storage
}  // namespace milvus
