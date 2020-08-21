// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>
//#include <fiu-local.h>
//#include <fiu-control.h>

#include "segment/SegmentReader.h"
#include "segment/SegmentWriter.h"
//#include "easyloggingpp/easylogging++.h"
#include "segment/utils.h"

//INITIALIZE_EASYLOGGINGPP

TEST_F(SegmentTest, SEGMENT_DELETEDDOCS_SINGLE_RW_TEST) {
    const std::string segment_dir = "/tmp";

    {
        // SegmentWriter do something here
        milvus::segment::SegmentWriter segment_writer(segment_dir);

        /* test to write deleted docs */
        milvus::segment::DeletedDocsPtr deleted_docs_ptr = std::make_shared<milvus::segment::DeletedDocs>();
        ASSERT_TRUE(segment_writer.WriteDeletedDocs(deleted_docs_ptr).ok());  // write empty deleted docs
    }

    {
        // SegmentReader do something here
        milvus::segment::SegmentReader segment_reader(segment_dir);

        /* test to read deleted docs */
        milvus::segment::DeletedDocsPtr deleted_docs_ptr = std::make_shared<milvus::segment::DeletedDocs>();
        ASSERT_TRUE(segment_reader.LoadDeletedDocs(deleted_docs_ptr).ok());
        size_t deleted_docs_size;
        ASSERT_TRUE(segment_reader.ReadDeletedDocsSize(deleted_docs_size).ok());
    }
}

TEST_F(SegmentTest, SEGMENT_DELETEDDOCS_MULTIPLE_RW_TEST) {
    const std::string segment_dir = "/tmp";

    {
        // SegmentWriter do something here
        milvus::segment::SegmentWriter segment_writer(segment_dir);

        /* test to write deleted docs */
        milvus::segment::DeletedDocsPtr deleted_docs_ptr = std::make_shared<milvus::segment::DeletedDocs>();
        ASSERT_TRUE(segment_writer.WriteDeletedDocs(deleted_docs_ptr).ok());  // write empty deleted docs
        deleted_docs_ptr = std::make_shared<milvus::segment::DeletedDocs>();
        ASSERT_TRUE(segment_writer.WriteDeletedDocs(deleted_docs_ptr).ok());
    }

    {
        // SegmentReader do something here
        milvus::segment::SegmentReader segment_reader(segment_dir);

        /* test to read deleted docs */
        milvus::segment::DeletedDocsPtr deleted_docs_ptr = std::make_shared<milvus::segment::DeletedDocs>();
        ASSERT_TRUE(segment_reader.LoadDeletedDocs(deleted_docs_ptr).ok());
        size_t deleted_docs_size;
        ASSERT_TRUE(segment_reader.ReadDeletedDocsSize(deleted_docs_size).ok());
    }
}
