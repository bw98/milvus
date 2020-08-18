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
#include <fiu-local.h>
#include <fiu-control.h>

#include "config/Config.h"
#include "segment/SegmentReader.h"
#include "segment/SegmentWriter.h"
#include "easyloggingpp/easylogging++.h"
#include "segment/utils.h"
#include "knowhere/index/vector_index/VecIndex.h"
#include "knowhere/index/vector_index/VecIndexFactory.h"
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/IndexType.h"

INITIALIZE_EASYLOGGINGPP

TEST_F(SegmentTest, SEGMENT_RW_TEST) {
    const std::string segment_dir = "/tmp";

    {
        // SegmentWriter do something here
        milvus::segment::SegmentWriter segment_writer(segment_dir);

        /* test vector index */
        milvus::knowhere::IndexType index_type = milvus::knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;
        milvus::knowhere::IndexMode index_mode = milvus::knowhere::IndexMode::MODE_CPU;
        auto index = milvus::knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type, index_mode);
        ASSERT_TRUE(index != nullptr);
        segment_writer.SetVectorIndex(index);
        const std::string location = "/tmp/test_index";
        ASSERT_TRUE(segment_writer.WriteVectorIndex(location).ok());  // Bug here!!!
    }

    {
        // SegmentReader do something here
        milvus::segment::SegmentReader segment_reader(segment_dir);

        /* test vector index */
        const std::string location = "/tmp/test_index";
        milvus::segment::VectorIndexPtr index = std::make_shared<milvus::segment::VectorIndex>();
        ASSERT_TRUE(segment_reader.LoadVectorIndex(location, index).ok());
    }
}