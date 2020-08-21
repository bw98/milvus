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
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/IndexIVFSQ.h"
#include "knowhere/index/vector_index/IndexType.h"
#include "index/unittest/Helper.h"
#include "index/unittest/utils.h"

//INITIALIZE_EASYLOGGINGPP

class SegmentIndexTest : public DataGen, public SegmentTest {
 protected:
    void
    SetUp() override {
        SegmentTest::SetUp();
        Generate(DIM, NB, NQ);
    }

    void
    TearDown() override {
        SegmentTest::TearDown();
    }

 protected:
    milvus::knowhere::IndexType index_type_;
    milvus::knowhere::IndexMode index_mode_;
    milvus::knowhere::IVFPtr index_ = nullptr;
};

TEST_F(SegmentIndexTest, SEGMENT_INDEX_RW_TEST) {
    const std::string segment_dir = "/tmp";

    {
        // SegmentWriter do something here
        milvus::segment::SegmentWriter segment_writer(segment_dir);

        /* test to write vector index */
        index_type_ = milvus::knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;
        index_mode_ = milvus::knowhere::IndexMode::MODE_CPU;
        //index_ = milvus::knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_, index_mode_);
        index_ = IndexFactory(index_type_, index_mode_);
        ASSERT_TRUE(index_ != nullptr);
        auto conf = ParamGenerator::GetInstance().Gen(index_type_);
        index_->Train(base_dataset, conf);
        index_->Add(base_dataset, conf);
        EXPECT_EQ(index_->Count(), nb);
        EXPECT_EQ(index_->Dim(), dim);

        segment_writer.SetVectorIndex(index_);
        const std::string location = "/tmp/test_index";
        ASSERT_TRUE(segment_writer.WriteVectorIndex(location).ok());
    }

    {
        // SegmentReader do something here
        milvus::segment::SegmentReader segment_reader(segment_dir);

        /* test to read vector index */
        const std::string location = "/tmp/test_index";
        milvus::segment::VectorIndexPtr index_ = std::make_shared<milvus::segment::VectorIndex>();
        ASSERT_TRUE(segment_reader.LoadVectorIndex(location, index_).ok());
    }
}
