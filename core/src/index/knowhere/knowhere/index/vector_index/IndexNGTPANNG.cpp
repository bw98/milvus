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

#include "knowhere/index/vector_index/IndexNGTPANNG.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

#include <memory>

namespace milvus {
namespace knowhere {

void
IndexNGTPANNG::BuildAll(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr);

    NGT::Property prop;
    prop.setDefaultForCreateIndex();
    prop.dimension = dim;

    auto edge_size = config[IndexParams::edge_size].get<int64_t>();
    prop.edgeSizeLimitForCreation = edge_size;

    MetricType metric_type = config[Metric::TYPE];

    if (metric_type == Metric::L2) {
        prop.distanceType = NGT::Index::Property::DistanceType::DistanceTypeL2;
    } else if (metric_type == Metric::HAMMING) {
        prop.distanceType = NGT::Index::Property::DistanceType::DistanceTypeHamming;
    } else if (metric_type == Metric::JACCARD) {
        prop.distanceType = NGT::Index::Property::DistanceType::DistanceTypeJaccard;
    } else {
        KNOWHERE_THROW_MSG("Metric type not supported: " + metric_type);
    }

    index_ =
        std::shared_ptr<NGT::Index>(NGT::Index::createGraphAndTree(reinterpret_cast<const float*>(p_data), prop, rows));

    auto forcedly_pruned_edge_size = config[IndexParams::forcedly_pruned_edge_size].get<int64_t>();
    auto selectively_pruned_edge_size = config[IndexParams::selectively_pruned_edge_size].get<int64_t>();

    if (!forcedly_pruned_edge_size && !selectively_pruned_edge_size) {
        return;
    }

    if (forcedly_pruned_edge_size && selectively_pruned_edge_size &&
        selectively_pruned_edge_size >= forcedly_pruned_edge_size) {
        KNOWHERE_THROW_MSG("Selectively pruned edge size should less than remaining edge size");
    }

    // prune
    auto& graph = dynamic_cast<NGT::GraphIndex&>(index_->getIndex());
    for (size_t id = 1; id < graph.repository.size(); id++) {
        try {
            NGT::GraphNode& node = *graph.getNode(id);
            if (node.size() >= forcedly_pruned_edge_size) {
                node.resize(forcedly_pruned_edge_size);
            }
            if (node.size() >= selectively_pruned_edge_size) {
                size_t rank = 0;
                for (auto i = node.begin(); i != node.end(); ++rank) {
                    if (rank >= selectively_pruned_edge_size) {
                        bool found = false;
                        for (size_t t1 = 0; t1 < node.size() && found == false; ++t1) {
                            if (t1 >= selectively_pruned_edge_size) {
                                break;
                            }
                            if (rank == t1) {
                                continue;
                            }
                            NGT::GraphNode& node2 = *graph.getNode(node[t1].id);
                            for (size_t t2 = 0; t2 < node2.size(); ++t2) {
                                if (t2 >= selectively_pruned_edge_size) {
                                    break;
                                }
                                if (node2[t2].id == (*i).id) {
                                    found = true;
                                    break;
                                }
                            }  // for
                        }      // for
                        if (found) {
                            // remove
                            i = node.erase(i);
                            continue;
                        }
                    }
                    i++;
                }  // for
            }
        } catch (NGT::Exception& err) {
            std::cerr << "Graph::search: Warning. Cannot get the node. ID=" << id << ":" << err.what() << std::endl;
            continue;
        }
    }
}

}  // namespace knowhere
}  // namespace milvus
