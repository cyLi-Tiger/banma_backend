/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <names.h>
#include <shared_state.h>

#include <rapids_triton/batch/batch.hpp>        // rapids::Batch
#include <rapids_triton/model/model.hpp>        // rapids::Model
#include <rapids_triton/triton/deployment.hpp>  // rapids::DeploymentType
#include <rapids_triton/triton/device.hpp>      // rapids::device_id_t

#include <km.h>

namespace triton {
namespace backend {
namespace NAMESPACE {

struct RapidsModel : rapids::Model<RapidsSharedState> {
  RapidsModel(std::shared_ptr<RapidsSharedState> shared_state,
              rapids::device_id_t device_id, cudaStream_t default_stream,
              rapids::DeploymentType deployment_type,
              std::string const& filepath)
      : rapids::Model<RapidsSharedState>(shared_state, device_id,
                                         default_stream, deployment_type,
                                         filepath) {}

  void predict(rapids::Batch& batch) const {
    auto matrix = get_input<float>(batch, "matrix");
    auto vec1 = get_output<float>(batch, "vec1");
    auto vec2 = get_output<float>(batch, "vec2");
    auto cost = get_output<float>(batch, "cost");
    
    auto epsilon = get_shared_state()->epsilon_;
    km(vec1.data(), vec2.data(), cost.data(), matrix.data(), epsilon);
    
    vec1.finalize();
    vec2.finalize();
    cost.finalize();
  }

  // TODO(template): Define any of the following only if necessary for your
  // backend
  // void load() {}
  // void unload() {}
  std::optional<rapids::MemoryType> preferred_mem_type(rapids::Batch& batch)
  const {
    return rapids::DeviceMemory;
  }
  // cudaStream_t get_stream() const {}

};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
