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

#include <memory>
#include <rapids_triton/model/shared_state.hpp>

namespace triton {
namespace backend {
namespace NAMESPACE {

struct RapidsSharedState : rapids::SharedModelState {
  RapidsSharedState(std::unique_ptr<common::TritonJson::Value>&& config)
      : rapids::SharedModelState{std::move(config)} {}
  // TODO(template): Define the following only if necessary for your backend
  void load() {
    epsilon_ = get_config_param<float>("epsilon", "1e-4");
  }
  void unload() {} 

  float epsilon_ = 1.0f;
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
