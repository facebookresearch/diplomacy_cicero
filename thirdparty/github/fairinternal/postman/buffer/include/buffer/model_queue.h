/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// Original implementation from https://github.com/facebookresearch/rela

#pragma once

#include <pybind11/pybind11.h>
#include <map>

namespace py = pybind11;
namespace buffer {

class ModelQueue {
 public:
  ModelQueue()
      : latest_model_(-1) {
  }

  ModelQueue(py::object py_model)
      : latest_model_(0) {
    py_models_[0] = py_model;
    model_call_counts_[0] = 0;
  }

  void update_model(py::object py_model) {
    py_models_[latest_model_ + 1] = py_model;
    model_call_counts_[latest_model_ + 1] = 0;
    ++latest_model_;
  }

  const std::tuple<int, py::object> get_model() {
    std::lock_guard<std::mutex> lk(m_);
    ++model_call_counts_[latest_model_];
    return std::make_tuple(latest_model_, py_models_[latest_model_]);
  }

  void release_model(int id) {
    std::unique_lock<std::mutex> lk(m_);
    --model_call_counts_[id];
    if ((model_call_counts_[id] == 0) && (id != latest_model_)) {
        py_models_.erase(id);
        model_call_counts_.erase(id);
    }
  }

 private:
  std::map<int, py::object> py_models_;
  std::map<int, int> model_call_counts_;
  int latest_model_;

  std::mutex m_;
};

}  // namespace buffer
