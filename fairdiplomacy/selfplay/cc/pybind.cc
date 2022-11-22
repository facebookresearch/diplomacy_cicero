/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// Original implementation from https://github.com/facebookresearch/rela

#include "prioritized_replay.h"
#include <nest.h>
#include <nest_pybind.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
namespace py = pybind11;
using namespace buffer;

PYBIND11_MODULE(rela, m) {
  py::class_<std::future<void>>(m, "Future")
      .def(py::init<>())
      .def("get", &std::future<void>::get);

  py::class_<NestPrioritizedReplay, std::shared_ptr<NestPrioritizedReplay>>(
      m, "NestPrioritizedReplay")
      .def(py::init<int, int, float, float, bool, bool>(), py::arg("capacity"),
           py::arg("seed"), py::arg("alpha"), py::arg("beta"),
           py::arg("prefetch"), py::arg("shuffle") = false)
      .def("load", &NestPrioritizedReplay::load)
      .def("save", &NestPrioritizedReplay::save,
           py::call_guard<py::gil_scoped_release>())
      .def("size", &NestPrioritizedReplay::size)
      .def("num_add", &NestPrioritizedReplay::num_add)
      .def("total_bytes", &NestPrioritizedReplay::total_bytes)
      .def("total_numel", &NestPrioritizedReplay::total_numel)
      .def("add_one", &NestPrioritizedReplay::add_one)
      .def("get_new_content", &NestPrioritizedReplay::get_new_content)
      .def("get_all_content", &NestPrioritizedReplay::get_all_content)
      .def("add_batch", &NestPrioritizedReplay::add_batch,
           py::call_guard<py::gil_scoped_release>())
      .def("add_batch_async", &NestPrioritizedReplay::add_batch_async,
           py::call_guard<py::gil_scoped_release>())
      .def("sample", &NestPrioritizedReplay::sample)
      .def("update_priority", &NestPrioritizedReplay::update_priority)
      .def("keep_priority", &NestPrioritizedReplay::keep_priority);
}
