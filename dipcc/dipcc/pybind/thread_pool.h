/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <vector>

#include "../cc/checks.h"
#include "../cc/data_fields.h"
#include "../cc/encoding.h"

namespace py = pybind11;

namespace dipcc {

torch::Tensor
py_thread_pool_encode_orders_strict(ThreadPool *thread_pool,
                                    std::vector<std::string> &orders,
                                    int input_version) {
  return thread_pool->encode_orders_strict(orders, input_version);
}

torch::Tensor
py_thread_pool_encode_orders_tolerant(ThreadPool *thread_pool, const Game &game,
                                      std::vector<std::string> &orders,
                                      int input_version) {
  return thread_pool->encode_orders_tolerant(game, orders, input_version);
}

TensorDict py_thread_pool_encode_inputs_state_only_multi(
    ThreadPool *thread_pool, std::vector<Game *> &games, int input_version) {
  return thread_pool->encode_inputs_state_only_multi(games, input_version);
}

TensorDict py_thread_pool_encode_inputs_multi(ThreadPool *thread_pool,
                                              std::vector<Game *> &games,
                                              int input_version) {
  return thread_pool->encode_inputs_multi(games, input_version);
}

TensorDict py_thread_pool_encode_inputs_all_powers_multi(
    ThreadPool *thread_pool, std::vector<Game *> &games, int input_version) {
  return thread_pool->encode_inputs_all_powers_multi(games, input_version);
}

std::vector<std::vector<std::vector<std::string>>>
py_decode_order_idxs(ThreadPool *thread_pool, torch::Tensor *order_idxs) {
  return thread_pool->get_orders_decoder().decode_order_idxs(order_idxs);
}

std::vector<std::vector<std::vector<std::string>>>
py_decode_order_idxs_all_powers(ThreadPool *thread_pool,
                                torch::Tensor *order_idxs,
                                torch::Tensor *x_in_adj_phase,
                                torch::Tensor *x_power,
                                int batch_repeat_interleave) {
  return thread_pool->get_orders_decoder().decode_order_idxs_all_powers(
      order_idxs, x_in_adj_phase, x_power, batch_repeat_interleave);
}

} // namespace dipcc
