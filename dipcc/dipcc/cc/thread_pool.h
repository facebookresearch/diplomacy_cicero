/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "data_fields.h"
#include "game.h"
#include "orders_encoder.h"

namespace py = pybind11;

namespace dipcc {

// Job Types
enum ThreadPoolJobType { STEP, ENCODE, ENCODE_STATE_ONLY, ENCODE_ALL_POWERS };

// Used for ENCODE* jobs
//
// Initialized in "new_data_fields" function
struct EncodingArrayPointers {
  float *x_board_state;
  float *x_prev_state;
  long *x_prev_orders;
  float *x_season;
  float *x_year_encoded;
  float *x_in_adj_phase;
  float *x_build_numbers;
  float *x_scoring_system;
  int8_t *x_loc_idxs;
  int32_t *x_possible_actions;
  int64_t *x_power;
};

// Struct for all job types
struct ThreadPoolJob {
  ThreadPoolJobType job_type;
  int input_version;
  std::vector<Game *> games;
  std::vector<EncodingArrayPointers> encoding_array_pointers;

  ThreadPoolJob() {}
  ThreadPoolJob(ThreadPoolJobType type, int iv)
      : job_type(type), input_version(iv) {}
};

class ThreadPool {
public:
  ThreadPool(size_t n_threads,
             std::unordered_map<std::string, int> order_vocabulary_to_idx,
             int max_order_cands);
  ~ThreadPool();

  const OrdersDecoder &get_orders_decoder() const { return orders_decoder_; }

  // Call game.process() on each of the games. Blocks until all process()
  // functions have exited.
  void process_multi(std::vector<Game *> &games);

  // Write a single sequence of orders as a feature tensor. The same format as
  // used from x_prev_orders features.
  // Any orders that fail to strictly match the exact strings in the
  // order_vocabulary may SILENTLY be ignored!
  torch::Tensor encode_orders_strict(std::vector<std::string> &orders,
                                     int input_version);

  // Same as encode_orders_strict, however will tolerate certain differences in
  // whether supportees are coast-qualified or not, using the Game object to
  // disambiguate. However, other invalidly formatted orders besides that may
  // still be SILENTLY ignored!
  torch::Tensor encode_orders_tolerant(const Game &game,
                                       std::vector<std::string> &orders,
                                       int input_version);

  // Fill a list of pre-allocated DataFields objects with the games' input
  // encodings
  TensorDict encode_inputs_multi(std::vector<Game *> &games, int input_version);

  // Fill a list of pre-allocated DataFields objects with the games' input
  // encodings
  TensorDict encode_inputs_state_only_multi(std::vector<Game *> &games,
                                            int input_version);

  // Fill a list of pre-allocated DataFields objects with the games' input
  // encodings
  TensorDict encode_inputs_all_powers_multi(std::vector<Game *> &games,
                                            int input_version);

private:
  /////////////
  // Methods //
  /////////////

  // Worker thread entrypoint function
  void thread_fn();

  // Top-level job handler
  void thread_fn_do_job_unsafe(ThreadPoolJob &);

  // Job handler methods
  void do_job_step(ThreadPoolJob &);
  void do_job_encode(ThreadPoolJob &);
  void do_job_encode_state_only(ThreadPoolJob &);
  void do_job_encode_all_powers(ThreadPoolJob &);

  // Job handler boilerplate
  void boilerplate_job_prep(ThreadPoolJobType, std::vector<Game *> &,
                            int input_version);
  void boilerplate_job_handle(std::unique_lock<std::mutex> &);

  // Helpers
  void encode_state_for_game(Game *, int input_version,
                             EncodingArrayPointers &);

  const OrdersEncoder &get_orders_encoder(int input_version);

  //////////
  // Data //
  //////////

  std::vector<ThreadPoolJob> jobs_;
  std::mutex mutex_;
  std::condition_variable cv_in_;
  std::condition_variable cv_out_;
  size_t unfinished_jobs_;
  bool time_to_die_ = false;

  std::vector<std::thread> threads_;
  const OrdersEncoder orders_encoder_nonbuggy_;
  const OrdersEncoder orders_encoder_buggy_;
  const OrdersDecoder orders_decoder_;
};

} // namespace dipcc
