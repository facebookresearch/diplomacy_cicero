/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "thread_pool.h"
#include "checks.h"
#include "data_fields.h"
#include "encoding.h"

using namespace std;

namespace dipcc {

ThreadPool::ThreadPool(
    size_t n_threads,
    std::unordered_map<std::string, int> order_vocabulary_to_idx,
    int max_order_cands)
    : orders_encoder_nonbuggy_(order_vocabulary_to_idx, max_order_cands, false),
      orders_encoder_buggy_(order_vocabulary_to_idx, max_order_cands, true),
      orders_decoder_(order_vocabulary_to_idx) {

  jobs_.reserve(n_threads);
  threads_.reserve(n_threads);
  for (int i = 0; i < n_threads; ++i) {
    threads_.push_back(thread(&ThreadPool::thread_fn, this));
  }
}

ThreadPool::~ThreadPool() {
  { // Locked critical section
    unique_lock<mutex> my_lock(mutex_);
    time_to_die_ = true;
  }
  cv_in_.notify_all();
  for (auto &th : threads_) {
    th.join();
  }
}

void ThreadPool::boilerplate_job_prep(ThreadPoolJobType job_type,
                                      vector<Game *> &games,
                                      int input_version) {
  JCHECK(jobs_.size() == 0, "ThreadPool called with non-empty jobs_");

  // Pack games into n_threads jobs
  size_t n_threads = threads_.size() > 0 ? threads_.size() : 1;
  for (int i = 0; i < n_threads; ++i) {
    jobs_.push_back(ThreadPoolJob(job_type, input_version));
  }
  for (int i = 0; i < games.size(); ++i) {
    jobs_[i % n_threads].games.push_back(games[i]);
  }
  for (int i = 0; i < games.size(); ++i) {
    // Poor man's race condition elimination. Should not take so much time as
    // stepping job calls get_all_possible_orders on all produced states.
    games[i]->get_all_possible_orders();
  }
}

void ThreadPool::boilerplate_job_handle(unique_lock<mutex> &my_lock) {
  // maybe handle in-thread
  if (threads_.size() == 0) {
    thread_fn_do_job_unsafe(jobs_[0]);
    jobs_.clear();
    return;
  }

  // Notify and wait for worker threads
  unfinished_jobs_ = jobs_.size();
  cv_in_.notify_all();
  while (unfinished_jobs_ != 0) {
    cv_out_.wait(my_lock);
  }
}

void ThreadPool::process_multi(vector<Game *> &games) {
  unique_lock<mutex> my_lock(mutex_);

  int input_version = MAX_INPUT_VERSION; // unused, dummy value

  boilerplate_job_prep(ThreadPoolJobType::STEP, games, input_version);
  boilerplate_job_handle(my_lock);
}

torch::Tensor ThreadPool::encode_orders_tolerant(const Game &game,
                                                 vector<std::string> &orders,
                                                 int input_version) {
  auto tensor = torch::empty({2, 100}, torch::kLong);
  const OrdersEncoder &orders_encoder_ = get_orders_encoder(input_version);
  auto tensor_ptr = tensor.data_ptr<long>();
  std::vector<Order> orders_parsed;
  std::vector<const GameState *> state_for_each_order;
  for (const auto &order_str : orders) {
    orders_parsed.emplace_back(order_str);
    state_for_each_order.push_back(&(game.get_state()));
  }
  orders_encoder_.encode_orders_deepmind(orders_parsed, state_for_each_order,
                                         tensor_ptr);
  return tensor;
}

torch::Tensor ThreadPool::encode_orders_strict(vector<std::string> &orders,
                                               int input_version) {
  auto tensor = torch::empty({2, 100}, torch::kLong);
  const OrdersEncoder &orders_encoder_ = get_orders_encoder(input_version);
  auto tensor_ptr = tensor.data_ptr<long>();
  std::vector<Order> orders_parsed;
  for (const auto &order_str : orders)
    orders_parsed.emplace_back(order_str);
  std::vector<const GameState *> state_for_each_order(orders_parsed.size(),
                                                      nullptr);
  orders_encoder_.encode_orders_deepmind(orders_parsed, state_for_each_order,
                                         tensor_ptr);
  return tensor;
}

TensorDict ThreadPool::encode_inputs_state_only_multi(vector<Game *> &games,
                                                      int input_version) {
  unique_lock<mutex> my_lock(mutex_);

  boilerplate_job_prep(ThreadPoolJobType::ENCODE_STATE_ONLY, games,
                       input_version);

  // Job-specific prep
  TensorDict fields(new_data_fields_state_only(games.size(), input_version));
  size_t n_threads = threads_.size() > 0 ? threads_.size() : 1;
  for (int i = 0; i < games.size(); ++i) {
    jobs_[i % n_threads].encoding_array_pointers.push_back(
        EncodingArrayPointers{
            fields["x_board_state"].index({i}).data_ptr<float>(),
            fields["x_prev_state"].index({i}).data_ptr<float>(),
            fields["x_prev_orders"].index({i}).data_ptr<long>(),
            fields["x_season"].index({i}).data_ptr<float>(),
            fields["x_year_encoded"].index({i}).data_ptr<float>(),
            fields["x_in_adj_phase"].index({i}).data_ptr<float>(),
            fields["x_build_numbers"].index({i}).data_ptr<float>(),
            fields["x_scoring_system"].index({i}).data_ptr<float>(),
            nullptr, // x_loc_idxs
            nullptr, // x_possible_actions
            nullptr, // x_max_seq_len
        });
  }

  boilerplate_job_handle(my_lock);

  return fields;
}

TensorDict ThreadPool::encode_inputs_all_powers_multi(vector<Game *> &games,
                                                      int input_version) {
  unique_lock<mutex> my_lock(mutex_);

  boilerplate_job_prep(ThreadPoolJobType::ENCODE_ALL_POWERS, games,
                       input_version);

  // Job-specific prep
  TensorDict fields(new_data_fields(games.size(), input_version, N_SCS, true));
  size_t n_threads = threads_.size() > 0 ? threads_.size() : 1;
  for (int i = 0; i < games.size(); ++i) {
    jobs_[i % n_threads].encoding_array_pointers.push_back(
        EncodingArrayPointers{
            fields["x_board_state"].index({i}).data_ptr<float>(),
            fields["x_prev_state"].index({i}).data_ptr<float>(),
            fields["x_prev_orders"].index({i}).data_ptr<long>(),
            fields["x_season"].index({i}).data_ptr<float>(),
            fields["x_year_encoded"].index({i}).data_ptr<float>(),
            fields["x_in_adj_phase"].index({i}).data_ptr<float>(),
            fields["x_build_numbers"].index({i}).data_ptr<float>(),
            fields["x_scoring_system"].index({i}).data_ptr<float>(),
            fields["x_loc_idxs"].index({i}).data_ptr<int8_t>(),
            fields["x_possible_actions"].index({i}).data_ptr<int32_t>(),
            fields["x_power"].index({i}).data_ptr<int64_t>(),
        });
  }

  boilerplate_job_handle(my_lock);

  return fields;
}

TensorDict ThreadPool::encode_inputs_multi(vector<Game *> &games,
                                           int input_version) {
  unique_lock<mutex> my_lock(mutex_);

  boilerplate_job_prep(ThreadPoolJobType::ENCODE, games, input_version);

  // Job-specific prep
  TensorDict fields(new_data_fields(games.size(), input_version));
  size_t n_threads = threads_.size() > 0 ? threads_.size() : 1;
  for (int i = 0; i < games.size(); ++i) {
    jobs_[i % n_threads].encoding_array_pointers.push_back(
        EncodingArrayPointers{
            fields["x_board_state"].index({i}).data_ptr<float>(),
            fields["x_prev_state"].index({i}).data_ptr<float>(),
            fields["x_prev_orders"].index({i}).data_ptr<long>(),
            fields["x_season"].index({i}).data_ptr<float>(),
            fields["x_year_encoded"].index({i}).data_ptr<float>(),
            fields["x_in_adj_phase"].index({i}).data_ptr<float>(),
            fields["x_build_numbers"].index({i}).data_ptr<float>(),
            fields["x_scoring_system"].index({i}).data_ptr<float>(),
            fields["x_loc_idxs"].index({i}).data_ptr<int8_t>(),
            fields["x_possible_actions"].index({i}).data_ptr<int32_t>(),
            nullptr, // x_max_seq_len
        });
  }

  boilerplate_job_handle(my_lock);

  return fields;
}

void ThreadPool::thread_fn() {
  while (true) {
    ThreadPoolJob job;
    { // Locked critical section
      unique_lock<mutex> my_lock(mutex_);
      while (!time_to_die_ && jobs_.size() == 0) {
        cv_in_.wait(my_lock);
      }
      if (time_to_die_) {
        return;
      }
      job = jobs_.back();
      jobs_.pop_back();
    }

    // Do the job
    thread_fn_do_job_unsafe(job);

    // Notify done (locked critical section)
    {
      unique_lock<mutex> my_lock(mutex_);
      unfinished_jobs_--;
      if (unfinished_jobs_ == 0) {
        cv_out_.notify_all();
      }
    }
  }
}

void ThreadPool::thread_fn_do_job_unsafe(ThreadPoolJob &job) {
  try {
    // Do the job
    if (job.job_type == ThreadPoolJobType::STEP) {
      do_job_step(job);
    } else if (job.job_type == ThreadPoolJobType::ENCODE) {
      do_job_encode(job);
    } else if (job.job_type == ThreadPoolJobType::ENCODE_STATE_ONLY) {
      do_job_encode_state_only(job);
    } else if (job.job_type == ThreadPoolJobType::ENCODE_ALL_POWERS) {
      do_job_encode_all_powers(job);
    } else {
      JCHECK(false, "ThreadPoolJobType Not Implemented");
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Worker thread exception: " << e.what();
    throw e;
  }
}

void ThreadPool::do_job_step(ThreadPoolJob &job) {
  for (Game *game : job.games) {
    game->process();
    game->get_all_possible_orders();
  }
}

void ThreadPool::do_job_encode_state_only(ThreadPoolJob &job) {
  JCHECK(job.job_type == ThreadPoolJobType::ENCODE_STATE_ONLY,
         "do_job_encode called with wrong ThreadPoolJobType");
  JCHECK(job.games.size() == job.encoding_array_pointers.size(),
         "do_job_encode called with wrong input sizes");

  for (int i = 0; i < job.games.size(); ++i) {
    Game *game = job.games[i];
    int input_version = job.input_version;
    EncodingArrayPointers &pointers = job.encoding_array_pointers[i];
    encode_state_for_game(game, input_version, pointers);
  }
}

void ThreadPool::do_job_encode_all_powers(ThreadPoolJob &job) {
  JCHECK(job.job_type == ThreadPoolJobType::ENCODE_ALL_POWERS,
         "do_job_encode called with wrong ThreadPoolJobType");
  JCHECK(job.games.size() == job.encoding_array_pointers.size(),
         "do_job_encode called with wrong input sizes");

  for (int i = 0; i < job.games.size(); ++i) {
    Game *game = job.games[i];
    int input_version = job.input_version;
    EncodingArrayPointers &pointers = job.encoding_array_pointers[i];

    encode_state_for_game(game, input_version, pointers);
    const OrdersEncoder &orders_encoder_ = get_orders_encoder(input_version);
    orders_encoder_.encode_valid_orders_all_powers(
        game->get_state(), pointers.x_possible_actions, pointers.x_loc_idxs,
        pointers.x_power);
  }
}

void ThreadPool::do_job_encode(ThreadPoolJob &job) {
  JCHECK(job.job_type == ThreadPoolJobType::ENCODE,
         "do_job_encode called with wrong ThreadPoolJobType");
  JCHECK(job.games.size() == job.encoding_array_pointers.size(),
         "do_job_encode called with wrong input sizes");

  for (int i = 0; i < job.games.size(); ++i) {
    Game *game = job.games[i];
    int input_version = job.input_version;
    EncodingArrayPointers &pointers = job.encoding_array_pointers[i];

    // encode all inputs except actions
    encode_state_for_game(game, input_version, pointers);

    // encode x_possible_actions, x_loc_idxs
    const OrdersEncoder &orders_encoder_ = get_orders_encoder(input_version);
    for (int power_i = 0; power_i < 7; ++power_i) {
      orders_encoder_.encode_valid_orders(
          POWERS[power_i], game->get_state(),
          pointers.x_possible_actions + (power_i * orders_encoder_.MAX_SEQ_LEN *
                                         orders_encoder_.get_max_cands()),
          pointers.x_loc_idxs + (power_i * 81));
    }
  }
}

const OrdersEncoder &ThreadPool::get_orders_encoder(int input_version) {
  static_assert(MAX_INPUT_VERSION <= 3,
                "Don't forget to update code here if necessary when changing "
                "MAX_INPUT_VERSION");
  if (input_version >= 3) {
    return orders_encoder_nonbuggy_;
  }
  return orders_encoder_buggy_;
}

void ThreadPool::encode_state_for_game(Game *game, int input_version,
                                       EncodingArrayPointers &pointers) {
  // encode x_board_state
  encode_board_state(game->get_state(), input_version, pointers.x_board_state);

  // encode x_prev_state, x_prev_orders
  GameState *prev_move_state = game->get_last_movement_phase();
  if (prev_move_state != nullptr) {
    encode_board_state(*prev_move_state, input_version, pointers.x_prev_state);
    const OrdersEncoder &orders_encoder_ = get_orders_encoder(input_version);
    orders_encoder_.encode_prev_orders_deepmind(game, pointers.x_prev_orders);
  } else {
    memset(pointers.x_prev_state, 0,
           81 * board_state_enc_width(input_version) * sizeof(float));
    memset(pointers.x_prev_orders, 0, 2 * PREV_ORDERS_CAPACITY * sizeof(long));
  }

  // encode x_season
  Phase current_phase = game->get_state().get_phase();
  memset(pointers.x_season, 0, 3 * sizeof(float));
  if (current_phase.season == 'S') {
    pointers.x_season[0] = 1;
  } else if (current_phase.season == 'F') {
    pointers.x_season[1] = 1;
  } else {
    pointers.x_season[2] = 1;
  }

  // encode x_year
  // Encoded as number of years after 1901, divided by 10.
  // Clamp after 1951 just in case to limit overly large values
  memset(pointers.x_year_encoded, 0, 1 * sizeof(float));
  pointers.x_year_encoded[0] =
      std::clamp(0.1 * (current_phase.year - 1901), 0.0, 5.0);

  // encode x_in_adj_phase, x_build_numbers
  if (current_phase.phase_type == 'A') {
    *pointers.x_in_adj_phase = 1;
    float *p = pointers.x_build_numbers;

    for (int i = 0; i < 7; ++i) {
      Power power = POWERS[i];
      int power_i = static_cast<int>(power) - 1;
      p[power_i] = game->get_state().get_n_builds(power);
    }
  } else {
    *pointers.x_in_adj_phase = 0;
    memset(pointers.x_build_numbers, 0, 7 * sizeof(float));
  }

  memset(pointers.x_scoring_system, 0, NUM_SCORING_SYSTEMS * sizeof(float));
  pointers.x_scoring_system[static_cast<int>(game->get_scoring_system())] =
      (float)1.0;
}

} // namespace dipcc
