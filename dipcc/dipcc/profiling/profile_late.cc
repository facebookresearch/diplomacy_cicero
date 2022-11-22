/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <chrono>
#include <fstream>
#include <glog/logging.h>
#include <random>
#include <streambuf>
#include <string>
#include <vector>

#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/encoding.h"
#include "../cc/order.h"
#include "../cc/power.h"
#include "../cc/thread_pool.h"

using namespace dipcc;

using OrderSet = std::unordered_map<std::string, std::vector<std::string>>;

const int kOrderVocabIdxs = 469;

const std::string kGamePath =
    "data/dipcc_profiling_game.json";

const std::string kOrderVocabPath =
    "data/order_vocab.txt";

const std::string kPhase = "S1911M";

const std::vector<std::pair<std::string, std::vector<std::vector<std::string>>>>
    kTopOrders = {
        {"AUSTRIA",
         {{"A BOH S A MUN", "A TYR S A VEN - PIE", "A UKR - WAR",
           "F ION S F NAP", "A TRI - VEN", "A APU S A TRI - VEN"},
          {"A BOH S A MUN", "A TYR S A MUN", "A UKR - SEV", "F ION S F NAP",
           "A TRI S A VEN", "A APU S F ROM"},
          {"A BOH S A MUN", "A TYR S A VEN - PIE", "A UKR - WAR",
           "F ION S F NAP", "A TRI - VEN", "A APU S F NAP"},
          {"A BOH S A MUN", "A TYR S A MUN", "A UKR - SEV", "F ION S F NAP",
           "A TRI S A VEN", "A APU S A VEN"},
          {"A BOH S A MUN", "A TYR - PIE", "A UKR - WAR", "F ION S F NAP",
           "A TRI - TYR", "A APU S F ROM"},
          {"A BOH S A MUN", "A TYR S A MUN", "A UKR - WAR", "F ION S F NAP",
           "A TRI S A VEN", "A APU S F ROM"},
          {"A BOH S A MUN", "A TYR - PIE", "A UKR - SEV", "F ION S F NAP",
           "A TRI - TYR", "A APU S F ROM"},
          {"A BOH S A MUN", "A TYR - PIE", "A UKR - MOS", "F ION S F NAP",
           "A TRI - TYR", "A APU S F ROM"}}},
        {"FRANCE",
         {{"F NTH - DEN", "F ENG - NTH", "A DEN - SWE", "F NWY - STP/NC",
           "A BUR - MUN", "A RUH S A KIE", "A KIE S A BUR - MUN", "F SWE - BAL",
           "F WES S F TYS", "A MAR - ROM VIA", "F BOT S F SWE - BAL",
           "F TUN - ION", "F LYO C A MAR - ROM", "F TYS C A MAR - ROM",
           "A TUS S A MAR - ROM"},
          {"F NTH - DEN", "F ENG - NTH", "A DEN - SWE", "F NWY - STP/NC",
           "A BUR - MUN", "A RUH S A KIE", "A KIE S A BUR - MUN", "F SWE - BAL",
           "F WES S F TYS", "A MAR - PIE", "F BOT S F SWE - BAL", "F TUN - ION",
           "F LYO S A MAR - PIE", "F TYS S F TUN - ION", "A TUS S A MAR - PIE"},
          {"F NTH - NWY", "F ENG - NTH", "A DEN S A KIE", "F NWY - BAR",
           "A BUR - MUN", "A RUH S A KIE", "A KIE S A BUR - MUN", "F SWE - BAL",
           "F WES S F TYS", "A MAR - PIE", "F BOT S F SWE - BAL", "F TUN - ION",
           "F LYO S A MAR - PIE", "F TYS S A TUS - ROM", "A TUS - ROM"},
          {"F NTH - NWY", "F ENG - NTH", "A DEN S A KIE", "F NWY - BAR",
           "A BUR - MUN", "A RUH S A KIE", "A KIE S A BUR - MUN", "F SWE - BAL",
           "F WES S F TYS", "A MAR - ROM VIA", "F BOT S F SWE - BAL",
           "F TUN - ION", "F LYO C A MAR - ROM", "F TYS C A MAR - ROM",
           "A TUS S A MAR - ROM"},
          {"F NTH - NWY", "F ENG - NTH", "A DEN S A KIE", "F NWY - BAR",
           "A BUR - MUN", "A RUH S A KIE", "A KIE S A BUR - MUN", "F SWE - BAL",
           "F WES S F TYS", "A MAR - PIE", "F BOT S F SWE - BAL", "F TUN - ION",
           "F LYO S A MAR - PIE", "F TYS S F TUN - ION", "A TUS S A MAR - PIE"},
          {"F NTH - NWY", "F ENG - NTH", "A DEN S A KIE", "F NWY - BAR",
           "A BUR - MUN", "A RUH S A KIE", "A KIE S A BUR - MUN", "F SWE - BAL",
           "F WES S F TYS", "A MAR - PIE", "F BOT S F SWE - BAL", "F TUN - ION",
           "F LYO S A MAR - PIE", "F TYS S A TUS", "A TUS S A MAR - PIE"},
          {"F NTH - NWY", "F ENG - NTH", "A DEN S A KIE", "F NWY - BAR",
           "A BUR - MUN", "A RUH S A KIE", "A KIE S A BUR - MUN", "F SWE - BAL",
           "F WES S F TYS", "A MAR - PIE", "F BOT S F SWE - BAL",
           "F TUN S F TYS", "F LYO S A MAR - PIE", "F TYS S A TUS",
           "A TUS S A MAR - PIE"},
          {"F NTH - DEN", "F ENG - NTH", "A DEN - SWE", "F NWY - STP/NC",
           "A BUR - MUN", "A RUH S A KIE", "A KIE S A BUR - MUN", "F SWE - BAL",
           "F WES S F TYS", "A MAR - PIE", "F BOT S F SWE - BAL", "F TUN - ION",
           "F LYO S A MAR - PIE", "F TYS S A TUS", "A TUS S A MAR - PIE"}}},
        {"ITALY",
         {{"F NAP S F ROM", "F ROM S F NAP", "A VEN H"},
          {"F NAP S F ROM", "F ROM S F NAP", "A VEN S A TRI"},
          {"F NAP S F ROM", "F ROM S F NAP", "A VEN S A APU"},
          {"F NAP S F ROM", "F ROM S F NAP", "A VEN S F ROM"},
          {"F NAP S F ROM", "F ROM S F NAP", "A VEN S A TUS"},
          {"F NAP S F ROM", "F ROM S F NAP", "A VEN - PIE"},
          {"F NAP S F AEG - ION", "F ROM S F NAP", "A VEN S F ROM"},
          {"F NAP S F ION", "F ROM S F NAP", "A VEN S F ROM"}}},
        {"RUSSIA",
         {{"F BAL - BOT", "A STP H", "A MUN H", "A BER S A MUN",
           "A MOS S A STP"},
          {"F BAL - BER", "A STP H", "A MUN S F BAL - BER", "A BER S A MUN",
           "A MOS S A STP"},
          {"F BAL S A BER", "A STP H", "A MUN H", "A BER S A MUN",
           "A MOS S A STP"},
          {"F BAL H", "A STP H", "A MUN H", "A BER H", "A MOS H"},
          {"F BAL - KIE", "A STP H", "A MUN S F BAL - KIE",
           "A BER S F BAL - KIE", "A MOS S A STP"},
          {"F BAL S A BER", "A STP H", "A MUN S A BER", "A BER S A MUN",
           "A MOS S A STP"},
          {"F BAL - BOT", "A STP H", "A MUN S A BER", "A BER S A MUN",
           "A MOS S A STP"},
          {"F BAL - KIE", "A STP - NWY", "A MUN S F BAL - KIE",
           "A BER S F BAL - KIE", "A MOS - STP"}}},
        {"TURKEY",
         {{"A SIL S A MUN", "F BLA C A CON - SEV", "F AEG S F ION",
           "F BUL/EC - BLA", "A CON - SEV VIA"},
          {"A SIL S A MUN", "F BLA C A CON - SEV", "F AEG S F ION",
           "F BUL/EC - RUM", "A CON - SEV VIA"},
          {"A SIL S A MUN", "F BLA C A CON - SEV", "F AEG S F ION",
           "F BUL/EC S F BLA", "A CON - SEV VIA"},
          {"A SIL S A MUN", "F BLA C A CON - SEV", "F AEG S F ION",
           "F BUL/EC - CON", "A CON - SEV VIA"},
          {"A SIL S A TYR - MUN", "F BLA C A CON - SEV", "F AEG S F ION",
           "F BUL/EC - CON", "A CON - SEV VIA"},
          {"A SIL S A BER", "F BLA C A CON - SEV", "F AEG S F ION",
           "F BUL/EC H", "A CON - SEV VIA"},
          {"A SIL S A BER", "F BLA C A CON - SEV", "F AEG S F ION",
           "F BUL/EC - CON", "A CON - SEV VIA"},
          {"A SIL S A MUN", "F BLA C A CON - SEV", "F AEG S F ION",
           "F BUL/EC H", "A CON - SEV VIA"}}}};

std::unordered_map<std::string, int> get_order_vocab() {
  std::ifstream f(kOrderVocabPath);
  std::string order;
  int id;

  std::unordered_map<std::string, int> vocab;
  for (;;) {
    f >> order;
    if (f.eof())
      break;
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] == '_') {
        order[i] = ' ';
      }
    }
    f >> id;
    vocab[order] = id;
  }
  return vocab;
}

int main(int argc, char *argv[]) {
  int batch_size = 100;
  int num_rounds = 100;
  int pool_size = 10;
  {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--batch_size") {
        assert(i + 1 < argc);
        batch_size = std::stoi(argv[++i]);
      } else if (arg == "--num_rounds") {
        assert(i + 1 < argc);
        num_rounds = std::stoi(argv[++i]);
      } else if (arg == "--pool_size") {
        assert(i + 1 < argc);
        pool_size = std::stoi(argv[++i]);
      } else {
        std::cerr << "Unknown flag: " << arg << "\n";
        return -1;
      }
    }
  }

  // order_batches[i] contains batch_size orders for i-th run.
  std::vector<std::vector<OrderSet>> order_batches(num_rounds);
  std::default_random_engine rng(0);
  for (auto &batch : order_batches) {
    for (int i = 0; i < batch_size; ++i) {
      OrderSet orders;
      for (const auto & [ power, order_options ] : kTopOrders) {
        std::uniform_int_distribution<int> distribution(
            0, order_options.size() - 1);
        const std::vector<std::string> &order_strings =
            order_options[distribution(rng)];
        for (const auto &s : order_strings) {
          orders[power].push_back(s);
        }
      }
      batch.push_back(orders);
    }
  }

  std::ifstream t(kGamePath);
  std::string game_json_str((std::istreambuf_iterator<char>(t)),
                            std::istreambuf_iterator<char>());
  Game init_game(game_json_str);
  init_game = init_game.rolled_back_to_phase_start(kPhase);
  init_game.get_all_possible_orders();

  int input_version = dipcc::MAX_INPUT_VERSION;
  ThreadPool pool(pool_size, get_order_vocab(), kOrderVocabIdxs);

  struct T {
    std::chrono::time_point<std::chrono::steady_clock> start =
        std::chrono::steady_clock::now();

    double tick() {
      const auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = end - start;
      start = end;
      return diff.count();
    }

  } timer;

  double clone_time = 0, encode_time = 0, encode_state_only_time = 0,
         set_order_time = 0, process_time = 0;
  for (int i = 0; i < num_rounds; ++i) {
    // Clone
    std::vector<std::shared_ptr<Game>> games;
    std::vector<Game *> game_ptrs; // That's what the pool expects.
    for (int game_id = 0; game_id < batch_size; ++game_id) {
      games.push_back(std::make_shared<Game>(init_game));
      game_ptrs.emplace_back(games.back().get());
    }
    clone_time += timer.tick();

    // Encode
    pool.encode_inputs_multi(game_ptrs, input_version);
    encode_time += timer.tick();

    // Encode State Only
    pool.encode_inputs_state_only_multi(game_ptrs, input_version);
    encode_state_only_time += timer.tick();

    // Set Orders
    for (int game_id = 0; game_id < batch_size; ++game_id) {
      for (const auto & [ power, orders ] : order_batches[i][game_id]) {
        games[game_id]->set_orders(power, orders);
      }
    }
    set_order_time += timer.tick();

    // Process
    pool.process_multi(game_ptrs);
    process_time += timer.tick();
  }

  LOG(ERROR) << "clone_time: " << clone_time << " / " << num_rounds << " = "
             << clone_time / num_rounds << " s";

  LOG(ERROR) << "encode_time: " << encode_time << " / " << num_rounds << " = "
             << encode_time / num_rounds << " s";

  LOG(ERROR) << "encode_state_only_time: " << encode_state_only_time << " / "
             << num_rounds << " = " << encode_state_only_time / num_rounds
             << " s";

  LOG(ERROR) << "set_order_time: " << set_order_time << " / " << num_rounds
             << " = " << set_order_time / num_rounds << " s";

  LOG(ERROR) << "process_time: " << process_time << " / " << num_rounds << " = "
             << process_time / num_rounds << " s";

  return 0;
}
