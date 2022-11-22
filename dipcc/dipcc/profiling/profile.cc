/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <chrono>
#include <glog/logging.h>

#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/order.h"
#include "../cc/power.h"

using namespace dipcc;

int main() {
  int N = 100;

  Game game;
  GameState state(game.get_state());

  std::unordered_map<Power, std::vector<Order>> orders;
  orders[Power::AUSTRIA].push_back(Order("A BUD - SER"));
  orders[Power::AUSTRIA].push_back(Order("F TRI - ALB"));
  orders[Power::AUSTRIA].push_back(Order("A VIE - GAL"));
  orders[Power::ENGLAND].push_back(Order("A LVP - EDI"));
  orders[Power::ENGLAND].push_back(Order("F EDI - NWG"));
  orders[Power::ENGLAND].push_back(Order("F LON - NTH"));
  orders[Power::FRANCE].push_back(Order("F BRE - MAO"));
  orders[Power::FRANCE].push_back(Order("A PAR - PIC"));
  orders[Power::FRANCE].push_back(Order("A MAR - BUR"));
  orders[Power::GERMANY].push_back(Order("F KIE - DEN"));
  orders[Power::GERMANY].push_back(Order("A MUN - RUH"));
  orders[Power::GERMANY].push_back(Order("A BER - KIE"));
  orders[Power::ITALY].push_back(Order("F NAP - ION"));
  orders[Power::ITALY].push_back(Order("A ROM - APU"));
  orders[Power::RUSSIA].push_back(Order("F STP/SC - BOT"));
  orders[Power::RUSSIA].push_back(Order("F SEV - BLA"));
  orders[Power::RUSSIA].push_back(Order("A MOS - UKR"));
  orders[Power::RUSSIA].push_back(Order("A WAR - GAL"));
  orders[Power::TURKEY].push_back(Order("F ANK - BLA"));
  orders[Power::TURKEY].push_back(Order("A CON - BUL"));
  orders[Power::TURKEY].push_back(Order("A SMY - CON"));

  auto t_start = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) {
    state.process(orders);
  }
  auto t_end = std::chrono::steady_clock::now();

  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  LOG(ERROR) << micros << " / " << N << " = " << static_cast<float>(micros) / N
             << " us / process";

  return 0;
}
