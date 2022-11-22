#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
from typing import Sequence, Dict, List


class OrderHandler(ABC):
    @abstractmethod
    def get_orders(self, game, power) -> List[str]:
        raise NotImplementedError()

    def get_orders_many_powers(self, game, powers: Sequence[str]) -> Dict[str, List[str]]:
        return {p: self.get_orders(game, p) for p in powers}
