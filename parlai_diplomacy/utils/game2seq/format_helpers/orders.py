#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
This file contains functions for converting different kinds of action data
structures to/from string sequences that can be fed to a parlai model.

Each type (e.g. Action, JointAction, RolloutJointAction) has two methods:
- a "flatten" method, which returns a str sequence, contained in the OrdersFlattener
- a "unflatten" method, which converts a str sequence back to the data struct, contained in the OrdersUnflattener

An example of the formatted string sequence is shown in each type's section header.
"""

from collections import defaultdict
import logging
from typing import Dict, List, Optional, TypeVar, Tuple

from fairdiplomacy.game import POWERS, sort_phase_key
from fairdiplomacy.typedefs import (
    Action,
    GameJson,
    JointAction,
    Phase,
    Power,
    RolloutAction,
    RolloutJointAction,
)
from fairdiplomacy.models.consts import LOCS
from fairdiplomacy.utils.order_idxs import canonicalize_action
from fairdiplomacy.utils.typedefs import is_phase_name

import parlai_diplomacy.utils.datapath_constants as constants
import parlai_diplomacy.utils.misc as misc
from parlai_diplomacy.utils.game2seq.format_helpers.base_helper import BaseFormatHelper
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    COUNTRY_ID_TO_POWER,
    add_end_token,
)
from parlai_diplomacy.utils.game2seq.typing import (
    OrderHistoryDict,
    TrainRolloutPseudoOrderDict,
)

RolloutDict = TypeVar("RolloutDict", RolloutAction, RolloutJointAction)

ALL_HOLDS_MARKER = "BAD"


class OrdersFlattener(BaseFormatHelper):
    """
    Shared utilities for order flattening.

    These orders are shared via a class in order to allow easy versioning.
    """

    def canonicalize_order_coasts(self, order: str) -> str:
        """Force the order string to match parlai model conventions.

        Due to some unfortunate historical quirks that we're stuck with
        due to the large number of models trained under each convention,
        some of our models (most of the no-press models) expect support
        holds of special coasts to look like:
        F MAO S F SPA/NC
        whereas our parlai-based models expect:
        F MAO S F SPA
        """
        # Quicker quit out if this order is definitely not a support of a fleet
        # with a special coast involved.
        if " S F " not in order or "/" not in order:
            return order
        pieces = order.split(" ")
        if (
            len(pieces) == 5
            and pieces[2] == "S"
            and pieces[3] == "F"
            and len(pieces[4]) >= 3
            and pieces[4][-3] == "/"
        ):
            pieces[4] = pieces[4][:-3]
            return " ".join(pieces)
        return order

    def canonicalize_action_coasts(self, action: Action) -> Action:
        return tuple(self.canonicalize_order_coasts(order) for order in action)

    def flatten_action(self, action: Action) -> str:
        """
        Flatten the order in game*.json

        Example output sequence (v1):

        A TRI - VEN; A TYR - PIE; F ADR - ION; F NAP S F ADR - ION [EO_O]
        """
        # canonicalize and sort the orders
        order_list = sorted(list(self.canonicalize_action_coasts(action)))
        # join the orders
        flat_order_str = "; ".join(order_list)
        # maybe add end_of_order token: [EO_O]
        if self.version <= 1:
            # In V2 we remove the [EO_O] tokens
            flat_order_str = add_end_token(flat_order_str, "[EO_O]")

        if self.version >= 2:
            # In V2, we remove the dashes in between orders
            flat_order_str = flat_order_str.replace(" - ", " ")

        # add bad format checking
        if flat_order_str.startswith(","):
            logging.warning(f"order starts with ',', original order_lst: {order_list}")

        return flat_order_str

    def flatten_joint_action(
        self,
        all_order_dct: JointAction,
        speaker: Power,
        mark_all_holds: bool = False,
        all_holds_dct: Optional[Dict[str, bool]] = None,
    ) -> str:
        """
        Example output sequence (speaker last) (v1):

           England: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O]
           France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]
           Italy: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]
           Germany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]
           Turkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]
           Russia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]
           Austria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]
        """
        if mark_all_holds:
            assert all_holds_dct is not None

        def format_one_order(one_order, player_name):
            """
            format the order for one player
            return:
                England: [flat_order]
            """
            one_flat_order = self.flatten_action(one_order)
            if mark_all_holds and all_holds_dct[player_name]:  # type: ignore
                one_flat_order = f"{ALL_HOLDS_MARKER} {one_flat_order}"

            if self.version <= 1:
                # In version 2, all player names are uppercase
                player_name = player_name.capitalize()

            one_flat_order = f"{player_name}: {one_flat_order}"

            return one_flat_order

        all_order_list = []
        non_speakers = [x for x in COUNTRY_ID_TO_POWER.values() if x != speaker]
        all_powers = non_speakers + [speaker]  # speaker should come last
        for power in all_powers:
            player_orders = all_order_dct.get(power, tuple())
            if self.version >= 2 and not player_orders:
                # In version 2, we don't include lines for empty orders
                continue

            one_flat_order = format_one_order(player_orders, power)
            all_order_list.append(one_flat_order)

        return "\n".join(all_order_list)

    def flatten_rollout_action(
        self, rollout_action: RolloutAction, strip_current_phase: bool = True
    ) -> str:
        """
        Example output sequence (v1):

           F1905R  # optional, depends on `strip_current_phase`
           F ION R NAP [EO_O]
           W1905A
            [EO_O]
           S1906M
           A TRI - VEN; A TYR - PIE; F ADR - ION; F NAP S F ADR - ION [EO_O]
        """
        substrs = []
        for phase, action in phase_sorted(rollout_action).items():
            substrs.append(phase)
            substrs.append(self.flatten_action(action))
        if strip_current_phase:
            substrs = substrs[1:]
        return "\n".join(substrs)

    def flatten_only_first_action_of_rollout_action(
        self, rollout_action: RolloutAction, strip_current_phase: bool = True
    ) -> str:
        """
        Example output sequence (v1):

           F1905R  # optional, depends on `strip_current_phase`
           F ION R NAP [EO_O]

           And the remaining text that flatten_rollout_action would have outputted:
           W1905A
            [EO_O]
           S1906M
           A TRI - VEN; A TYR - PIE; F ADR - ION; F NAP S F ADR - ION [EO_O]

           is all omitted.
        """
        substrs = []
        for phase, action in phase_sorted(rollout_action).items():
            substrs.append(phase)
            substrs.append(self.flatten_action(action))
            break
        if strip_current_phase:
            substrs = substrs[1:]
        return "\n".join(substrs)

    def flatten_rollout_joint_action(
        self,
        rollout_joint_action: RolloutJointAction,
        speaker: Power,
        mark_all_holds: bool = False,
        all_holds_dcts: Dict[Phase, Dict[str, bool]] = {},
    ) -> str:
        """
        Example output sequence (v1):

            W1907A
            England:  [EO_O]
            France:  [EO_O]
            Italy: A PIE D [EO_O]
            Germany:  [EO_O]
            Austria: A VIE B [EO_O]
            Russia:  [EO_O]
            Turkey: A CON B [EO_O]
            S1908M
            England: A YOR - LVP; F EDI - NWG; F NTH - NWY; F STP/NC S F NTH - NWY; F WES - TUN [EO_O]
            France: A BEL S A HOL; A BUR - MAR; A GAS - SPA; F IRI S F NAO - LVP; F MAO S A GAS - SPA; F NAO - LVP [EO_O]
            Italy: F TYS H [EO_O]
            Germany: A HOL H; A MUN S A SIL; A NWY H; A PRU S A SIL; A SIL S A PRU; F DEN - NTH; F NWG S A NWY [EO_O]
            Austria: A BOH S A GAL - SIL; A GAL - SIL; A TRI S A VIE - TYR; A VEN - PIE; A VIE - TYR; A WAR S A GAL - SIL [EO_O]
            Russia:  [EO_O]
            Turkey: A ALB - SER; A APU S F NAP - ROM; A CON - BUL; A MOS - LVN; A SER - BUL; A UKR - MOS; F GRE - ALB; F ION S F NAP; F NAP - ROM [EO_O]
        """
        substrs = []
        for phase, joint_action in phase_sorted(rollout_joint_action).items():
            substrs.append(phase)
            substrs.append(
                self.flatten_joint_action(
                    joint_action,
                    speaker,
                    mark_all_holds=mark_all_holds,
                    all_holds_dct=(all_holds_dcts.get(phase) if mark_all_holds else None),
                )
            )
        return "\n".join(substrs)

    def flatten_rollout_joint_action_bilateral_powermajor(
        self, rollout_joint_action: RolloutJointAction, speaker: Power, recipient: Power
    ) -> str:
        """
        Example output BILATERAL sequence (recipient first) (v1):

           Germany: A BER B; F KIE B [EO_O]
           S1902M
           A BER - KIE; A HOL - RUH; A MUN S A HOL - RUH; F DEN - SKA; F KIE - DEN [EO_O]
           Russia:  [EO_O]
           S1902M
           A UKR S A WAR - GAL; A WAR - GAL; F BOT - SWE; F SEV - RUM [EO_O]
        """
        substrs = []
        for power in [recipient, speaker]:  # recipient first
            if self.version <= 1:
                # In version 2, all power names are uppercase
                power = power.capitalize()
            substrs.append(
                power
                + ": "
                + self.flatten_rollout_action(
                    extract_rollout_action_for_power(rollout_joint_action, power.upper())
                )
            )

        return "\n".join(substrs)

    def flatten_rollout_joint_action_bilateral_phasemajor(
        self,
        joint_action_bilateral: RolloutJointAction,
        speaker: Power,
        recipient: Power,
        speaker_first: bool,
    ) -> str:
        """Example output (sender=ENGLAND, recipient=GERMANY):

        F1901R
        W1901A
        GERMANY: A BER B; F KIE B
        ENGLAND: F LON B
        S1902M
        GERMANY: A BER MUN; A DEN H; A MUN BOH; F HOL S F KIE HEL; F KIE HEL
        ENGLAND: A WAL PIC VIA; F ENG C A WAL PIC; F LON S F ENG; F NWY NTH
        """
        if self.version < 2:
            raise NotImplementedError("Introduced in version 2")
        substrs = []
        for phase, joint_action in sorted(
            joint_action_bilateral.items(), key=lambda p: sort_phase_key(p[0])
        ):
            substrs.append(phase)
            power_list = (speaker, recipient) if speaker_first else (recipient, speaker)
            for power in power_list:
                substr = power + ": " + self.flatten_action(joint_action.get(power, tuple()))
                substrs.append(substr)
        return "\n".join(substrs)

    def flatten_one_phase_order(self, order_dct: JointAction, phase_name: Phase) -> str:
        """
        Example output sequence (v1):

            S1908M
            England: A YOR - LVP; F EDI - NWG; F NTH - NWY; F STP/NC S F NTH - NWY; F WES - TUN [EO_O]
            ...
            Turkey: A ALB - SER; A APU S F NAP - ROM; A CON - BUL; A MOS - LVN; A SER - BUL; A UKR - MOS; F GRE - ALB; F ION S F NAP; F NAP - ROM [EO_O]
        """
        flat_orders = []
        for speaker, order in order_dct.items():
            if self.version >= 2 and not order:
                # In V2, we don't include empty orders in the joint action
                continue
            flat_order = self.flatten_action(order)
            if self.version <= 1:
                # In V2, all power names are uppercased
                speaker = speaker.capitalize()
            flat_order = f"{speaker}: {flat_order}"
            flat_orders.append(flat_order)

        # add phase info
        flat_orders = phase_name + "\n" + "\n".join(flat_orders)

        return flat_orders

    def flatten_order_history(self, order_history_dct: OrderHistoryDict) -> str:
        """
        Example output sequence (v1):

           S1901M
           France: [action] [EO_O]
           Italy: [action] [EO_O]
           ...
           [phase_name]
           [player1]: [action] [EO_O]
        """
        phase_orders = []
        for phase_name, order_dct in order_history_dct.items():
            phase_order = self.flatten_one_phase_order(order_dct, phase_name)
            phase_orders.append(phase_order)
        return "\n".join(phase_orders)

    def flatten_last_movement_phase_order(self, order_history_dct: OrderHistoryDict) -> str:
        """
        Given an order history dict, formats the last movement phase orders into a flat string
        """
        last_movement_phase = get_last_movement_phase(order_history_dct)
        if not last_movement_phase:
            return ""
        last_movement_phase_order_dct = {
            last_movement_phase: order_history_dct[last_movement_phase]
        }
        return self.flatten_order_history(last_movement_phase_order_dct)

    def flatten_order_history_since_last_movement_phase(
        self, order_history_dct: OrderHistoryDict
    ) -> str:
        """
        Given an order history dict, formats the order history through the last movement phase orders
        into a flat string
        """
        last_movement_phase = get_last_movement_phase(order_history_dct)
        if not last_movement_phase:
            return ""
        ordered_phases = misc.get_ordered_dict_keys(order_history_dct)
        last_movement_phase_idx = ordered_phases.index(last_movement_phase)
        phases_since_last_movement_phase = ordered_phases[last_movement_phase_idx:]
        order_history_since_last_movement_phase = {
            p: order_history_dct[p] for p in phases_since_last_movement_phase
        }
        return self.flatten_order_history(order_history_since_last_movement_phase)

    def flatten_order_history_since_last_n_movement_phases(
        self, order_history_dct: OrderHistoryDict, n: int
    ) -> str:
        """
        Given an order history dict, formats the order history through the last n movement phase orders
        into a flat string
        """
        last_n_movement_phase = get_last_n_movement_phases(order_history_dct, n)
        if not last_n_movement_phase:
            return ""
        ordered_phases = misc.get_ordered_dict_keys(order_history_dct)
        last_n_movement_phase_idx = ordered_phases.index(last_n_movement_phase)
        phases_since_last_n_movement_phase = ordered_phases[last_n_movement_phase_idx:]
        order_history_since_last_n_movement_phase = {
            p: order_history_dct[p] for p in phases_since_last_n_movement_phase
        }
        return self.flatten_order_history(order_history_since_last_n_movement_phase)

    def flatten_last_phase_order(self, order_history_dct: OrderHistoryDict) -> str:
        """
        Given an order history dict, flatten the last phase
        """
        if not order_history_dct:
            return ""

        last_phase = misc.last_dict_key(order_history_dct)
        last_phase_order_dct = {last_phase: order_history_dct[last_phase]}
        return self.flatten_order_history(last_phase_order_dct)

    def flatten_train_singleview_pseudo_orders(
        self,
        pseudo_orders: TrainRolloutPseudoOrderDict,
        speaker: Power,
        recipient: Power,
        phase: Phase,
        rollout: bool = False,
        rollout_except_movement: bool = True,
    ) -> str:
        """
        Provided annotated singleview pseudo orders from the train set, we flatten them
        to provide as input to the dialogue model.
        """
        if rollout and not rollout_except_movement:
            # rollout on every phase
            (
                self_pseudos_flat,
                partner_pseudos_flat,
            ) = self._reflatten_train_bilateral_rollout_pseudoorders(
                pseudo_orders["self_prefix_rollout"],
                pseudo_orders["partner_prefix_rollout"],
                unflattener_version=constants.PSEUDO_ORDER_PREFIX_ROLLOUT_DIR_VERSION,  # Version these pseudo orders are compiled with
            )
        elif rollout and not is_movement_phase(phase):
            # we use rollout pseudo orders on builds and retreats
            (
                self_pseudos_flat,
                partner_pseudos_flat,
            ) = self._reflatten_train_bilateral_rollout_pseudoorders(
                pseudo_orders["rollout_self"],
                pseudo_orders["rollout_partner"],
                unflattener_version=constants.PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR_VERSION,  # Version these pseudo orders are compiled with
            )
        else:
            # Non-rollout pseudo orders
            self_pseudos_flat, partner_pseudos_flat = self._reflatten_train_bilateral_pseudoorders(
                pseudo_orders["self"],
                pseudo_orders["partner"],
                unflattener_version=constants.PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR_VERSION,  # Version these pseudo orders are compiled with
            )

        if self.version <= 1:
            # In version 2, all power names are uppercase
            recipient = recipient.capitalize()
            speaker = speaker.capitalize()

        str_orders = f"{recipient}: {partner_pseudos_flat}\n{speaker}: {self_pseudos_flat}"
        return str_orders

    def _reflatten_train_bilateral_pseudoorders(
        self,
        generated_speaker_orders: str,
        generated_recipient_orders: str,
        unflattener_version: int,
    ) -> Tuple[str, str]:
        """
        Take the generated self/partner pseudo orders from a model, unflatten to a structured object,
        and then reflatten to a string given the formatting version requested.

        Returns tuple of [flat_self_pseudos, flat_partner_pseudos]
        """
        unflattener = OrdersUnflattener(
            unflattener_version,
        )  # What version were the latest pseudo orders compiled with
        self_pseudos = unflattener.unflatten_action(generated_speaker_orders)
        partner_pseudos = unflattener.unflatten_action(generated_recipient_orders)
        self_pseudos_flat = self.flatten_action(self_pseudos)
        partner_pseudos_flat = self.flatten_action(partner_pseudos)

        return self_pseudos_flat, partner_pseudos_flat

    def _reflatten_train_bilateral_rollout_pseudoorders(
        self,
        generated_speaker_rollout: str,
        generated_recipient_rollout: str,
        unflattener_version: int,
    ) -> Tuple[str, str]:
        """
        Take the generated self/partner ROLLOUT pseudo orders from a model, unflatten to a structured object,
        and then reflatten to a string given the formatting version requested.

        Returns tuple of [flat_self_pseudos, flat_partner_pseudos]
        """
        unflattener = OrdersUnflattener(
            unflattener_version,
        )  # What version were the latest pseudo orders compiled with
        # we use rollout pseudo orders on builds and retreats
        self_pseudos = unflattener.unflatten_rollout_action(generated_speaker_rollout)
        partner_pseudos = unflattener.unflatten_rollout_action(generated_recipient_rollout)
        # now reflatten, corresponding to the version
        self_pseudos_flat = self.flatten_rollout_action(self_pseudos)
        partner_pseudos_flat = self.flatten_rollout_action(partner_pseudos)

        return self_pseudos_flat, partner_pseudos_flat


class OrdersUnflattener(BaseFormatHelper):
    """
    Shared utilities for order unflattening.

    These orders are shared via a class in order to allow easy versioning.
    """

    def unflatten_action(self, seq: str) -> Action:
        """
        Example input (v1):

        A TRI - VEN; A TYR - PIE; F ADR - ION; F NAP S F ADR - ION [EO_O]
        """
        # remove [EO_O]
        order = seq.replace("[EO_O]", "").strip()

        if not order:
            # empty order
            return tuple()

        order_lst = order.split("; ")

        if self.version >= 2:
            # In V2, we remove the dashes in the orders when flattening
            new_order_lst = []
            for order in order_lst:
                order_space_split = order.split(" ")
                for i in range(0, len(order_space_split) - 1):
                    chunk = order_space_split[i : i + 2]
                    if chunk[0] in LOCS and chunk[1] in LOCS:
                        order = order.replace(f"{chunk[0]} {chunk[1]}", f"{chunk[0]} - {chunk[1]}")
                new_order_lst.append(order)
            order_lst = new_order_lst

        # sort the orders lexicographically, and put them in a tuple so they can be used as dict keys
        action = canonicalize_action(tuple(order_lst))
        return action

    def unflatten_joint_action(self, all_orders_sequence: str) -> JointAction:
        """
        Example input sequence (speaker last) (v1):

        England: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O]
        France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]
        Italy: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]
        Germany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]
        Turkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]
        Russia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]
        Austria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]
        """
        order_dct = {}
        if not all_orders_sequence:
            return order_dct

        order_split = all_orders_sequence.split("\n")
        for order in order_split:
            try:
                power, order_seq = order.split(": ")
            except ValueError:
                logging.warning("Failed to parse power-orders pair: %s", order)
                continue
            order_dct[power.upper()] = self.unflatten_action(order_seq)

        return order_dct

    def unflatten_rollout_action(
        self, seq: str, current_phase: Optional[Phase] = None
    ) -> RolloutAction:
        """
        Example input sequence (v1):

           F1905R  # optional
           F ION R NAP [EO_O]
           W1905A
            [EO_O]
           S1906M
           A TRI - VEN; A TYR - PIE; F ADR - ION; F NAP S F ADR - ION [EO_O]
        """
        substrs = seq.split("\n")

        # The current phase is optional in the seq. Parse it if it's present.
        if is_phase_name(substrs[0]):
            assert current_phase is None, "Why set current_phase if it's found in the seq?"
            current_phase, substrs = substrs[0], substrs[1:]
        else:
            assert current_phase is not None, (
                "No current phase in seq, must set current_phase. First line of seq: " + substrs[0]
            )

        current_action_seq, substrs = substrs[0], substrs[1:]
        if len(substrs) % 2 != 0:
            raise ValueError(seq)

        rollout_action = {
            substrs[x]: self.unflatten_action(substrs[x + 1])
            for x in range(0, len(substrs), 2)
            if is_phase_name(substrs[x])
        }
        rollout_action[Phase(current_phase)] = self.unflatten_action(current_action_seq)
        return rollout_action

    def unflatten_rollout_joint_action(
        self, seq: str, current_phase: Optional[Phase] = None
    ) -> RolloutJointAction:
        """
        Example input of a rollout joint action:

        W1907A
        England:  [EO_O]
        France:  [EO_O]
        Italy: A PIE D [EO_O]
        Germany:  [EO_O]
        Austria: A VIE B [EO_O]
        Russia:  [EO_O]
        Turkey: A CON B [EO_O]
        S1908M
        England: A YOR - LVP; F EDI - NWG; F NTH - NWY; F STP/NC S F NTH - NWY; F WES - TUN [EO_O]
        France: A BEL S A HOL; A BUR - MAR; A GAS - SPA; F IRI S F NAO - LVP; F MAO S A GAS - SPA; F NAO - LVP [EO_O]
        Italy: F TYS H [EO_O]
        Germany: A HOL H; A MUN S A SIL; A NWY H; A PRU S A SIL; A SIL S A PRU; F DEN - NTH; F NWG S A NWY [EO_O]
        Austria: A BOH S A GAL - SIL; A GAL - SIL; A TRI S A VIE - TYR; A VEN - PIE; A VIE - TYR; A WAR S A GAL - SIL [EO_O]
        Russia:  [EO_O]
        Turkey: A ALB - SER; A APU S F NAP - ROM; A CON - BUL; A MOS - LVN; A SER - BUL; A UKR - MOS; F GRE - ALB; F ION S F NAP; F NAP - ROM [EO_O]
        """
        order_groups = defaultdict(list)
        substrs = seq.split("\n")

        # The current phase is optional in the seq. Parse it if it's present.
        if is_phase_name(substrs[0]):
            assert current_phase is None, "Why set current_phase if it's found in the seq?"
            current_phase, substrs = substrs[0], substrs[1:]
        else:
            assert current_phase is not None, (
                "No current phase in seq, must set current_phase. First line of seq: " + substrs[0]
            )

        for line in substrs:
            if is_phase_name(line):
                current_phase = line
            else:
                assert current_phase is not None
                order_groups[current_phase].append(line)

        phase_to_jointaction = {}
        for phase, order_lst in order_groups.items():
            phase_to_jointaction[phase] = self.unflatten_joint_action("\n".join(order_lst))

        return phase_to_jointaction

    def unflatten_rollout_joint_action_bilateral_phasemajor(self, seq: str) -> RolloutJointAction:
        final = {}
        cur_phase = None
        for line in seq.split("\n"):
            if is_phase_name(line):
                cur_phase = line
                final[cur_phase] = {}
            else:
                if cur_phase is None:
                    logging.error(f"Badly formatted, missing phase {cur_phase}:\n" + seq)
                    continue
                line_fields = line.split(":")
                if len(line_fields) != 2:
                    logging.error(f"Bad output: {line}")
                    continue
                power, action_seq = line_fields
                if power not in POWERS:
                    logging.error(f"Bad power={power}\n{seq}")
                    continue
                action = self.unflatten_action(action_seq)
                final[cur_phase][power] = action
        return final

    def unflatten_rollout_joint_action_bilateral_powermajor(
        self, seq: str, current_phase: Phase,
    ) -> RolloutJointAction:
        """
        Example input BILATERAL sequence (recipient first) (v1):

           Germany: A BER B; F KIE B [EO_O]
           S1902M
           A BER - KIE; A HOL - RUH; A MUN S A HOL - RUH; F DEN - SKA; F KIE - DEN [EO_O]
           Russia:  [EO_O]
           S1902M
           A UKR S A WAR - GAL; A WAR - GAL; F BOT - SWE; F SEV - RUM [EO_O]
        """
        order_groups = defaultdict(lambda: defaultdict(Action))
        substrs = seq.split("\n")

        def contains_power_name(line) -> Optional[Power]:
            powers = [x.capitalize() for x in POWERS] if self.version <= 1 else POWERS
            for power in powers:
                if f"{power}:" in line:
                    return power

            return None

        phase = None
        current_power = None
        for line in substrs:
            contains_power = contains_power_name(line)
            if contains_power:
                phase = current_phase  # Phase resets with a new power (see docstring)
                current_power = contains_power.upper()
                order_groups[phase][current_power] = self.unflatten_action(
                    line.split(f"{contains_power}: ")[-1]
                )
            elif is_phase_name(line):
                phase = line
            else:
                assert current_phase is not None
                assert current_power is not None
                order_groups[phase][current_power] = self.unflatten_action(line)

        phase_to_jointaction = {}
        for phase, joint_action in order_groups.items():
            phase_to_jointaction[phase] = dict(joint_action)

        return phase_to_jointaction

    def train_pseudo_orders_to_rollout_jointaction(
        self,
        train_pseudo_orders: TrainRolloutPseudoOrderDict,
        power: Power,
        recipient: Power,
        curr_phase: Phase,
        rollout_except_movement: bool = True,
    ) -> RolloutJointAction:
        """
        Takes the train pseudo orders and converts them to a rollout joint action.

        Train rollout pseudo orders are of the following format:
            {
                "self": <self actions>
                "partner": <partner actions>
                "rollout_self": <rollout self>
                "rollout_partner": <rollout partner>
            }

        This is useful for the experiment utils
        """
        keys = (
            {"self", "partner", "rollout_self", "rollout_partner"}
            if rollout_except_movement
            else {"self_prefix_rollout", "partner_prefix_rollout"}
        )
        for key in keys:
            # Check that pseudo orders are the right format
            assert key in train_pseudo_orders

        rollout_joint_action = {}
        if rollout_except_movement and is_movement_phase(curr_phase):
            # Get curr phase from the single view pseudo orders
            rollout_joint_action[curr_phase] = {}
            rollout_joint_action[curr_phase][power] = self.unflatten_action(
                train_pseudo_orders["self"]
            )
            rollout_joint_action[curr_phase][recipient] = self.unflatten_action(
                train_pseudo_orders["partner"]
            )
        else:
            # Should be a rollout model
            self_key = "rollout_self" if rollout_except_movement else "self_prefix_rollout"
            partner_key = (
                "rollout_partner" if rollout_except_movement else "partner_prefix_rollout"
            )
            self_rollout = self.unflatten_rollout_action(train_pseudo_orders[self_key])
            partner_rollout = self.unflatten_rollout_action(train_pseudo_orders[partner_key])

            for phase, action in self_rollout.items():
                rollout_joint_action.setdefault(phase, {})
                rollout_joint_action[phase][power] = action
            for phase, action in partner_rollout.items():
                rollout_joint_action.setdefault(phase, {})
                rollout_joint_action[phase][recipient] = action

        return rollout_joint_action


#####################################
#  MISCELLANEOUS UTILITY FUNCTIONS  #
#####################################


def order_is_empty(order_sequence: str) -> bool:
    """
    Check if an order sequence is empty
    """
    order = order_sequence.replace("[EO_O]", "").strip()
    if not order:
        return True

    return False


def extract_rollout_action_for_power(
    rollout_joint_action: RolloutJointAction, power: Power
) -> RolloutAction:
    return {
        phase: joint_action.get(power, ())
        for phase, joint_action in rollout_joint_action.items()
        if phase != "COMPLETED"
    }


def build_order_history_dct(game_json: GameJson, cur_phase: Phase) -> OrderHistoryDict:
    """
    Return order history dict given a game json

    get all speakers' previous orders
    """
    orders = {}
    for phase in game_json:
        if phase == "is_partial" or phase == "partial":
            continue
        if phase == cur_phase:
            break

        orders[phase] = {}
        for _, speaker in COUNTRY_ID_TO_POWER.items():
            if "orders" in game_json[phase]:
                order = game_json[phase]["orders"].get(speaker, [])
            else:
                order = []
            orders[phase][speaker] = order

    return orders


def is_movement_phase(phase: Phase) -> bool:
    if phase.endswith("M"):
        return True

    return False


def get_last_movement_phase(order_history_dict):
    """
    Get the phase key for the last movement phase
    """
    previous_phases = misc.get_ordered_dict_keys(order_history_dict)
    for phase in reversed(previous_phases):
        if is_movement_phase(phase):
            return phase
    return None


def get_last_n_movement_phases(history_dict, n: int):
    """
    Get the phase key for the last movement phase
    """
    previous_phases = misc.get_ordered_dict_keys(history_dict)
    last_movement_phase = None
    movement_phases_found = 0
    for phase in reversed(previous_phases):
        if is_movement_phase(phase):
            last_movement_phase = phase
            movement_phases_found += 1
        if movement_phases_found == n:
            break

    return last_movement_phase


def get_phases_until_next_movement_phase(
    curr_phase: Phase, game_json: GameJson, including_current: bool = True
) -> List[Phase]:
    """
    Given the current phase and game json, return a list of phases containing
    the current phase up until the next movement phase.

    If including_current is True, stop if the current phase is a movement phase
    """
    if including_current and is_movement_phase(curr_phase):
        return [curr_phase]

    all_phases = misc.get_ordered_dict_keys(game_json)
    all_subsequent_phases = all_phases[all_phases.index(curr_phase) :]
    all_phases_until_next_movement_phase = []
    for phase in all_subsequent_phases:
        all_phases_until_next_movement_phase.append(phase)
        if (
            is_movement_phase(phase) and phase != curr_phase
        ):  # we already returned early if including_current = True
            break

    return all_phases_until_next_movement_phase


def phase_sorted(rollout_dict: RolloutDict) -> RolloutDict:
    return {k: v for k, v in sorted(rollout_dict.items(), key=lambda p: sort_phase_key(p[0]))}  # type: ignore
