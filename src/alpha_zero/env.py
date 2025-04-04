from copy import deepcopy
from typing import List, Optional
import random

from src.alpha_zero.utils import *


class DurakEnv:

    def __init__(self):
        self.hands = [[], []]
        self.deck = []
        self.discard = []
        self.trump_suit = None
        self.attacker = 0
        self.defender = 1
        self.state = 0
        self.table_cards: List[Tuple[Optional[int], Optional[int]]] = []
        self.done = False
        self.payoff = 0

    def reset(self) -> Tuple[Tuple, float, bool]:
        all_cards = list(range(36))
        random.shuffle(all_cards)
        self.hands[0] = all_cards[:6]
        self.hands[1] = all_cards[6:12]
        self.deck = all_cards[12:]
        self.discard = []

        trump_card = self.deck[-1]
        _, suit = card_id_to_rank_suit(trump_card)
        self.trump_suit = suit

        self.attacker = random.randint(0, 1)
        self.defender = 1 - self.attacker
        self.state = self.attacker

        self.table_cards = []
        self.done = False
        self.payoff = 0

        return self.get_current_obs()

    def get_current_obs(self) -> Tuple[Tuple, float, bool]:
        obs = self._get_obs(self.state)
        return obs, self.payoff, self.done

    def _get_obs(self, player_idx: int) -> Tuple:
        opp_idx = 1 - player_idx
        table_tuple = tuple((ac, dc) for (ac, dc) in self.table_cards)
        obs = (
            self.trump_suit,
            self.attacker,
            self.defender,
            table_tuple,
            len(self.deck),
            tuple(sorted(self.hands[player_idx])),
            len(self.hands[opp_idx])
        )
        return obs

    def step(self, action: Optional[int]) -> Tuple[Tuple, float, bool]:
        if self.done:
            return self.get_current_obs()

        current_player = self.state
        if action is not None:
            if action not in self.hands[current_player]: # illegal action
                return self.get_current_obs()

        self._apply_action(current_player, action)
        self._check_end()

        return self.get_current_obs()

    def _apply_action(self, player_idx: int, action: Optional[int]):
        if player_idx == self.attacker:
            if action is None:
                all_covered = all(dc is not None for (ac, dc) in self.table_cards)
                if all_covered and len(self.table_cards) > 0:
                    for (ac, dc) in self.table_cards:
                        self.discard.append(ac)
                        if dc is not None:
                            self.discard.append(dc)
                    self.table_cards.clear()
                    self._distribute_cards()

                    old_att = self.attacker
                    self.attacker = self.defender
                    self.defender = old_att
                    self.state = self.attacker
                else:
                    pass
            else:
                self.hands[player_idx].remove(action)
                self.table_cards.append((action, None))
                self.state = self.defender
        else:
            if action is None:
                for (ac, dc) in self.table_cards:
                    self.hands[player_idx].append(ac)
                    if dc is not None:
                        self.hands[player_idx].append(dc)
                self.table_cards.clear()
                self._distribute_cards()
                self.state = self.attacker
            else:
                idx_to_cover = None
                for i, (ac, dc) in enumerate(self.table_cards):
                    if dc is None:
                        idx_to_cover = i
                        break
                if idx_to_cover is None:
                    return
                ac, dc = self.table_cards[idx_to_cover]
                if can_beat(ac, action, self.trump_suit):
                    self.hands[player_idx].remove(action)
                    self.table_cards[idx_to_cover] = (ac, action)
                    self.state = self.attacker
                else:
                    return

    def _distribute_cards(self):
        for who in [self.attacker, self.defender]:
            while len(self.hands[who]) < 6 and len(self.deck) > 0:
                c = self.deck.pop()
                self.hands[who].append(c)

    def sample_opp_hand(self, k):
        opp_idx = 1 - self.state
        candidates = list(self.deck + self.hands[opp_idx])

        opp_count = len(self.hands[opp_idx])

        variants = []
        for _ in range(k):
            random.shuffle(candidates)
            new_opp = deepcopy(candidates[:opp_count])
            variants.append(new_opp)

        return variants

    def _check_end(self):
        if len(self.deck) != 0:
            self.done = False
            self.payoff = 0
            return

        c0 = len(self.hands[0])
        c1 = len(self.hands[1])
        if c0 == 0 and c1 == 0:
            self.done = True
            self.payoff = 0
        elif c0 == 0:
            self.done = True
            self.payoff = +1
        elif c1 == 0:
            self.done = True
            self.payoff = -1


def valid_actions(env: DurakEnv) -> List[Optional[int]]:
    if env.done:
        return []

    current = env.state
    att = env.attacker
    hand = env.hands[current]
    actions: List[Optional[int]] = []

    if current == att: # valid actions for attack
        if len(env.table_cards) == 0:
            actions.extend(hand)
        else:
            ranks_on_table = set()
            for (ac, dc) in env.table_cards:
                if ac is not None:
                    rA, _ = card_id_to_rank_suit(ac)
                    ranks_on_table.add(rA)
                if dc is not None:
                    rD, _ = card_id_to_rank_suit(dc)
                    ranks_on_table.add(rD)
            for c in hand:
                rC, sC = card_id_to_rank_suit(c)
                if rC in ranks_on_table:
                    actions.append(c)

        if len(env.table_cards) > 0:
            all_cov = all(dc is not None for (ac, dc) in env.table_cards)
            if all_cov:
                actions.append(None)
    else: # valid actions for defend
        actions.append(None)
        uncovered = []
        for (ac, dc) in env.table_cards:
            if ac is not None and dc is None:
                uncovered.append(ac)
        if uncovered:
            for c_def in hand:
                for c_att in uncovered:
                    if can_beat(c_att, c_def, env.trump_suit):
                        actions.append(c_def)
                        break

    return actions