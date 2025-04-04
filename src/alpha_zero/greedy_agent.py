from src.alpha_zero.env import valid_actions
from src.alpha_zero.utils import *


class GreedyAgent:

    @staticmethod
    def act(env):
        if env.done:
            return None

        va = valid_actions(env)
        current_player = env.state
        attacker = env.attacker

        if current_player == attacker:
            card_actions = [a for a in va if a is not None]
            if not card_actions:
                return None

            best_card = min(card_actions, key=lambda c: (c//4))
            return best_card
        else:
            uncovered = []
            for (ac, dc) in env.table_cards:
                if ac is not None and dc is None:
                    uncovered.append(ac)

            if not uncovered:
                return None

            suitable_cards = []
            for a_def in va:
                if a_def is not None:
                    for c_att in uncovered:
                        if can_beat(c_att, a_def, env.trump_suit):
                            suitable_cards.append(a_def)
                            break

            if suitable_cards:
                best_card = min(suitable_cards, key=lambda c: (c//4))
                return best_card
            else:
                return None
