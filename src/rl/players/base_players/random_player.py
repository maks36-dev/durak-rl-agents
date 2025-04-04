import random
from src.rl.players.base_players.base_player import BasePlayer


class RandomPlayer(BasePlayer):
    def __init__(self):
        super().__init__()

    def get_action(self, trump, action, table_cards, deck_count, opponent_hand_count, hand_cards):
        if action: # attack
            var_of_ranks = self.table_cards_rank(table_cards)

            if len(var_of_ranks) == 0:
                attack_card = random.choice(self.hand)
                self.hand.remove(attack_card)
                return attack_card
            else:
                for card in self.hand:
                    if card.rank in var_of_ranks:
                        attack_card = card
                        self.hand.remove(attack_card)
                        return attack_card
                return None

        else:  # defend
            for attack_card, defend_card in table_cards:
                if attack_card is not None and defend_card is None:
                    for card in self.hand:
                        if card > attack_card or (card.suit == trump and attack_card.suit != trump):
                            self.hand.remove(card)
                            return card
            return None
