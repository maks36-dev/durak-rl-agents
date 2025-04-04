from src.rl.players.base_players.base_player import BasePlayer


class GreedyPlayer(BasePlayer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_min_card(cards, trump):
        min_card = cards[0]
        for card in cards:
            if card.suit == min_card.suit:
                if card < min_card:
                    min_card = card
            elif card.suit != trump and min_card.suit == trump:
                min_card = card
        return min_card

    def get_action(self, trump , action, table_cards, deck_count, opponent_hand_count, hand_cards):

        if action:  # attack
            var_of_ranks = self.table_cards_rank(table_cards)

            if len(var_of_ranks) == 0:
                attack_card = self.get_min_card(self.hand, trump)
                self.hand.remove(attack_card)
                return attack_card
            else:
                attack_cards = []
                for card in self.hand:
                    if card.rank in var_of_ranks:
                        attack_cards.append(card)
                if attack_cards:
                    attack_card = self.get_min_card(attack_cards, trump)
                    self.hand.remove(attack_card)
                    return attack_card
                return None
        else:  # defend
            for attack_card, defend_card in table_cards:
                if attack_card is not None and defend_card is None:
                    defend_cards = []
                    for card in self.hand:
                        if card > attack_card or (card.suit == trump and attack_card.suit != trump):
                            defend_cards.append(card)

                    if defend_cards:
                        defend_card = self.get_min_card(defend_cards, trump)
                        self.hand.remove(defend_card)
                        return defend_card
                    break
            return None

