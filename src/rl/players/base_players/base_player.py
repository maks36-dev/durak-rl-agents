class BasePlayer:
    def __init__(self):
        self.hand = []

    def get_min_trump(self, trump):
        min_trump_card = None
        if self.hand:
            for card in self.hand:
                if card.suit == trump:
                    if min_trump_card is None:
                        min_trump_card = card
                    elif card < min_trump_card:
                        min_trump_card = card
        return min_trump_card

    def reset(self):
        self.hand = []

    @staticmethod
    def table_cards_rank(table_cards):
        variance_of_cards = set()
        for card1, card2 in table_cards:
            if card1 and card2:
                variance_of_cards = variance_of_cards.union({card1.rank, card2.rank})
            else:
                break
        return variance_of_cards

    @property
    def hand_size(self):
        return len(self.hand)

    @property
    def cards(self):
        return "(" + " ".join([str(card) for card in self.hand]) + ")"

    def take_card(self, card):
        assert card is not None, "Card is None!"
        self.hand.append(card)

    def get_table_cards(self, table_cards):
        for attack_card, defend_card in table_cards:
            if attack_card is not None:
                self.take_card(attack_card)
            if defend_card is not None:
                self.take_card(defend_card)

    def is_hand_empty(self):
        return len(self.hand) == 0

    def get_action(self, *args, **kwargs):
        ...
