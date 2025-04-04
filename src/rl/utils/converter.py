import numpy as np

trumps_id = {'clubs': np.array([0, 0]), 'diamonds': np.array([0, 1]), 'hearts': np.array([1, 1]), 'spades': np.array([1, 0])}

def valid_actions(trump, action, table_cards, deck_count, opponent_hand_count, hand_cards):
    hand_cards_mask = np.zeros(37)

    for card in hand_cards:
        hand_cards_mask[card.id] = 1

    if action:
        if table_cards[0][0] is None:
            return hand_cards_mask
        else:
            table_cards_mask = np.zeros(37)
            for attack_card, defend_card in table_cards:
                if attack_card is None:
                    break

                for i in range(4):
                    table_cards_mask[4 * (attack_card.id // 4) + i] = 1
                    if defend_card is not None:
                        table_cards_mask[4 * (defend_card.id // 4) + i] = 1

            valid_cards = hand_cards_mask * table_cards_mask
            valid_cards[36] = 0.5
            return valid_cards
    else:
        for attack_card, defend_card in table_cards:
            if attack_card is not None and defend_card is None:
                defend_cards = np.zeros(37)
                for card in hand_cards:
                    if card > attack_card or (card.suit == trump and attack_card.suit != trump):
                        defend_cards[card.id] = 1

                defend_cards[36] = 0.5

                return defend_cards

def convert_to_tensor(trump, action, table_cards, deck_count, opponent_hand_count, hand_cards):
    deck_count_mask = np.array(list(format(deck_count ^ (deck_count >> 1), '06b')), dtype=np.float32)
    opponent_hand_count_mask = np.array(list(format(opponent_hand_count ^ (opponent_hand_count >> 1), '06b')),
                                        dtype=np.float32)
    trump = trumps_id[trump]
    hand_cards_mask = np.zeros(36, dtype=np.float32)
    table_cards_mask = np.zeros(36, dtype=np.float32)
    action_mask = np.array([int(action)], dtype=np.float32)

    for first_card, second_card in table_cards:
        if first_card:
            table_cards_mask[first_card.id] = 1
        if second_card:
            table_cards_mask[second_card.id] = 1

    for card in hand_cards:
        if card:
            hand_cards_mask[card.id] = 1

    return np.hstack([table_cards_mask, deck_count_mask, opponent_hand_count_mask, action_mask, trump, hand_cards_mask])