from typing import Tuple
import numpy as np


RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['clubs', 'diamonds', 'hearts', 'spades']

rank_to_index = {r: i for i, r in enumerate(RANKS)}
suit_to_index = {s: i for i, s in enumerate(SUITS)}


def card_id_to_rank_suit(card_id: int) -> Tuple[str, str]:
    rank_i = card_id // 4
    suit_i = card_id % 4
    return RANKS[rank_i], SUITS[suit_i]


def card_str(card_id: int) -> str:
    rank, suit = card_id_to_rank_suit(card_id)
    suit_symbol = {
        'clubs': '♣',
        'diamonds': '♦',
        'hearts': '♥',
        'spades': '♠'
    }[suit]
    return f"{rank}{suit_symbol}"


def can_beat(card_attacker: int, card_defender: int, trump_suit: str) -> bool:
    rankA, suitA = card_id_to_rank_suit(card_attacker)
    rankD, suitD = card_id_to_rank_suit(card_defender)
    trumpA = (suitA == trump_suit)
    trumpD = (suitD == trump_suit)
    if trumpA and trumpD:
        return rank_to_index[rankD] > rank_to_index[rankA]
    elif trumpD and not trumpA:
        return True
    elif (not trumpA) and (not trumpD) and (suitA == suitD):
        return rank_to_index[rankD] > rank_to_index[rankA]
    else:
        return False


def encode_observation(obs, opp_hand) -> np.ndarray:
    (trump_suit, attacker, defender, table_cards, deck_count, my_hand, opp_count) = obs

    vec = np.zeros(154, dtype=np.float32)

    suit_idx = suit_to_index[trump_suit]
    vec[suit_idx] = 1.0

    vec[4 + attacker] = 1.0
    vec[6 + defender] = 1.0

    vec[8] = deck_count / 24.0

    vec[9] = opp_count / 36.0

    offset = 10
    for c in my_hand:
        vec[offset + c] = 1.0
    offset += 36

    table_set = set()
    for (ac, dc) in table_cards:
        if ac is not None:
            table_set.add(ac)
        if dc is not None:
            table_set.add(dc)
    for c in table_set:
        if c < 36:
            vec[offset + c] = 1.0
    offset += 36

    for (ac, dc) in table_cards:
        if ac is not None and dc is None:
            vec[offset + ac] = 1.0
            break
    offset += 36

    for c in opp_hand:
        vec[offset + c] = 1.0
    offset += 36


    return vec

