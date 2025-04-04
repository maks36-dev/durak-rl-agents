from typing import Tuple

RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['clubs', 'diamonds', 'hearts', 'spades']

rank_to_index = {r: i for i, r in enumerate(RANKS)}
suit_to_index = {s: i for i, s in enumerate(SUITS)}


def card_id_to_rank_suit(card_id: int) -> Tuple[str, str]:
    rank_i = card_id // 4
    suit_i = card_id % 4
    return RANKS[rank_i], SUITS[suit_i]


def rank_suit_to_id(rank: str, suit: str) -> int:
    return rank_to_index[rank] * 4 + suit_to_index[suit]


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
    elif suitA == suitD and not trumpA and not trumpD:
        return rank_to_index[rankD] > rank_to_index[rankA]
    else:
        return False


