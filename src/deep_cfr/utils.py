from typing import Tuple
import torch

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


def encode_observation(obs) -> torch.Tensor:
    trump_suit, attacker, defender, table_cards, deck_size, my_hand, opp_cards_count = obs

    suit_map = {'clubs': 0, 'diamonds': 1, 'hearts': 2, 'spades': 3}
    suit_vec = [0, 0, 0, 0]
    if trump_suit in suit_map:
        suit_vec[suit_map[trump_suit]] = 1

    att_def_vec = [attacker, defender]

    deck_size_vec = [deck_size / 36.0]

    opp_vec = [opp_cards_count / 36.0]

    hand_vec = [0] * 36
    for c in my_hand:
        hand_vec[c] = 1

    table_att_vec = [0] * 36
    table_def_vec = [0] * 36
    for (ac, dc) in table_cards:
        if ac is not None:
            table_att_vec[ac] = 1
        if dc is not None:
            table_def_vec[dc] = 1

    features = suit_vec + att_def_vec + deck_size_vec + opp_vec + hand_vec + table_att_vec + table_def_vec

    return torch.tensor(features, dtype=torch.float32)