from typing import Tuple
from random import randint

from src.rl.players.base_players.base_player import BasePlayer
from src.rl.utils.deck import Deck, Card

class DurakEnv:
    state: bool # 1 - player, 0 computer
    action: bool # 1 - attack, 0 defend

    def __init__(self, player: BasePlayer, computer: BasePlayer):
        self.first_start = None
        self.player = player
        self.computer = computer

        self.deck = None
        self.table_cards = None
        self.trump = None


    def _get_state(self) -> Tuple[Tuple, int, bool]:
        done, reward = self.is_game_over()
        return (self.trump, self.action, self.table_cards, self.deck.size, self.computer.hand_size, self.player.hand), reward, done


    def reset(self) -> Tuple[Tuple, int, bool]:
        self.player.reset()
        self.computer.reset()

        self.deck = Deck()
        self.table_cards = [[None, None] for _ in range(6)]
        self.trump = self.deck.cards[-1].suit

        self.deck.update_player_cards(self.player)
        self.deck.update_player_cards(self.computer)


        if randint(0, 1):
            self.state = True
            self.action = True
            self.first_start = 'player'
        else:
            self.state = False
            self.action = True
            self.first_start = 'computer'

            # computer make action (because it automatically starts)
            computer_action = self.computer.get_action(self.trump, self.action, self.table_cards, self.deck.size, self.computer.hand_size, self.computer.hand)
            self._play_action(computer_action)


        return self._get_state()

    def is_game_over(self) -> tuple[bool, int]:
        if self.deck.is_empty():
            if len(self.player.hand) == 0 and len(self.computer.hand) == 0:
                return True, 0
            if len(self.player.hand) == 0 and self.state:
                return True, 1
            if len(self.computer.hand) == 0 and not self.state:
                return True, -1
            return False, 0
        else:
            return False, 0

    def put_card(self, card: Card, position: int) -> None:
        for idx in range(6):
            if self.table_cards[idx][position] is None:
                self.table_cards[idx][position] = card
                break

    def _play_action(self, action: Card) -> None:
        if self.state and self.action:
            if self.player.is_hand_empty() or self.computer.is_hand_empty() or self.table_cards[-1][0] is not None:
                self.table_cards = [[None, None] for _ in range(6)]
                self.deck.update_player_cards(self.player)
                self.deck.update_player_cards(self.computer)
                self.state = False
                self.action = True
            else:
                if action:
                    self.put_card(action, 0)
                    self.state = False
                    self.action = False
                else:
                    self.state = False
                    self.action = True
                    self.table_cards = [[None, None] for _ in range(6)]
                    self.deck.update_player_cards(self.player)
                    self.deck.update_player_cards(self.computer)
        elif self.state and not self.action:
            if action:
                self.put_card(action, 1)
                self.state = False
                self.action = True
            else:
                self.player.get_table_cards(self.table_cards)
                self.table_cards = [[None, None] for _ in range(6)]
                self.state = False
                self.action = True
                self.deck.update_player_cards(self.computer)
                self.deck.update_player_cards(self.player)
        elif not self.state and self.action:
            if self.player.is_hand_empty() or self.table_cards[-1][0] is not None:
                self.state = True
                self.action = True
                self.table_cards = [[None, None] for _ in range(6)]
                self.deck.update_player_cards(self.computer)
                self.deck.update_player_cards(self.player)
            else:
                if action:
                    self.put_card(action, 0)
                    self.state = True
                    self.action = False
                else:
                    self.state = True
                    self.action = True
                    self.table_cards = [[None, None] for _ in range(6)]
                    self.deck.update_player_cards(self.computer)
                    self.deck.update_player_cards(self.player)
        elif not self.state and not self.action:
            if action:
                self.put_card(action, 1)
                self.state = True
                self.action = True
            else:
                self.computer.get_table_cards(self.table_cards)
                self.table_cards = [[None, None] for _ in range(6)]
                self.state = True
                self.action = True
                self.deck.update_player_cards(self.player)
                self.deck.update_player_cards(self.computer)

    def step(self, action: Card) -> Tuple[Tuple, int, bool]:
        self._play_action(action)

        if not self.is_game_over()[0]:
            computer_action = self.computer.get_action(self.trump, self.action, self.table_cards, self.deck.size, self.computer.hand_size, self.computer.hand)
            self._play_action(computer_action)

        return self._get_state()
