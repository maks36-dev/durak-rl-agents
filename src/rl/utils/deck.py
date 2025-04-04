import os
import pygame
import random

CARD_WIDTH = 80
CARD_HEIGHT = 120

ASSETS_DIR = os.path.join("assets", "cards")

RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['spades', 'clubs', 'hearts', 'diamonds']
SUITS_IMAGES = {"spades": "♠", "clubs": "♣", "hearts": "♥", "diamonds": "♦"}

class Card:
    def __init__(self, rank, suit, load_image=False):
        self.rank = rank
        self.suit = suit
        self.image = None
        if load_image:
            self.load_image()

    def load_image(self):
        filename = f"{self.rank}_{self.suit}.png"
        path = os.path.join(ASSETS_DIR, filename)
        if os.path.exists(path):
            self.image = pygame.image.load(path)
            self.image = pygame.transform.scale(self.image, (CARD_WIDTH, CARD_HEIGHT))
        else:
            print(f"Не удалось найти файл: {path}")
            self.image = None

    @property
    def id(self):
        return 4 * RANKS.index(self.rank) + SUITS.index(self.suit)

    @classmethod
    def create_card(cls, id):
        if id == 36:
            return None
        return Card(RANKS[id // 4], SUITS[id % 4])

    def __gt__(self, other):
        if (self.suit == other.suit and RANKS.index(self.rank) > RANKS.index(other.rank)):
            return True
        else:
            return False

    def __repr__(self):
        return f"{self.rank}{SUITS_IMAGES[self.suit]}"

    def __eq__(self, other):
        return self.id == other.id

class Deck:
    def __init__(self, load_image=False):
        self.load_image = load_image

        self.cards = []
        self.build()
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def build(self):
        for suit in SUITS:
            for rank in RANKS:
                self.cards.append(Card(rank, suit, self.load_image))

    def draw_card(self):
        card = self.cards.pop(0) if self.cards else None
        return card

    def update_players_cards(self, player, computer):
        while len(player.hand) < 6:
            if len(self.cards) == 0:
                break
            player.take_card(self.draw_card())

        while len(computer.hand) < 6:
            if len(self.cards) == 0:
                break
            computer.take_card(self.draw_card())

    def update_player_cards(self, player):
        while len(player.hand) < 6:
            if len(self.cards) == 0:
                break
            player.take_card(self.draw_card())

    def is_empty(self):
        return len(self.cards) == 0

    @property
    def size(self):
        return len(self.cards)