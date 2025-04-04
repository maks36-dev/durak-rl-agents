import torch

from src.rl.players.base_players.base_player import BasePlayer
from src.rl.networks.ppo import PPO
from src.rl.memory import PPOMemory
from src.rl.utils.deck import Card
from src.rl.utils.converter import valid_actions, convert_to_tensor


class PPOPlayer(BasePlayer):
    def __init__(self, batch_size=1000, model_path=""):
        super().__init__()
        self.batch_size = batch_size
        self.memory = PPOMemory(batch_size)

        self.model = PPO(87, 37, gamma=0.99)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

    def save_replay(self, state, action, reward, done):
        self.memory.put(state, action, reward, int(done))

    def fit(self):
        if self.memory.size > self.batch_size:
            batch = self.memory.get_batch()
            self.model.fit(batch)

    def get_action(self, trump, action, table_cards, deck_count, opponent_hand_count, hand_cards):
        try:
            state = torch.tensor([convert_to_tensor(trump, action, table_cards, deck_count, opponent_hand_count, hand_cards)], dtype=torch.float32)
            valid_actions_mask = torch.tensor(valid_actions(trump, action, table_cards, deck_count, opponent_hand_count, hand_cards), dtype=torch.float32)
        except Exception:
            return None

        card_id = torch.argmax((torch.argsort(self.model.get_action(state)) + 1) * valid_actions_mask).item()


        card = Card.create_card(card_id)
        if card:
            self.hand.remove(card)

        return card
