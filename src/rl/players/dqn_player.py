import torch

from src.rl.networks.dqn import DQN
from src.rl.players.base_players.base_player import BasePlayer
from src.rl.memory import DQNMemory
from src.rl.utils.deck import Card
from src.rl.utils.converter import valid_actions, convert_to_tensor


class DQNPlayer(BasePlayer):
    def __init__(self, gamma=0.999, lr=0.001, batch_size=100, model_path=""):
        super().__init__()
        self.batch_size = batch_size
        self.memory = DQNMemory(batch_size)

        self.model = DQN()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.rate_model = DQN()
        self.rate_model.load_state_dict(self.model.state_dict())

    def update_weights(self):
        self.rate_model.load_state_dict(self.model.state_dict())

    def save_replay(self, state, action, reward, next_state, done):
        self.memory.put(state, action, reward, next_state, int(done))

    def fit(self):
        if self.memory.size > self.batch_size:
            batch = self.memory.get_batch()
            states, actions, rewards, next_states, dones = map(torch.FloatTensor, zip(*batch))

            actions = torch.tensor(actions, dtype=torch.int)

            with torch.no_grad():
                targets = rewards + (1 - dones) * self.gamma * torch.max(self.model(next_states), dim=1).values

            q_values = self.model(states)[torch.arange(len(batch)), actions]
            loss = torch.mean((q_values - targets) ** 2)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


    def get_action(self, trump, action, table_cards, deck_count, opponent_hand_count, hand_cards):
        state = torch.tensor([convert_to_tensor(trump, action, table_cards, deck_count, opponent_hand_count, hand_cards)], dtype=torch.float32)
        valid_actions_mask = torch.tensor(valid_actions(trump, action, table_cards, deck_count, opponent_hand_count, hand_cards), dtype=torch.float32)

        card_id = torch.argmax((torch.argsort(self.model(state))[0] + 1) * valid_actions_mask).item()

        card = Card.create_card(card_id)

        if card is not None:
            self.hand.remove(card)

        return card
