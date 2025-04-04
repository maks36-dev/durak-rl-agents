import random
from src.rl.utils.converter import convert_to_tensor


class DQNMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data = []

    def put(self, state, action, reward, next_state, done):
        # valid_actions_mask = valid_actions(*state)
        action = action.id if action else 36
        state = convert_to_tensor(*state)
        next_state = convert_to_tensor(*next_state)

        self.data.append((state, action, reward, next_state, done))
        if self.size > self.batch_size * 300:
            self.data.pop(0)

    def get_batch(self):
        return random.sample(self.data, self.batch_size)

    @property
    def size(self):
        return len(self.data)

class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data = []

    def put(self, state, action, reward, done):
        action = action.id if action else 36
        state = convert_to_tensor(*state)

        self.data.append((state, action, reward, done))
        if self.size > self.batch_size * 300:
            self.data.pop(0)

    def get_batch(self):
        return random.sample(self.data, self.batch_size)

    @property
    def size(self):
        return len(self.data)
