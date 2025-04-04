import random

from tqdm import tqdm

from src.rl.players.base_players.greedy_player import GreedyPlayer
from src.rl.env import DurakEnv
from copy import deepcopy

def train_dqn(model, computer):
    epochs = 10
    games = 100

    player = model
    computer = computer

    epsilon = 1
    epsilon_step = 0.005
    epsilon_min = 0.001
    eps_player = GreedyPlayer()

    env = DurakEnv(player, computer)

    for _ in tqdm(range(epochs)):
        for game in range(games):
            player.reset()
            computer.reset()
            state, reward, done = env.reset()

            for r in range(100):
                old_state = deepcopy(state)
                if random.random() < epsilon and False:
                    eps_player.hand = player.hand
                    action = eps_player.get_action(*state)
                    player.hand = eps_player.hand
                else:
                    action = player.get_action(*state)
                next_state, reward, done = env.step(action)

                player.save_replay(old_state, action, reward, next_state, done)

                state = next_state


                if done:
                    break
            epsilon = max(epsilon - epsilon_step, epsilon_min)

            if game % 10 == 0:
                player.fit()

            if game % 30 == 0:
                player.update_weights()


def train_ppo(player, computer, epsilon_step=0.005, epsilon_min=0.001):
    games = 1000

    epsilon = 1
    eps_player = GreedyPlayer()

    env = DurakEnv(player, computer)

    for game in tqdm(range(games)):
        player.reset()
        computer.reset()

        state, reward, done = env.reset()
        for r in range(100):
            old_state = deepcopy(state)

            if random.random() < epsilon and False:
                eps_player.hand = player.hand
                action = eps_player.get_action(*state)
                player.hand = eps_player.hand
            else:
                action = player.get_action(*state)

            next_state, reward, done = env.step(action)
            player.save_replay(old_state, action, reward, done)

            state = next_state

            if done:
                break
        if game % 10 == 0:
            player.fit()

        epsilon = max(epsilon - epsilon_step, epsilon_min)
