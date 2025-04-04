from src.alpha_zero.env import DurakEnv
from src.alpha_zero.greedy_agent import GreedyAgent

from tqdm import tqdm

def play_one_game(env, agent0, agent1):
    obs, payoff, done = env.reset()
    step_count = 0

    while not done:
        step_count += 1
        current_player = env.state

        if current_player == 0:
            action = agent0.act(env)
        else:
            action = agent1.act(env)

        obs, payoff, done = env.step(action)

    return payoff


def accuracy(agent0, num_games=100):
    env = DurakEnv()

    agent1 = GreedyAgent()

    agent0_win = 0
    agent1_win = 0
    for _ in tqdm(range(num_games), desc="Check accuracy"):
        payoff = play_one_game(env, agent0, agent1)
        if payoff>0:
            agent0_win += 1
        elif payoff<0:
            agent1_win += 1

    return agent0_win / num_games






