from tqdm import tqdm

from src.po_mcts.env import DurakEnv
from src.po_mcts.agent import POMCTSAgent, POMCTSNode
from src.po_mcts.greedy_player import GreedyAgent


def play_one_game(env, agent0, agent1):
    obs, payoff, done = env.reset()

    while not done:
        current_player = env.state

        if current_player == 0:
            action = agent0.act(env)
        else:
            action = agent1.act(env)

        obs, payoff, done = env.step(action)

    return payoff


def evaluate(num_games=100, num_simulations=100):
    env = DurakEnv()
    agent0 = POMCTSAgent(num_simulations=num_simulations, use_memory=True)
    agent1 = POMCTSAgent(num_simulations=num_simulations, use_memory=False)

    agent0_win = 0
    agent1_win = 0
    for _ in tqdm(range(num_games)):
        payoff = play_one_game(env, agent0, agent1)
        if payoff>0:
            agent0_win += 1
        elif payoff<0:
            agent1_win += 1

    print(f"Result (MCTS+ vs Greedy) of {num_games} games: MCTS+ = {agent0_win}, MCTS = {agent1_win}")

def main():
    evaluate(num_games=100, num_simulations=100)

if __name__ == "__main__":
    main()
