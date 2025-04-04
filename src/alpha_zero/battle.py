import torch
from tqdm import tqdm

from src.alpha_zero.env import DurakEnv
from src.alpha_zero.agent import AlphaZeroMCTS, PolicyValueNet


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


def battle(num_games=100, num_simulations=100):
    env = DurakEnv()
    policy_value_net0 = PolicyValueNet(state_dim=154, action_size=37)
    policy_value_net0.load_state_dict(torch.load(r"C:\Users\Maxim\Desktop\Курсовая работа\src\alpha_zero\models\alpha_zero_agent.pth"))
    agent0 = AlphaZeroMCTS(policy_value_net0, num_simulations=num_simulations)

    policy_value_net1 = PolicyValueNet(state_dim=154, action_size=37)
    agent1 = AlphaZeroMCTS(policy_value_net1, num_simulations=num_simulations)

    agent0_win = 0
    agent1_win = 0
    for _ in tqdm(range(num_games)):
        payoff = play_one_game(env, agent0, agent1)
        if payoff>0:
            agent0_win += 1
        elif payoff<0:
            agent1_win += 1
    print(f"Result of {num_games} games: agent0 = {agent0_win}, agent1 = {agent1_win}")

def main():
    battle(num_games=100, num_simulations=50)

if __name__ == "__main__":
    main()

