from src.rl.players.ppo_player import PPOPlayer
from src.rl.players.dqn_player import DQNPlayer
from src.rl.env import DurakEnv


epoch_result = {"player": 0, "computer": 0, "draw": 0}

if __name__ == "__main__":
    n_games = 100

    agent1 = PPOPlayer(model_path=r"C:\src\rl\models\ppo.pth")
    agent2 = DQNPlayer(model_path=r"C:\src\rl\models\1head_180epochs.pth")

    for _ in (range(n_games)):

        agent1.reset()
        agent2.reset()

        env = DurakEnv(agent1, agent2)

        state, reward, done = env.reset()
        for r in range(100):
            action = agent1.get_action(*state)
            next_state, reward, done = env.step(action)

            state = next_state

            if done:
                break

        winner = 'player' if reward == 1 else ('computer' if reward == -1 else 'draw')
        epoch_result[winner] += 1

    epoch_result["player"] /= n_games
    epoch_result["computer"] /= n_games
    epoch_result["draw"] /= n_games

    print(epoch_result)