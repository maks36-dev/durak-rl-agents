from src.rl.train import train_ppo, train_dqn
from src.rl.env import DurakEnv

epochs_result = {"player": [], "computer": [], "draw": []}
who_start = {"player": 0, "computer": 0}

def experiment(player, computer):
    epoch_result = {"player": 0, "computer": 0, "draw": 0}
    n_games = 100

    agent1 = player
    agent2 = computer

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

    return epoch_result


def accuracy_dqn(player, computer):
    global epochs_result

    train_dqn(player, computer)
    epoch_result = experiment(player, computer)

    epochs_result["player"].append(epoch_result["player"])
    epochs_result["computer"].append(epoch_result["computer"])
    epochs_result["draw"].append(epoch_result["draw"])

    return epochs_result


def accuracy_ppo(player, computer):
    global epochs_result

    train_ppo(player, computer)
    epoch_result = experiment(player, computer)

    epochs_result["player"].append(epoch_result["player"])
    epochs_result["computer"].append(epoch_result["computer"])
    epochs_result["draw"].append(epoch_result["draw"])

    return epochs_result