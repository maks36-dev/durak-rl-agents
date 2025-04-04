import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from src.alpha_zero.accuracy import accuracy
from src.alpha_zero.agent import AlphaZeroMCTS, encode_observation, PolicyValueNet
from src.alpha_zero.env import DurakEnv


def play_one_game(env: DurakEnv, mcts: AlphaZeroMCTS, temperature=1.0, train_data=None):
    """
    Self-play.
    Every step:
      - MCTS search
      - choose action ~ pi^temp
      - step
      - save (obs, pi, _) в train_data
    Convert train_data in (obs, pi, z).
    """

    obs, payoff, done = env.reset()

    trajectory = []
    while not done:
        pi_dict = mcts.search(env)
        if pi_dict is None:
            break

        pi_array = np.zeros(37, dtype=np.float32)
        for a, prob in pi_dict.items():
            idx = 36 if (a is None) else a
            pi_array[idx] = prob

        s = encode_observation(obs, env.hands[1 - env.state])

        if temperature < 1e-6:
            a_idx = np.argmax(pi_array)
        else:
            pi_exp = pi_array ** (1.0 / temperature)
            pi_exp_sum = np.sum(pi_exp)
            if pi_exp_sum < 1e-8:
                a_idx = np.argmax(pi_array)
            else:
                pi_norm = pi_exp / pi_exp_sum
                a_idx = np.random.choice(np.arange(37), p=pi_norm)

        if a_idx == 36:
            action = None
        else:
            action = a_idx

        obs, payoff, done = env.step(action)

        trajectory.append((s, pi_array, env.state))

    z_final = 0.0
    if payoff > 0:
        z_final = +1.0
    elif payoff < 0:
        z_final = -1.0

    for (s, pi, who) in trajectory:
        if who == 0:
            zz = z_final
        else:
            zz = -z_final

        if train_data is not None:
            train_data.append((s, pi, zz))


def train_policy_value_net(policy_value_net: PolicyValueNet, train_data, batch_size=32, epochs=5, lr=1e-3):
    optimizer = optim.Adam(policy_value_net.parameters(), lr=lr)

    states = []
    targets_pi = []
    targets_v = []
    for (s, pi, z) in train_data:
        states.append(s)
        targets_pi.append(pi)
        targets_v.append(z)

    states_t = torch.FloatTensor(states)
    policy_t = torch.FloatTensor(targets_pi)
    values_t = torch.FloatTensor(targets_v)

    dataset = torch.utils.data.TensorDataset(states_t, policy_t, values_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    n = 0
    for epoch in range(epochs):
        for batch_states, batch_pi, batch_values in dataloader:
            optimizer.zero_grad()

            policy_logits, value_pred = policy_value_net.forward(batch_states)

            # ====== Policy Loss ======
            logp = F.log_softmax(policy_logits, dim=1)
            loss_p = -(batch_pi * logp).sum(dim=1).mean()

            # ====== Value Loss ======
            loss_v = F.mse_loss(value_pred, batch_values)

            # ====== Total sum ======
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / n

def train(num_iterations=10, games_per_iteration=100):
    env = DurakEnv()
    policy_value_net = PolicyValueNet(state_dim=154, action_size=37)
    mcts = AlphaZeroMCTS(policy_value_net, c_puct=1.5, num_simulations=50)

    # Parameters
    num_iterations = num_iterations
    games_per_iteration = games_per_iteration
    batch_size = 1000
    epochs = 50
    lr = 1e-3
    train_data = []

    for it in range(num_iterations):
        for _ in tqdm(range(games_per_iteration), desc=f"Num of games in epoch {it}"):
            play_one_game(env, mcts, temperature=1.0, train_data=train_data)

        loss = train_policy_value_net(policy_value_net, train_data, batch_size=batch_size, epochs=epochs, lr=lr)
        score = accuracy(mcts, num_games=10)

        if it % 10 == 0:
            torch.save(policy_value_net.state_dict(), r"src\alpha_zero\models\alpha_zero_agent.pth")


if __name__ == "__main__":
    train(1000, 1)
