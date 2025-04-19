import numpy as np
import torch
import torch.nn as nn


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99,
                 batch_size=128, eps_clip=0.2,
                 pi_lr=3e-4, v_lr=1e-3, epoch_n=5):
        super().__init__()

        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        self.v_model = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.gamma, self.eps_clip = gamma, eps_clip
        self.batch_size, self.epoch_n = batch_size, epoch_n
        self.pi_opt = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_opt  = torch.optim.Adam(self.v_model.parameters(),  lr=v_lr)

    def get_action(self, state, greedy=False):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.pi_model(state)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.probs.argmax() if greedy else dist.sample()
        return action

    def fit(self, traj):
        states, actions, rewards, dones = map(np.array, zip(*traj))
        rewards, dones = rewards.reshape(-1,1), dones.reshape(-1,1)

        returns = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(traj))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        states  = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        returns = torch.tensor(returns, dtype=torch.float32)

        with torch.no_grad():
            logits = self.pi_model(states)
            dist   = torch.distributions.Categorical(logits=logits)
            old_log_probs = dist.log_prob(actions)
            values = self.v_model(states).squeeze()
            advantages = (returns.squeeze() - values).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epoch_n):
            idx = np.random.permutation(len(traj))
            for start in range(0, len(traj), self.batch_size):
                batch = idx[start:start + self.batch_size]
                b_s, b_a, b_rt = states[batch], actions[batch], returns[batch].squeeze()
                b_adv, b_old = advantages[batch], old_log_probs[batch]

                logits = self.pi_model(b_s)
                dist   = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_a)
                ratio = (new_log_probs - b_old).exp()

                # clipped surrogate
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * b_adv
                pi_loss = -torch.min(surr1, surr2).mean()  \
                          - 0.01 * dist.entropy().mean()

                self.pi_opt.zero_grad()
                pi_loss.backward()
                self.pi_opt.step()

                # value loss
                v_loss = ((self.v_model(b_s).squeeze() - b_rt)**2).mean()
                self.v_opt.zero_grad()
                v_loss.backward()
                self.v_opt.step()