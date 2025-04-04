import random
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.deep_cfr.env import DurakEnv, valid_actions
from src.deep_cfr.utils import encode_observation
from src.deep_cfr.networks import RegretModel, PolicyModel


class DeepCFRAgent:
    def __init__(self,
                 num_iterations: int = 10,
                 num_traversals_per_iter: int = 1000,
                 lr: float = 1e-3):
        self.num_iterations = num_iterations
        self.num_traversals_per_iter = num_traversals_per_iter

        self.regret_net = RegretModel()
        self.policy_net = PolicyModel()

        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Buffer (features, action_idx, regret, weight).
        self.regret_buffer = []
        # Buffer (features, action_idx, prob).
        self.policy_buffer = []

    def train(self):
        for _ in tqdm(range(self.num_iterations), desc="Iterations"):

            # Get data for regret
            self.regret_buffer.clear()
            for _ in tqdm(range(self.num_traversals_per_iter), desc="Regret Sampling"):
                env = DurakEnv()
                env.reset()
                self.traverse_cfr(env, reach_prob_p0=1.0, reach_prob_p1=1.0)


            self.train_regret_model()

            # Get data for policy
            self.policy_buffer.clear()
            for _ in tqdm(range(self.num_traversals_per_iter), desc="Policy Sampling"):
                env = DurakEnv()
                env.reset()
                self.traverse_policy(env)

            self.train_policy_model()

    def traverse_cfr(self, env: DurakEnv, reach_prob_p0: float, reach_prob_p1: float):
        """Save train data for regret model"""

        if env.done:
            payoff = env.payoff
            return payoff, -payoff

        current_player = env.state

        acts = valid_actions(env)
        acts = random.choices(acts, k=1) # To test the game, instead of just bypassing the legal actions, we will take one and use it
        if not acts:
            payoff = env.payoff
            return payoff, -payoff

        obs, _, _ = env.get_current_obs()
        features = encode_observation(obs)

        with torch.no_grad():
            regrets_all = self.regret_net.forward(features.unsqueeze(0)).squeeze(0)
        regrets_all = regrets_all.tolist()  # длина 37

        action_index_map = []
        for a in acts:
            if a is None:
                action_index_map.append(36)
            else:
                action_index_map.append(a)

        regrets_filtered = [regrets_all[idx] for idx in action_index_map]
        positive_regrets = [max(r, 0.0) for r in regrets_filtered]
        sum_pos = sum(positive_regrets)
        if sum_pos > 1e-9:
            strategy = [r / sum_pos for r in positive_regrets]
        else:
            strategy = [1.0 / len(acts)] * len(acts)

        # External sampling
        cf_values_for_each_action = {}
        for i, a in enumerate(acts):
            env_copy = deepcopy(env)
            env_copy.step(a)

            if current_player == 0:
                new_reach_p0 = reach_prob_p0
                new_reach_p1 = reach_prob_p1 * strategy[i]
            else:
                new_reach_p0 = reach_prob_p0 * strategy[i]
                new_reach_p1 = reach_prob_p1

            payoff0, payoff1 = self.traverse_cfr(env_copy, new_reach_p0, new_reach_p1)
            cf_values_for_each_action[a] = (payoff0, payoff1)

        # node_value for current_player
        if current_player == 0:
            node_value = 0.0
            for i, a in enumerate(acts):
                node_value += strategy[i] * cf_values_for_each_action[a][0]
        else:
            node_value = 0.0
            for i, a in enumerate(acts):
                node_value += strategy[i] * cf_values_for_each_action[a][1]

        # Regrets
        regrets = []
        for i, a in enumerate(acts):
            cf_val = cf_values_for_each_action[a][current_player]
            regrets.append(cf_val - node_value)

        # save data in regret buffer
        weight = reach_prob_p1 if current_player == 0 else reach_prob_p0
        for (reg, a) in zip(regrets, acts):
            if a is None:
                a_idx = 36
            else:
                a_idx = a
            self.regret_buffer.append((features, a_idx, reg, weight))

        # outcome sampling using strategy
        sampled_action_idx = random.choices(range(len(acts)), weights=strategy, k=1)[0]
        chosen_action = acts[sampled_action_idx]
        return cf_values_for_each_action[chosen_action]

    def traverse_policy(self, env: DurakEnv):
        """Save data for train policy network"""

        while not env.done:
            obs, _, _ = env.get_current_obs()
            acts = valid_actions(env)
            if not acts:
                break

            features = encode_observation(obs)
            with torch.no_grad():
                logits = self.policy_net.forward(features.unsqueeze(0)).squeeze(0)  # (37,)

            action_index_map = []
            for a in acts:
                action_index_map.append(36 if a is None else a)
            chosen_logits = [logits[idx].item() for idx in action_index_map]

            max_l = max(chosen_logits)
            exps = [pow(2.71828, (l - max_l)) for l in chosen_logits]
            s = sum(exps)
            probs = [e/s for e in exps]

            for i, a in enumerate(acts):
                a_idx = 36 if a is None else a
                self.policy_buffer.append((features, a_idx, probs[i]))

            chosen_i = random.choices(range(len(acts)), weights=probs, k=1)[0]
            chosen_action = acts[chosen_i]
            env.step(chosen_action)

    def train_regret_model(self):
        if not self.regret_buffer:
            return

        batch_features = []
        batch_labels = []
        batch_action_idx = []
        batch_weights = []

        for (feat, a_idx, reg, w) in self.regret_buffer:
            batch_features.append(feat)
            batch_labels.append(reg)
            batch_action_idx.append(a_idx)
            batch_weights.append(w)

        X = torch.stack(batch_features)
        y = torch.tensor(batch_labels, dtype=torch.float32)
        a = torch.tensor(batch_action_idx, dtype=torch.long)
        wts = torch.tensor(batch_weights, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X, y, a, wts)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        self.regret_net.train()
        for _ in range(3):
            for batch_x, batch_y, batch_a, batch_w in loader:
                self.regret_optimizer.zero_grad()
                out = self.regret_net.forward(batch_x)  # (B,37)
                pred = out.gather(1, batch_a.unsqueeze(1)).squeeze(1)  # (B,)

                # Weighted MSE:
                loss_raw = (pred - batch_y)**2
                loss = (batch_w * loss_raw).mean()

                loss.backward()
                self.regret_optimizer.step()

    def train_policy_model(self):
        if not self.policy_buffer:
            return

        grouped_data = defaultdict(lambda: dict())

        for (feat, a_idx, pr) in self.policy_buffer:
            key = tuple(feat.numpy().tolist())
            grouped_data[key][a_idx] = pr

        batch_X = []
        batch_dist = []

        for key, adict in grouped_data.items():
            flist = list(key)
            batch_X.append(torch.tensor(flist, dtype=torch.float32))
            dist = [0.0]*37
            for a_idx, pr in adict.items():
                dist[a_idx] = pr
            batch_dist.append(dist)

        X = torch.stack(batch_X)
        dist_tgt = torch.tensor(batch_dist, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X, dist_tgt)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        self.policy_net.train()
        for _ in range(3):
            for bx, bd in loader:
                self.policy_optimizer.zero_grad()

                logits = self.policy_net.forward(bx)
                logp = torch.log_softmax(logits, dim=1)

                loss = nn.KLDivLoss(reduction='batchmean')(logp, bd)

                loss.backward()
                self.policy_optimizer.step()


if __name__ == "__main__":
    agent = DeepCFRAgent(num_iterations=2, num_traversals_per_iter=50, lr=1e-3)
    agent.train()