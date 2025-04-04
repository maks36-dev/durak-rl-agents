import math
import copy
from typing import Dict

import torch
import torch.nn as nn

from src.alpha_zero.env import *


class PolicyValueNet(nn.Module):
    
    def __init__(self, state_dim=154, action_size=37):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.policy_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))

        policy_logits = self.policy_head(h) 
        value = torch.tanh(self.value_head(h))

        return policy_logits, value.squeeze(-1)


class AlphaZeroNode:
    def __init__(self, obs_key):
        self.obs_key = obs_key

        self.P: Dict[Optional[int], float] = {}
        self.N: Dict[Optional[int], int] = {}
        self.W: Dict[Optional[int], float] = {}
        self.Q: Dict[Optional[int], float] = {}

        self.is_expanded = False
        self.value_estimate = 0.0

    def total_visits(self):
        return sum(self.N[a] for a in self.N)


class AlphaZeroMCTS:

    def __init__(self, policy_value_net: PolicyValueNet, c_puct: float = 1.5, num_simulations: int = 50, k: int = 1):
        self.net = policy_value_net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.k = k

        self.nodes: Dict[Tuple, AlphaZeroNode] = {}
        self.root_node = None

    def act(self, env: DurakEnv):
        """Make action in game"""
        pi = self.search(env)

        action = None
        prob = 0

        for key, value in pi.items():
            if value >= prob:
                action = key
        return action

    def search(self, env: DurakEnv):
        """Search like in MCTS"""
        obs, payoff, done = env.get_current_obs()
        
        if done:
            return None

        self.root_node = self._get_node(obs)
        if not self.root_node.is_expanded:
            self._expand(self.root_node, env)

        for _ in range(self.num_simulations):
            env_copy = copy.deepcopy(env)
            self._simulate(env_copy)

        sum_n = sum(self.root_node.N[a] for a in self.root_node.N)
        if sum_n == 0:
            pi = {a: 1.0 / len(self.root_node.N) for a in self.root_node.N}
        else:
            pi = {a: (self.root_node.N[a] / sum_n) for a in self.root_node.N}
        return pi

    def _simulate(self, env_copy: DurakEnv):
        """
        One iteration MCTS: selection -> expansion -> evaluate -> backprop
        :return z - {-1,0,+1}
        """
        path = []
        while True:
            obs, payoff, done = env_copy.get_current_obs()
            if done:
                # payoff>0 => win p0, <0 => win p1
                z = self._payoff_to_z(payoff)
                break

            node = self._get_node(obs)
            if not node.is_expanded:
                self._expand(node, env_copy)
                z = node.value_estimate
                break

            action = self._select_action(node)
            path.append((node, action))

            env_copy.step(action)

        self._backprop(path, z)
        return z

    def _expand(self, node: AlphaZeroNode, env_copy: DurakEnv):
        """
        Expand node using neural network:
        """

        # Get policy_logits and value_pred
        obs = node.obs_key
        opp_hand_variants = env_copy.sample_opp_hand(self.k)
        state_vec = np.array([encode_observation(obs, opp_hand) for opp_hand in opp_hand_variants])
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)  # [1, state_dim]

        with torch.no_grad():
            policy_logits, value_pred = self.net(state_tensor)

        argmax = torch.argmax(value_pred)
        value_pred = value_pred[argmax]
        policy_logits = policy_logits[argmax]

        policy_logits = policy_logits[0].cpu().numpy()
        v = float(value_pred[0].cpu().numpy())

        # Convert logits in distribution (softmax) for valid_actions
        valid_acts = valid_actions(env_copy)
        mask = np.zeros(37, dtype=np.float32)
        for a in valid_acts:
            if a is None:
                idx = 36
            else:
                idx = a
            mask[idx] = 1.0

        max_logit = np.max(policy_logits) 
        exp_logits = np.exp(policy_logits - max_logit) * mask
        sum_exp = np.sum(exp_logits)
        if sum_exp > 1e-6:
            pi = exp_logits / sum_exp
        else:
            pi = np.zeros_like(exp_logits)
            n_valid = len(valid_acts)
            for a in valid_acts:
                idx = 36 if a is None else a
                pi[idx] = 1.0 / n_valid

        # Save P[a], value_estimate
        node.is_expanded = True
        node.value_estimate = v
        for a in valid_acts:
            if a not in node.N:
                node.N[a] = 0
                node.W[a] = 0.0
                node.Q[a] = 0.0

        for a in node.N:
            idx = 36 if a is None else a
            node.P[a] = pi[idx]

    def _select_action(self, node: AlphaZeroNode) -> Optional[int]:
        """Select action using UCT"""

        sN = node.total_visits() + 1e-8
        best_a = None
        best_score = -float('inf')
        for a in node.N:
            q = node.Q[a]
            p = node.P[a]
            na = node.N[a]
            u = self.c_puct * p * math.sqrt(sN) / (1 + na)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _backprop(self, path: List, z: float):
        """
        path: list of (node, action)
        z: float in range [-1..+1]
        """

        for (node, a) in path:
            node.N[a] += 1
            node.W[a] += z
            node.Q[a] = node.W[a] / node.N[a]

    def _get_node(self, obs_key) -> AlphaZeroNode:
        """Get node by observation"""

        if obs_key not in self.nodes:
            self.nodes[obs_key] = AlphaZeroNode(obs_key)
        return self.nodes[obs_key]

    def _payoff_to_z(self, payoff: float) -> float:
        """Who win"""

        if payoff > 0: # win player0
            return +1.0
        elif payoff < 0: # win player1
            return -1.0
        else: # draw
            return 0.0
