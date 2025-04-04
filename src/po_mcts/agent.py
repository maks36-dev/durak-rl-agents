import math
import random
from copy import deepcopy
import pickle
from typing import Dict, Optional
from tqdm import tqdm

from src.po_mcts.env import DurakEnv, valid_actions
from src.po_mcts.utils import *


class POMCTSNode:
    def __init__(self, observation, node_owner: int):
        self.observation = observation
        self.node_owner = node_owner
        self.N: Dict[Optional[int], int] = {}
        self.W: Dict[Optional[int], float] = {}
        self.Q: Dict[Optional[int], float] = {}
        self.children: Dict[Optional[int], 'POMCTSNode'] = {}
        self.is_expanded = False

    def update_stats(self, action: Optional[int], value: float):
        if action not in self.N:
            self.N[action] = 0
            self.W[action] = 0.0
            self.Q[action] = 0.0
        self.N[action] += 1
        self.W[action] += value
        self.Q[action] = self.W[action] / self.N[action]

    def total_visits(self) -> int:
        return sum(self.N[a] for a in self.N)

    def select_action_ucb(self, c_puct=1.4) -> Optional[int]:
        best_a = None
        best_ucb = -999999.0
        sN = self.total_visits() + 1e-8
        for a in self.N:
            q = self.Q[a]
            n_a = self.N[a]
            ucb = q + c_puct * math.sqrt(math.log(sN + 1) / (1 + n_a))
            if ucb > best_ucb:
                best_ucb = ucb
                best_a = a
        return best_a


class POMCTSAgent:
    def __init__(self, num_simulations=200, c_puct=1.4, use_memory=True):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.use_memory = use_memory

        self.tree: Dict[Tuple[Tuple, int], POMCTSNode] = {}

    def _get_node(self, obs: Tuple, node_owner: int) -> POMCTSNode:
        key = (obs, node_owner)
        if key not in self.tree:
            self.tree[key] = POMCTSNode(obs, node_owner)
        return self.tree[key]

    def act(self, env: DurakEnv) -> Optional[int]:
        if env.done:
            return None
        obs, payoff, done = env.get_current_obs()
        if done:
            return None

        node_owner = env.state
        root_node = self._get_node(obs, node_owner)

        if not root_node.is_expanded:
            self._expand(root_node, env)

        for _ in range(self.num_simulations):
            env_copy = deepcopy(env)
            self._sample_full_state(env_copy)
            self._simulate(env_copy)

        best_a = None
        best_visits = -1
        for a in root_node.N:
            if root_node.N[a] > best_visits:
                best_visits = root_node.N[a]
                best_a = a
        return best_a

    def _simulate(self, env_copy: DurakEnv) -> float:
        path = []
        while True:
            obs, payoff, done = env_copy.get_current_obs()
            if done:
                break

            node_owner = env_copy.state
            node = self._get_node(obs, node_owner)
            if not node.is_expanded:
                self._expand(node, env_copy)
                payoff = self._rollout(env_copy)
                break

            action = node.select_action_ucb(self.c_puct)
            path.append((node, action))
            env_copy.step(action)

        # Backprop
        for (node, action) in path:
            if payoff > 0:
                val = +1.0 if node.node_owner == 0 else -1.0
            elif payoff < 0:
                val = +1.0 if node.node_owner == 1 else -1.0
            else:
                val = 0.0
            node.update_stats(action, val)

        return payoff

    def _expand(self, node: POMCTSNode, env: DurakEnv):
        va = valid_actions(env)
        for a in va:
            node.N[a] = 0
            node.W[a] = 0.0
            node.Q[a] = 0.0
        node.is_expanded = True

    def _rollout(self, env_copy: DurakEnv) -> float:
        while not env_copy.done:
            obs, payoff, done = env_copy.get_current_obs()
            if done:
                return payoff
            va = valid_actions(env_copy)
            if len(va) == 0:
                env_copy._check_end()
                return env_copy.payoff
            action = random.choice(va)
            env_copy.step(action)
        return env_copy.payoff

    def _sample_full_state(self, env_copy: DurakEnv):
        current_player = env_copy.state
        opp_idx = 1 - current_player

        used = set()
        for c in env_copy.hands[current_player]:
            used.add(c)
        for (ac, dc) in env_copy.table_cards:
            if ac is not None:
                used.add(ac)
            if dc is not None:
                used.add(dc)
        for c in env_copy.discard:
            used.add(c)

        all_cards = set(range(36))

        if self.use_memory:
            candidates = list(env_copy.deck + env_copy.hands[opp_idx])
        else:
            candidates = list(all_cards - used)
        random.shuffle(candidates)

        opp_count = len(env_copy.hands[opp_idx])
        deck_count = len(env_copy.deck)

        new_opp = candidates[:opp_count]
        new_deck = candidates[opp_count:opp_count + deck_count]

        env_copy.hands[opp_idx] = list(new_opp)
        env_copy.deck = list(new_deck)


def main():
    env = DurakEnv()
    agent0 = POMCTSAgent(num_simulations=10)
    agent1 = POMCTSAgent(num_simulations=10)

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


def train(agent: POMCTSAgent, epochs: int = 1000, games: int=100):
    env = DurakEnv()

    for it in tqdm(range(epochs * games), desc="Games"):
        obs, payoff, done = env.reset()

        while not done:
            current_player = env.state
            if current_player == 0:
                action = agent.act(env)
            else:
                action = agent.act(env)

            obs, payoff, done = env.step(action)

        if it % 10 == 0:
            save_agent(agent)

def save_agent(agent: POMCTSAgent):
    with open("agents/agent.pkl", "wb") as file:
        pickle.dump(agent, file)

if __name__ == "__main__":
    agent = POMCTSAgent(num_simulations=1000, use_memory=True)
    # train(agent, epochs=5)

