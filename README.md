# Reinforcement learning algorithms for the "Fool" card game

This project is a study of RL algorithms for the game "Fool". Quite well-known algorithms are implemented here in order to analyze their usefulness for this game.

---

## Project Description

[The Fool Game](https://en.wikipedia.org/wiki/Durak)

The aim of this project is to explore how various reinforcement learning algorithms can be applied to the complex card game "Fool". This implementation of the game "Fool" has a number of features:
- Quite a large, incomplete and hidden set of states from the opponent;
- There are 2 players playing in this implementation;
- In one turn, you can lay out 1 card (it does not matter whether it is an attack or a defense), make a pass (in defense) or a rebound (in attack);
- In each state of the game, the player does not know the opponent's cards and the cards he will get from the deck. That's why it's an Imperfect game;
- In each state of the game, there is a certain set of "legal" actions (that is, of all the cards in his hand in a certain state, he can use a subset of them).

## Repository structure
```
.
├─ src/
│   ├─ alpha_zero/         # An agent implemented based on the Alpha Zero architecture
│   ├─ deep_cfr/           # A realizational agent based on the Deep CFR algorithm
│   ├─ llm/                # An agent using LLM to analyze the game and perform actions
│   ├─ neat/               # An agent implemented based on the NEAT algorithm
│   ├─ po_mcts/            # An agent implemented based on the MCTS algorithm with PO modification for Imperfect games
│   ├─ rl/                 # Agents implemented on the basis of DQN and PPO algorithms
│   ├─ .env
│   └─ main.py
├─ assets/                 # Images for the game
├─ requirements.txt        # List of Python dependencies
└─ README.md               # Project Description
```


