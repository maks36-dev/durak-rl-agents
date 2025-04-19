import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

from src.rl.players.base_players.greedy_player import GreedyPlayer
from src.rl.players.ppo_player import PPOPlayer
from src.rl.utils.accuracy import accuracy_ppo

fig, ax = plt.subplots(1, 1, figsize=[10, 5])

model = PPOPlayer(10000)
computer = GreedyPlayer()


def update(frame):
    epochs_result = accuracy_ppo(model, computer)

    ax.clear()

    # plot data
    ax.plot(epochs_result["player"], color='r')


    # set options
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_xlabel('Epochs', fontweight='bold', fontsize=15)
    ax.set_ylabel('Win Rate', fontweight='bold', fontsize=15)

    fig.canvas.draw()

    if frame % 10 == 0:
        torch.save(model.model.state_dict(), r"\src\checkpoint_ppo.pth")


anim = FuncAnimation(fig, update, frames=range(300), repeat=False)
plt.show()

