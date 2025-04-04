import torch.nn as nn
import torch


class RegretModel(nn.Module):
    def __init__(self, input_size=116, hidden_size=128, output_size=37):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        out = self.fc3(h)
        return out


class PolicyModel(nn.Module):
    def __init__(self, input_size=116, hidden_size=128, output_size=37):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        out = self.fc3(h)
        return out

