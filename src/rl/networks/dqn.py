import torch.nn as nn

INPUT_SIZE = 87
NUM_CLASSES = 37

class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input_layer = nn.Linear(INPUT_SIZE, 512)
        self.hidden_1 = nn.Linear(512, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.hidden_3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, NUM_CLASSES)

        self.activation = nn.ReLU()


    def forward(self, input_state):
        x = self.activation(self.input_layer(input_state))
        x = self.activation(self.hidden_1(x))
        x = self.activation(self.hidden_2(x))
        x = self.activation(self.hidden_3(x))
        return self.output_layer(x)
