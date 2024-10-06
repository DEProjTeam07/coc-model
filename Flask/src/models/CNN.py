import torch
import torch.nn as nn

from src.models.Basicblock import BasicBlock

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.block1 = BasicBlock(hidden_units=32, in_channels=3, out_channels=32)
        self.block2 = BasicBlock(hidden_units=128, in_channels=32, out_channels=128)
        self.block3 = BasicBlock(hidden_units=256, in_channels=128, out_channels=256)

        self.fc1 = nn.Linear(in_features=256 * 32 * 32, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def get_model_name(self):
        return "CNN"
