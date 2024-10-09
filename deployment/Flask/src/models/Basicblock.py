import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, hidden_units, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_units, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_units, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    