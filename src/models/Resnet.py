import torch.nn as nn
import torchvision

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=2)
        )
        
    def forward(self, x):
        x = self.transform(x)
        x = self.resnet(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=101)
        
    def forward(self, x):
        x = self.transform(x)
        x = self.resnet(x)
        return x