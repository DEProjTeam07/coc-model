import torch
import torch.nn as nn
import torchvision


class Efficientnets(nn.Module):
    def __init__(self, version):
        super(Efficientnets, self).__init__()
        self.version = version

        match self.version:
            case 0:
                self.transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
                self.effi = torchvision.models.efficientnet(wights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            case 1:
                self.transform = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1.transforms()
                self.effi = torchvision.models.efficientnet(wights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            case 2:
                self.transform = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()
                self.effi = torchvision.models.efficientnet(wights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            case _:
                raise ValueError("Efficientnets에 존재하지 않는 버전입니다.")
        
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 2)

    def forward(self, x):
        x = self.transform(x)
        x = self.effi(x)
        return x