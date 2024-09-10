import torch
import torch.nn as nn
import torchvision


# class Efficientnets(nn.Module):
#     def __init__(self, version):
#         super(Efficientnets, self).__init__()
#         self.version = version        
#         # match 구문을 찾아보세용... 확장성에 용이합니당

#         if self.version == 0:
#             self.efficientnet = torchvision.models.efficientnet(wights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
#         elif self.version == 1:
#             self.efficientnet = torchvision.models.efficientnet(wights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
#         elif self.version == 2:
#             self.efficientnet = torchvision.models.efficientnet(wights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
#         else:
#             raise ValueError("Efficientnet에 존재하지 않는 버전입니다.")

#         self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 2)

#     def forward(self, x):
#         x = self.efficientnet(x)
#         return x

class Efficientnets(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        self.effib0 = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
    def forward(self, x):
        x = self.effib0(x)
        return x
