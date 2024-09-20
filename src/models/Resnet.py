import torch.nn as nn
import torchvision

from src.models.Basicblock import BasicBlock


class Resnets(nn.Module):
    def __init__(self, version):
        super(Resnets, self).__init__()
        self.version = version

        match self.version:
            case 18:
                self.transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
                self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                
                self.block1 = BasicBlock(hidden_units=32, in_channels=3, out_channels=64)
                self.block2 = BasicBlock(hidden_units=64, in_channels=64, out_channels=128)
                                
                self.resnet.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features=512, out_features=2)
                )
            case 50:
                self.transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
                self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
                                
                self.resnet.fc = nn.Linear(in_features=2048, out_features=101)
            case _:
                raise ValueError("Resnets에 존재하지 않는 버전입니다.")

    def forward(self, x):
        x = self.transform(x)

        # x = self.block1(x)
        # x = self.block2(x)

        x = self.resnet(x)
        return x
    
    def get_model_name(self):
        return f"Resnet_{self.version}"

# # class ResNet18(nn.Module):
# #     def __init__(self):
# #         super().__init__()
        
# #     def forward(self, x):
# #         x = self.transform(x)
# #         x = self.resnet(x)
# #         return x


# # class ResNet50(nn.Module):
# #     def __init__(self):
# #         super().__init__()
        
# import torch
# class ResNet18(nn.Module):
#     def __init__(self, hidden_units: int) -> None:
#         super().__init__()
#         self.conv_block_1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=hidden_units, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.conv_block_2 = nn.Sequential(
#             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=hidden_units * 56 * 56, out_features=2)
#         )
    
#     def forward(self, x: torch.Tensor):
#         x = self.conv_block_1(x)
#         x = self.conv_block_2(x)
#         x = self.classifier(x)
#         return x
