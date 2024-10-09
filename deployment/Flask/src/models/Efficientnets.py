import torch.nn as nn
import torchvision


class Efficientnets(nn.Module):
    def __init__(self, version):
        super(Efficientnets, self).__init__()
        self.version = version

        match self.version:
            case 0:
                self.effi = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            case 1:
                self.effi = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            case 2:
                self.effi = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            case _:
                raise ValueError("Efficientnets에 존재하지 않는 버전입니다.")

        self.effi.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.effi.classifier[1].in_features, 2)
        )

    def forward(self, x):
        return self.effi(x)  # 전처리 함수를 모델에 통합

    def get_model_name(self):
        return f"Efficientnet_B{self.version}"
