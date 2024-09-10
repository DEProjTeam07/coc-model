from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
from src.models import Resnet

model_list = []

effecientnet = EfficientNet.from_pretrained('effecientnet-b1', num_classes=2)
effecientnet_b2 = EfficientNet.from_pretrained('effecientnet-b2', num_classes=2)
effecientnet_b3 = EfficientNet.from_pretrained('effecientnet-b3', num_classes=2)

resnet = Resnet

class StackedEnsembleModel(nn.Module):
    def __init__(self, resnet, efficientnet, num_classes):
        super(StackedEnsembleModel, self).__init__()
        self.resnet = resnet
        self.efficientnet = efficientnet
        
        # ResNet과 EfficientNet의 출력이 결합될 크기
        self.stack_fc = nn.Sequential(
            nn.Linear(resnet.fc.out_features + efficientnet.classifier[1].out_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        resnet_output = self.resnet(x)
        efficientnet_output = self.efficientnet(x)

        # 두 모델의 출력을 연결
        concatenated_output = torch.cat((resnet_output, efficientnet_output), dim=1)

        # 스태킹 모델에 입력
        output = self.stack_fc(concatenated_output)
        return output

# 앙상블 모델 생성
ensemble_model = StackedEnsembleModel(resnet, efficientnet, num_classes=1000)

# 예측 수행
outputs = ensemble_model(inputs)
print(outputs.shape)
