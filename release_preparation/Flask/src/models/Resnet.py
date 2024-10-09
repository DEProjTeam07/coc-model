import torch.nn as nn
import torchvision


class Resnets(nn.Module):
    def __init__(self, version):
        super(Resnets, self).__init__()
        self.version = version

        match self.version:
            case 18:
                self.transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
                self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                
                                               
                self.resnet.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features=512, out_features=2)
                )
            case 50:
                self.transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
                self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
                              
                                
                self.resnet.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features=2048, out_features=2)
                )

            case _:
                raise ValueError("Resnets에 존재하지 않는 버전입니다.")

    def forward(self, x):
        x = self.transform(x)
        x = self.resnet(x)

        return x
    
    def get_model_name(self):
        return f"Resnet_{self.version}"