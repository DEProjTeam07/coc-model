from enum import Enum

from src.models.Efficientnets import Efficientnets
from src.models.Resnet import Resnets
from src.models.TinyVGG import TinyVGG
from src.models.CNN import CNN


class ModelType(Enum):
    EFFICIENTNET = "efficientnet"
    RESNET = "resnet"
    TINYVGG = "tinyvgg"
    CNN = "cnn"

# 모델 정의 함수
def get_model(model_type, version=None):
    if model_type == ModelType.EFFICIENTNET:
        if version is None:
            print("efficientnet에 지정되지 않은 버전입니다.\n지정된 버전 : 0, 1, 2\nefficientnet_b0으로 학습을 진행합니다.")
            version=0
        model = Efficientnets(version)
    elif model_type == ModelType.RESNET:
        if version is None:
            print("resnet에 지정되지 않은 버전입니다.\n지정된 버전 : 18, 50\nresnet 18로 학습을 진행합니다.")
            version = 18
        model = Resnets(version)
    elif model_type == ModelType.TINYVGG:
        model = TinyVGG()
    elif model_type == ModelType.CNN:
        model = CNN()
    else:
        raise ValueError("학습할 수 있는 모델이 아닙니다\n사용할 수 있는 모델 : efficientnet, resnet, tinyvgg, cnn")
    return model

