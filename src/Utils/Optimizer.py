from enum import Enum
import torch.optim as optim

# 옵티마이저 종류 정의
class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"

# 옵티마이저 선택
def get_optimizer(optimizer_type, model, learning_rate, weight_decay=0.0001):
    if optimizer_type == OptimizerType.SGD:
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerType.ADAM:
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == OptimizerType.RMSPROP:
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("정의되지 않은 옵티마이저 타입입니다.\n사용할 수 있는 옵티마이저 타입 : sgd, adam, rmsprop")
