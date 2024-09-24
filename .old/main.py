import os
import torch
import mlflow
from dotenv import load_dotenv

# from src.models.Efficientnets import Efficientnets
# # from src.models.Resnet import Resnets
# from src.models.TinyVGG import TinyVGG
# # from src.models.CNN import CNN

from src.Utils.models import ModelType, get_model

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

#모델 선정
# model = Efficientnets(version=0)
# # model = Resnets(version=50)
# model = TinyVGG()
# # model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

lr = 0.001

from src.Train import run_training

run_training(model, device, AWS_BUCKET_NAME, 'split_1', 10, learning_rate=0.001)

