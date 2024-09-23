import os
import torch
import mlflow
from dotenv import load_dotenv

from src.models.Efficientnets import Efficientnets
# from src.models.Resnet import Resnets
# from src.models.TinyVGG import TinyVGG
# from src.models.CNN import CNN

#변수 로드
load_dotenv(dotenv_path= '/home/ubuntu/coc-model/.env',
            verbose= True,)
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)

#모델 선정
model = Efficientnets(version=0)
# model = Resnets(version=50)
# model = TinyVGG(32)
# model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

lr = 0.001

from src.Train import run_training

run_training(model, device, AWS_BUCKET_NAME, 'split_1', 10, learning_rate=0.001)
