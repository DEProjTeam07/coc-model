import mlflow.pytorch
from src.Inference import infer_images_in_folder, save_to_parquet
import torch
from torch.utils.data import DataLoader


model_name = 'Efficientnet_B1'
model_version = 1

model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")

import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 로컬 데이터셋 로드
folder_path = '/home/ubuntu/coc-model/inference_dataset/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()
#추론할 데이터셋 설정 : 로컬에 저장된 폴더를 사용합니다
# folder_path = '/home/ubuntu/modeling/seowoo/coc-model/inference_dataset/'
output_file = 'inference_result.parquet'

df = infer_images_in_folder(folder_path, model,device=device)
# save_to_parquet(df, output_file)

print(df)