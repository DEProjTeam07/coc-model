import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.S3ImageDatasets import S3ImageDatasets
from src.models.Efficientnets import Efficientnets
from src.TrainValidateEvaluation import train_model
from src.Inference import infer_images_in_folder, save_to_parquet

# AWS_BUCKET_NAME = 'deprojteam07-labeledrawdata'

# #학습 및 평가용 데이터셋 로드
# train_dataset = S3ImageDatasets(bucket_name=AWS_BUCKET_NAME,version='split_1',usage='train')
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# test_dataset = S3ImageDatasets(bucket_name=AWS_BUCKET_NAME,version='split_1',usage='test')
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


# #모델 선정
model = Efficientnets(version=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# #모델 학습
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # weight decay 추가

# train_losses = []
# test_losses = []
# train_accuracies = []
# test_accuracies = []

# trained_model = train_model(model=model, device=device, criterion=criterion, optimizer=optimizer, train_loader=train_loader,
#             test_loader=test_loader, save_file='/home/ubuntu/model_registry/EffiB1.pth')

#학습된 모델 로드
model.load_state_dict(torch.load('/home/ubuntu/model_registry/EffiB1.pth', map_location=device))
model = model.to(device)
model.eval()
#추론할 데이터셋 설정 : 로컬에 저장된 폴더를 사용합니다
folder_path = '/home/ubuntu/modeling/seowoo/coc-model/inference_dataset/'
output_file = 'inference_result.parquet'

df = infer_images_in_folder(folder_path, model,device=device)
save_to_parquet(df, output_file)

