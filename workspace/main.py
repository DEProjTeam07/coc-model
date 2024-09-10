import torch
from torch.utils.data import DataLoader

from src.S3ImageDatasets import S3ImageDatasets

AWS_BUCKET_NAME = 'deprojteam07-labeledrawdata'

dataset = S3ImageDatasets(bucket_name=AWS_BUCKET_NAME,usage='train')

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=20)

