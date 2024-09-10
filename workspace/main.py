import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


from src.S3ImageDatasets import S3ImageDatasets
#from src.models.Efficientnets import Efficientnets
from src.models.Resnet import ResNet18
from src.Train import train_model

AWS_BUCKET_NAME = 'deprojteam07-labeledrawdata'

dataset = S3ImageDatasets(bucket_name=AWS_BUCKET_NAME,usage='train')

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

dataset = S3ImageDatasets(bucket_name=AWS_BUCKET_NAME,usage='test')

test_loader = DataLoader(dataset, batch_size=16, shuffle=True)



#model = Efficientnets()
model = ResNet18()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # weight decay 추가

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

trained_model = train_model(model=model, device=device, criterion=criterion, optimizer=optimizer, train_loader=train_loader,
            test_loader=test_loader)
