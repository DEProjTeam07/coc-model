import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from tqdm.notebook import tqdm

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

def train_model(model,device, criterion, optimizer, train_loader, save_file, num_epochs=10) :
    for epoch in range(num_epochs):
        #최소 손실값 초기화
        valid_loss_min = np.inf

        #train
        model.train()
        running_loss = 0.0 # 손실값 초기화
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() #옵티마이저 내 기울기 초기화
            outputs = model(inputs) #순전파
            loss = criterion(outputs, labels) #손실값 계산
            loss.backward() #역전파
            optimizer.step() #가중치 갱신

            running_loss += loss.item() #현재 배치에서의 손실 추가

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f'/t훈련데이터 손실값: {train_loss:.4f}/t훈련데이터 정확도: {train_acc}/t epoch({epoch}/{num_epochs})')

        #최적 모델 가중치 찾기
        ## 현재 에포크에서 손실값이 최소 손실값 이하이면 모델 가중치 저장
        if train_loss <= valid_loss_min:
            print(f'/t@@@ 검증 데이터 손실값 감소 ({valid_loss_min:.4f} --> {train_loss:.4f})/n/t모델저장')

            #모델 저장
            torch.save(model.state_dict(), save_file)
            valid_loss_min = train_loss
        
    return train_losses, train_accuracies