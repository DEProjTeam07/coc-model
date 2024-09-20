import mlflow
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from mlflow.exceptions import MlflowException


from src.S3ImageDatasets import build_set_loaders

def train_model(model, device, criterion, optimizer, train_loader) :
    size = len(train_loader.dataset)
    model.train()
    for batch, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(inputs)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
def test_model(model, device, criterion, optimizer, test_loader):
    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total

    return test_loss, test_acc    
    

def run_training(model, device, bucket_name, version, epochs, learning_rate):
    train_dataset, test_dataset, train_loader, test_loader = build_set_loaders(bucket_name=bucket_name, version=version)

    valid_loss_min = np.inf
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)  # weight decay 추가
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1)
    
    with mlflow.start_run( ) as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n--------------------------")
            train_model(model, device, criterion, optimizer, train_loader)
            test_loss, test_acc = test_model(model, device, criterion, optimizer, test_loader)

            mlflow.log_metric('test_loss', test_loss)
            mlflow.log_metric('test_accuracy', test_acc)

            if test_loss <= valid_loss_min:
                print("검증 손실값 감소 ------------ 모델 가중치를 저장합니다")
                print(f"new best test loss : {test_loss}")
                mlflow.log_metric('best_loss', test_loss)

                valid_loss_min = test_loss

                mlflow.pytorch.log_model(model, "model")
            
            scheduler.step()
        mlflow.pytorch.log_model(model, "final_model")
        
        model_uri = f"runs:/{run.info.run_id}/final_model"
        model_name = model.get_model_name()
        
        try:
            mlflow.register_model(model_uri, model_name)
            print(f"Model successfully registerd at {model_uri}")
        except MlflowException as e:
            print(f"Model register faile : {e}")

        print("Done!")            
        mlflow.end_run()
