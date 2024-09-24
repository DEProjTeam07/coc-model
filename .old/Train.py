import mlflow
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from mlflow.exceptions import MlflowException
from sklearn.metrics import precision_score, recall_score, f1_score

from src.S3ImageDatasets import build_set_loaders

#모델학습
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
    
#모델 평가
def test_model(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_predictions =[]

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total

    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return test_loss, test_acc, precision, recall, f1
    
def run_training(model, device, bucket_name, version, epochs, learning_rate):
    train_dataset, test_dataset, train_loader, test_loader = build_set_loaders(bucket_name=bucket_name, version=version)


    valid_loss_min = np.inf
    best_accuracy = 0  # 이전 최고 정확도

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)  
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1)
    
    model_name = model.get_model_name()
    
    # 이전 모델 기록 가져오기
    previous_model_runs = mlflow.search_runs(order_by=["metrics.test_accuracy desc"], filter_string=f"tags.model_name='{model_name}'")
    if not previous_model_runs.empty:
        best_accuracy = previous_model_runs.iloc[0]['metrics.test_accuracy']

    with mlflow.start_run() as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n--------------------------")
            train_model(model, device, criterion, optimizer, train_loader)
            test_loss, test_acc, precision, recall, f1 = test_model(model, device, criterion, test_loader)

            mlflow.log_metric('test_loss', test_loss)
            mlflow.log_metric('test_accuracy', test_acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1)

            if test_acc > best_accuracy:  # 이전 최고 정확도보다 높으면
                print("검증 정확도가 증가 ------------ 모델 정확도를 저장합니다")
                print(f"new best test accuracy: {test_acc}")
                mlflow.log_metric('best_accuracy', test_acc)

                # 이전 최고 정확도를 갱신
                best_accuracy = test_acc
                
                # 모델을 로그합니다
                mlflow.pytorch.log_model(model, "model")
            
            scheduler.step()

        # 마지막 모델 로그
        mlflow.pytorch.log_model(model, "final_model")
        
        model_uri = f"runs:/{run.info.run_id}/final_model"
        model_details = mlflow.register_model(model_uri, model_name)
        # 등록된 모델에 "champion" 태그 추가
        mlflow.set_tag(model_details.name, "alias", "champion")  
        print(f"Model successfully registered at {model_uri}")
        # try:
        #     # 최고 정확도일 경우 모델 등록
        #     if test_acc > best_accuracy:
        #         model_details = mlflow.register_model(model_uri, model_name)
        #         # 등록된 모델에 "champion" 태그 추가
        #         mlflow.set_model_tag(model_details.name, "alias", "champion")  
        #         print(f"Model successfully registered at {model_uri}")
        # except MlflowException as e:
        #     print(f"Model registration failed: {e}")

        print("Done!")            
        mlflow.end_run()




# #에포크별로 학습 진행
# def run_training(model, device, bucket_name, version, epochs, learning_rate):
#     _, train_loader, test_loader = build_set_loaders(bucket_name=bucket_name, version=version)

#     valid_loss_min = np.inf
#     best_acc = 0
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)  # weight decay 추가
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1)
    
#     model_name = model.get_model_name()

#     #이전 모델 최고 정확도 가져오기
#     previous_model_runs = mlflow.search_runs(order_by=["test_accuracy desc"], filter_string=f"tags.model_name='{model_name}'")
#     if not previous_model_runs.empty:
#         best_accuracy = previous_model_runs.iloc[0]['test_accuracy']

#     #mlflow로 메트릭 등 수집
#     with mlflow.start_run( ) as run:
#         mlflow.log_param("epochs", epochs)
#         mlflow.log_param("learning_rate", learning_rate)
        
#         for epoch in range(epochs):
#             print(f"Epoch {epoch+1}\n--------------------------")
#             train_model(model, device, criterion, optimizer, train_loader)
#             test_loss, test_acc, precision, recall, f1 = test_model(model, device, criterion, optimizer, test_loader)

#             mlflow.log_metric('Test Loss', test_loss)
#             mlflow.log_metric('Test Accuracy', test_acc)
#             mlflow.log_metric('Precision', precision)
#             mlflow.log_metric('Recall', recall)
#             mlflow.log_metric('F1', f1)

#             if test_loss <= valid_loss_min:
#                 print("검증 손실값 감소 ------------ 모델 가중치를 저장합니다")
#                 print(f"new best test loss : {test_loss}")
#                 mlflow.log_metric('best_loss', test_loss)

#                 valid_loss_min = test_loss

#                 mlflow.pytorch.log_model(model, "model")
            
#             scheduler.step()
        
#         #학습이 완료된 모델 로그
#         mlflow.pytorch.log_model(model, "final_model")
        
#         #모델이 로그 되었는지 확인
#         model_uri = f"runs:/{run.info.run_id}/final_model"
        
#         #최고 기록으로 갱신 되었으면 모델 레지스트리에 등록
#         try:
#             if test_loss < 
#             mlflow.register_model(model_uri, model_name)
#             print(f"Model successfully registerd at {model_uri}")
#         except MlflowException as e:
#             print(f"Model register faile : {e}")

#         print("Done!")            
#         mlflow.end_run()