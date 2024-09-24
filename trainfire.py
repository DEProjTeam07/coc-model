import mlflow
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from mlflow.exceptions import MlflowException
from sklearn.metrics import precision_score, recall_score, f1_score
from enum import Enum
import fire

from src.S3ImageDatasets import build_set_loaders

# 학습 파라미터 정의
class TrainingParams(Enum):
    EPOCHS = 10
    LEARNING_RATE = 0.001
    BUCKET_NAME = "your_bucket_name"
    VERSION = "latest"

# 모델 학습
def train_model(model, device, criterion, optimizer, train_loader):
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

# 모델 평가
def test_model(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []

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

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total

    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return test_loss, test_acc, precision, recall, f1

# 모델 학습 및 평가
def run_training(bucket_name=TrainingParams.BUCKET_NAME.value, version=TrainingParams.VERSION.value, epochs=TrainingParams.EPOCHS.value, learning_rate=TrainingParams.LEARNING_RATE.value):
    model = ...  # 모델 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, train_loader, test_loader = build_set_loaders(bucket_name=bucket_name, version=version)

    valid_loss_min = np.inf
    best_accuracy = 0
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1)
    
    model_name = model.get_model_name()
    previous_model_runs = mlflow.search_runs(order_by=["metrics.test_accuracy desc"], filter_string=f"tags.model_name='{model_name}'")
    if not previous_model_runs.empty:
        best_accuracy = previous_model_runs.iloc[0]['metrics.test_accuracy']

    with mlflow.start_run() as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n--------------------------")
            train_model(model, device, criterion, optimizer, train_loader)
            test_loss, test_acc, precision, recall, f1 = test_model(model, device, criterion, test_loader)

            mlflow.log_metric('test_loss', test_loss)
            mlflow.log_metric('test_accuracy', test_acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1)

            if test_acc > best_accuracy:
                print("검증 정확도가 증가 ------------ 모델 정확도를 저장합니다")
                best_accuracy = test_acc
                mlflow.pytorch.log_model(model, "model")

            scheduler.step()

        mlflow.pytorch.log_model(model, "final_model")
        model_uri = f"runs:/{run.info.run_id}/final_model"
        model_details = mlflow.register_model(model_uri, model_name)
        mlflow.set_tag(model_details.name, "alias", "champion")
        print(f"Model successfully registered at {model_uri}")

        print("Done!")
        mlflow.end_run()

if __name__ == "__main__":
    fire.Fire(run_training)
