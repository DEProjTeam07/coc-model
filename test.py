model_type = 'efficientnet'
model_version = 1
epochs = 10
evaluation_metric = 'loss'
optimizer_type = 'adam'
dataset_version = 'split_1'
learning_rate = 0.001
batch_size = 16
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from src.Utils.Models import ModelType, get_model
from src.Utils.Optimizer import OptimizerType, get_optimizer
from src.S3ImageDatasets import build_set_loaders
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_score, recall_score, f1_score
import torch


# 모델 타입을 enum으로 변환
model_type_enum = getattr(ModelType, model_type.upper(), None)
# print(model_type_enum)
if model_type_enum is None:
    raise ValueError(f"지원되지 않는 모델 타입입니다.: {model_type}")

# 모델 선택
model = get_model(model_type_enum, model_version).to(device)

#옵티마이저 타입을 enum으로 변환
optimizer_type_enum = getattr(OptimizerType, optimizer_type.upper(), None)
if optimizer_type_enum is None:
    raise ValueError(f"지원되지 않는 옵티마이저 타입입니다.: {optimizer_type}")

optimizer = get_optimizer(optimizer_type_enum, model, learning_rate)

train_dataset, test_dataset, train_loader, test_loader = build_set_loaders(dataset_version)

criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1)

# 모델 학습
def train_model():
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
            print(f"Batch {batch}: Loss: {loss:.7f}  [{current}/{size}]")

# 모델 평가
def test_model():
    model.eval()
    test_loss, total, correct = 0.0, 0, 0
    all_labels, all_predictions = [], []
    
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

import mlflow
from src.Utils.LoadTrackingURI import get_tracking_uri
from src.Utils.TrainigParams import validate_params
from src.Utils.EarlyStopping import EarlyStopping

def run_training():
    model_name = model.get_model_name()
    artifact_path = f'models/{model_name}'
    mlflow.set_tag('model_type',model_name)
    mlflow.set_experiment(f'{model_name}_experiment')
    validate_params(epochs, learning_rate)

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("optimizer_type", optimizer_type)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}\n--------------------------")
            train_model()
            loss, acc, precision, recall, f1 = test_model()

            mlflow.log_metric('Loss', loss, step=epoch)
            mlflow.log_metric('Accuracy', acc, step=epoch)
            mlflow.log_metric('Precision', precision, step=epoch)
            mlflow.log_metric('Recall', recall, step=epoch)
            mlflow.log_metric('F1_Score', f1, step=epoch)

            mlflow.pytorch.log_model(model, artifact_path=artifact_path, register_model_name=model_name) 

            scheduler.step()
        
        print("Done!")
        mlflow.end_run()            

get_tracking_uri()
run_training()