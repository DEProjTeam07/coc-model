import mlflow
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_score, recall_score, f1_score

from src.Utils.Models import ModelType, get_model
from src.S3ImageDatasets import build_set_loaders
from src.Utils.TrainingParams import validate_params
from src.Utils.Optimizer import OptimizerType, get_optimizer
from src.Utils.EarlyStopping import EarlyStopping

class TrainModel():
    def __init__(self, model_type, model_version, epochs, 
                 optimizer_type, dataset_version, learning_rate, batch_size,
                 min_loss=0.3, min_accuracy=70):
        self.model_type = model_type
        self.model_version = model_version
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.dataset_version = dataset_version
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.min_loss = min_loss
        self.min_accuracy = min_accuracy

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 타입을 enum으로 변환
        self.model_type_enum = getattr(ModelType, self.model_type.upper(), None)
        # print(model_type_enum)
        if self.model_type_enum is None:
            raise ValueError(f"지원되지 않는 모델 타입입니다.: {self.model_type}")

        # 모델 선택
        self.model = get_model(self.model_type_enum, model_version).to(self.device)

        #옵티마이저 타입을 enum으로 변환
        optimizer_type_enum = getattr(OptimizerType, self.optimizer_type.upper(), None)
        if optimizer_type_enum is None:
            raise ValueError(f"지원되지 않는 옵티마이저 타입입니다.: {self.optimizer_type}")

        self.optimizer = get_optimizer(optimizer_type_enum, self.model, self.learning_rate)

        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = build_set_loaders(self.dataset_version)

        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.epochs, gamma=0.1)

    # 모델 학습
    def train_model(self):
        size = len(self.train_loader.dataset)
        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(inputs)
                print(f"Batch {batch}: Loss: {loss:.7f}  [{current}/{size}]")

    # 모델 평가
    def test_model(self):
        self.model.eval()
        test_loss, total, correct = 0.0, 0, 0
        all_labels, all_predictions = [], []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        test_loss /= len(self.test_loader)
        test_acc = 100 * correct / total

        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        return test_loss, test_acc, precision, recall, f1

    #에포크별 학습
    def run_training(self):
        model_name = self.model.get_model_name()
        
        mlflow.set_tag('model_type',model_name)
        mlflow.set_experiment(f'Experiment_{model_name}')

        validate_params(self.epochs, self.learning_rate)
        early_stopping = EarlyStopping(min_loss=self.min_loss, min_acc=self.min_accuracy)

        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("optimizer_type", self.optimizer_type)
            mlflow.log_param("early_stopping_min_loss", self.min_loss)
            mlflow.log_param("early_stopping_min_accuracy", self.min_accuracy)

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}\n--------------------------")
                self.train_model()
                loss, acc, precision, recall, f1 = self.test_model()
                print(f"loss : {loss}, acc : {acc}")
                mlflow.log_metric('Loss', loss, step=epoch)
                mlflow.log_metric('Accuracy', acc, step=epoch)
                mlflow.log_metric('Precision', precision, step=epoch)
                mlflow.log_metric('Recall', recall, step=epoch)
                mlflow.log_metric('F1_Score', f1, step=epoch)

                mlflow.log_param("Learning_rate_step", self.scheduler.get_last_lr()[0])
                
                early_stopping(self.model, current_loss=loss, current_acc=acc)

                if early_stopping.early_stop:
                    print('--------조기종료--------')
                    break

                self.scheduler.step()
            
            if early_stopping.model_log_triggered :
                artifact_path = f'{model_name}'
                mlflow.pytorch.log_model(self.model, 
                                         artifact_path=artifact_path
                                         )
                model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
                mlflow.register_model(model_uri=model_uri, name='Test', tags={"model_name":model_name})
                print('최소값을 넘겨 모델 등록에 성공하였습니다.')
            else:
                print("최소값을 넘지 못해 모델 등록에 실패하였습니다.")
        mlflow.end_run()