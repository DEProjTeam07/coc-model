import fire
import mlflow
# mlflow.set_tracking_uri('http://127.0.0.1:5000')

from src.Train import TrainModel
from src.StageAlias import stage_alias_first_second, produce_alias
from src.Production import production_model_info

#모델 학습
def train_model(dataset_version: str, model_type,
                model_version=None, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16):

    # 모델 학습 실행
    model = TrainModel(
        model_type=model_type,
        model_version=model_version,
        optimizer_type=optimizer_type,
        dataset_version=dataset_version,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

    model.run_training()
    
#운영 모델 uri 반환

if __name__ == "__main__":
    train_model(dataset_version='version_1', model_type='efficientnet', model_version=0,optimizer_type='adam',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='efficientnet', model_version=0,optimizer_type='sgd',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='efficientnet', model_version=1,optimizer_type='adam',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='efficientnet', model_version=1,optimizer_type='sgd',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='efficientnet', model_version=2,optimizer_type='adam',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='efficientnet', model_version=2,optimizer_type='sgd',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='resnet', model_version=18,optimizer_type='adam',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='resnet', model_version=18,optimizer_type='sgd',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='resnet', model_version='50',optimizer_type='adam',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='resnet', model_version='50',optimizer_type='sgd',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='cnn', optimizer_type='adam',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='cnn', model_version='sgd',optimizer_type='',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='tinyvgg', model_version='adam',optimizer_type='',epochs=10, learning_rate=0.01, batch_size=16)
    train_model(dataset_version='version_1', model_type='tinyvgg', model_version='sgd',optimizer_type='',epochs=10, learning_rate=0.01, batch_size=16)
