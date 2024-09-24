import os
import fire
import torch
import mlflow
from src.Utils.TrainigParams import validate_params  # 파라미터 검증 함수
from src.Train import TrainModel
from src.Utils.LoadTrackingURI import get_tracking_uri
get_tracking_uri()

# from src.Utils.ProductionAlias import production_alias
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#모델 학습
def train_model(dataset_version: str, model_type, evaluation_metric, 
                model_version=None, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16):

    # 모델 학습 실행
    model = TrainModel(
        model_type=model_type,
        model_version=model_version,
        device=device,
        optimizer_type=optimizer_type,
        dataset_version=dataset_version,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

    model.run_training()
    
#학습된 모델의 성능 확인해서 운영 모델 결정
# def production_alias(model_type, param):
#     production_alias(model_name=model_type, param=param)

#운영중인 모델 로드 - 성능이랑 등등
# def load_production_model(model_type):
    

# #운영중인 모델로 추론 서버 띄우기 
# def load_inf_server():
#     pass

from src.Utils.ProductionAlias import get_all_registered_models
if __name__ == "__main__":
    # fire.Fire({
    #     'train_model': train_model
    # })
    # mlflow.set_tracking_uri('s3://deprojteam07-datalake/model_registry')
    # train_model(dataset_version='split_1', model_type="efficientnet", evaluation_metric='loss',
    #             model_version=0, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16)
    # train_model(dataset_version='split_1', model_type="efficientnet", evaluation_metric='loss',
    #             model_version=1, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16)
    # train_model(dataset_version='split_1', model_type="efficientnet", evaluation_metric='loss',
    #             model_version=1, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16)
    # train_model(dataset_version='split_1', model_type="efficientnet", evaluation_metric='loss',
    #             model_version=0, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16)
    # train_model(dataset_version='split_1', model_type="efficientnet", evaluation_metric='loss',
    #             model_version=1, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16)
    # train_model(dataset_version='split_1', model_type="efficientnet", evaluation_metric='loss',
    #             model_version=0, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16)
    # train_model(dataset_version='split_1', model_type="efficientnet", evaluation_metric='loss',
    #             model_version=2, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16)
    client = mlflow.tracking.MlflowClient()
    def get_all_registered_models(client):
        # 모든 모델 이름을 수동으로 가져오기
        registered_models = []
        for model_name in client.search_registered_models():
            registered_models.append(model_name.name)
        return registered_models

    for registered_model in get_all_registered_models(client):
        versions = client.search_model_versions(f"name='{registered_model}")

        best_metric_value = None
        best_version = None

        for version in versions:
            run_id = version.run_id
            print(run_id)
            metrics = client.get_run(run_id).data.metrics
            print(metrics)
