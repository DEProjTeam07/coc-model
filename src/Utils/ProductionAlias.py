import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import mlflow
import numpy as np
from LoadTrackingURI import get_tracking_uri

get_tracking_uri()
# mlflow.set_tracking_uri('s3://deprojteam07-datalake/model_registry')


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



# def get_best_model_metric(model_name, metric_type):
#         try:
#             client = mlflow.tracking.MlflowClient()
#             # 모델이 레지스트리에 없으면 패스하고 로그된 모델을 비교
#             best_metric = np.inf if metric_type == 'loss' else 0
#             runs = client.search_runs(experiment_ids=["0"], filter_string=f"params.model_name = '{model_name}'")
#             for run in runs:
#                 metrics = run.data.metrics
#                 if metric_type == 'loss' and "Loss" in metrics:
#                     if metrics["Loss"] < best_metric:
#                         best_metric = metrics["Loss"]
#                 elif metric_type == 'accuracy' and "Accuracy" in metrics:
#                     if metrics["Accuracy"] > best_metric:
#                         best_metric = metrics["Accuracy"]
#             return best_metric
#         except Exception as e:
#             print(f"모델을 가져오는 중 오류가 발생했습니다: {e}")
#             return np.inf if metric_type == 'loss' else 0

# def register_model( model_name, model_uri):
#     client = mlflow.tracking.MlflowClient()
#     try:
#         # 모델이 이미 등록되어 있는지 확인
#         client.get_registered_model(model_name)
#     except mlflow.exceptions.RestException:
#         # 모델이 등록되어 있지 않으면 새로 등록
#         print(f"{model_name} 모델이 없으므로 새로 등록합니다.")
#         mlflow.register_model(model_uri, model_name)

#     model_details = mlflow.register_model(model_uri, model_name)
#     client.transition_model_version_stage(
#         name=model_name,
#         version=model_details.version,
#         stage="Production"
#     )
#     client.update_model_version(
#         name=model_name,
#         version=model_details.version,
#         description="현재 가장 좋은 성능의 모델입니다."
#     )
#     print(f"모델이 '{model_name}'로 레지스트리에 등록되었습니다.")


