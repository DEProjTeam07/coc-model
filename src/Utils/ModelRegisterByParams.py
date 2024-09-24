import mlflow
from mlflow.tracking import MlflowClient

def get_all_registered_models(client):
    # 모든 모델 이름을 수동으로 가져오기
    registered_models = []
    for model_name in client.search_registered_models():
        registered_models.append(model_name.name)
    return registered_models

def promote_best_model_to_production_for_all_models(evaluation_metric):
    # MLflowClient를 사용하여 레지스트리와 상호작용
    client = MlflowClient()

    # 모든 모델 이름 가져오기
    all_registered_models = get_all_registered_models(client)

    # 각 모델에 대해 처리
    for model_name in all_registered_models:
        print(f"Processing model: {model_name}")

        # 모델의 모든 버전 가져오기 (필터링 형식을 수정)
        versions = client.search_model_versions(f"name = '{model_name}'")
        
        best_metric_value = None
        best_version = None
        
        # 각 모델 버전의 메트릭을 조회하고, 가장 좋은 메트릭을 가진 모델 찾기
        for version in versions:
            run_id = version.run_id
            metrics = client.get_run(run_id).data.metrics
            if evaluation_metric in metrics:
                metric_value = metrics[evaluation_metric]
                print(f"Model: {model_name}, Version: {version.version}, {evaluation_metric}: {metric_value}")
                
                if best_metric_value is None or metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_version = version
        
        # 가장 좋은 메트릭을 가진 모델의 스테이지를 'Production'으로 업데이트
        if best_version is not None:
            print(f"Promoting model '{model_name}' version {best_version.version} with best {evaluation_metric}: {best_metric_value} to Production")
            client.transition_model_version_stage(
                name=model_name,
                version=best_version.version,
                stage="Production"
            )
        else:
            print(f"No suitable model found for metric {evaluation_metric} for model '{model_name}'")

# 평가할 메트릭을 정의
evaluation_metric = "Accuracy"  # 가장 좋은 모델을 선택할 때 기준이 되는 메트릭

# 모든 모델에 대해 가장 좋은 메트릭을 가진 모델을 프로덕션으로 전환
promote_best_model_to_production_for_all_models(evaluation_metric)
