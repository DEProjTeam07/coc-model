import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import mlflow
from src.Utils.LoadTrackingURI import get_tracking_uri

get_tracking_uri()

client = mlflow.tracking.MlflowClient()

def get_all_registered_models(client):
    try:
        registered_models = []
        for model in client.search_registered_models():
            registered_models.append(model.name)
        return registered_models
    except Exception as e:
        print(f"등록된 모델을 가져오는 중 오류 발생: {e}")
        return []

def production_model_alias(evaluation_metric):
    all_registered_models = get_all_registered_models(client)
    
    if not all_registered_models:
        print("등록된 모델이 없습니다.")
        return
    
    for model_name in all_registered_models:
        print(f"Processing model: {model_name}")
        
        try:
            versions = client.search_model_versions(f"name = '{model_name}'")
        except Exception as e:
            print(f"모델 버전을 가져오는 중에 오류 발생. model_name : {model_name} \n Error : {e}")
            continue
        
        best_metric_value = None
        second_metric_value = None
        best_version = None
        second_version = None

        for version in versions:
            run_id = version.run_id
            try:
                metrics = client.get_run(run_id).data.metrics
            except Exception as e:
                print(f"해당 run에 대한 메트릭을 가져오는 중 오류 발생 \n {run_id}: {e}")
                continue
            
            if evaluation_metric in metrics:
                metric_value = metrics[evaluation_metric]
                print(f"Model: {model_name} Version: {version.version}, {evaluation_metric}: {metric_value}")

                # Metric type branching
                if evaluation_metric == 'loss':
                    # For loss, lower is better
                    if best_metric_value is None or metric_value < best_metric_value:
                        second_metric_value = best_metric_value
                        second_version = best_version

                        best_metric_value = metric_value
                        best_version = version.version
                    elif second_metric_value is None or metric_value < second_metric_value:
                        second_metric_value = metric_value
                        second_version = version.version
                else:
                    # For accuracy, f1, precision, recall, higher is better
                    if best_metric_value is None or metric_value > best_metric_value:
                        second_metric_value = best_metric_value
                        second_version = best_version

                        best_metric_value = metric_value
                        best_version = version.version
                    elif second_metric_value is None or metric_value > second_metric_value:
                        second_metric_value = metric_value
                        second_version = version.version

        if best_version is not None:
            print(f'------------------------------\n 버전 {best_version} 을 {model_name}의 Production 버전으로 설정합니다.')
            try:
                client.set_model_version_tag(
                    name=model_name,
                    version=best_version,
                    key="stage",
                    value="Production"
                )
            except Exception as e:
                print(f"모델 {model_name}에 대해 Production 태그를 붙이는 중 오류 발생 : version {best_version}\n Error: {e}")

        if second_version is not None:
            print(f'------------------------------\n 버전 {second_version} 을 {model_name}의 Challenger 버전으로 설정합니다.')
            try:
                client.set_model_version_tag(
                    name=model_name,
                    version=second_version,
                    key="stage",
                    value="Challenger"
                )
            except Exception as e:
                print(f"모델 {model_name} 에 대해 Challenger 태그를 붙이는 중 오류 발생: version {second_version}\n Error: {e}")
        else:
            print("Challenger 버전으로 설정할 수 없음.")
