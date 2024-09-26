import os
import sys
import mlflow
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


client = mlflow.tracking.MlflowClient()

versions = client.search_model_versions("name='experiments'")

best_model_version = None
best_metric_value = None

evaluation_metric = 'loss'

for version in versions:
    run_id = version.run_id
    experiment_name = version.tags['model_name']
    print(experiment_name)
    try:
        # 각 버전의 메트릭 가져오기
        run_data = client.get_run(run_id).data
        metrics = run_data.metrics

        if evaluation_metric in metrics:
            metric_value = metrics[evaluation_metric]
            print(f"Model: {experiment_name}, Version: {version.version}, {evaluation_metric}: {metric_value}")

            # 정확도 (혹은 다른 평가 메트릭) 비교
            if best_metric_value is None or metric_value > best_metric_value:
                best_metric_value = metric_value
                best_model_version = version.version
                print(f"best_model:{best_model_version}")
    except Exception as e:
        print(f"런 메트릭을 가져오는 중 오류 발생 (run_id: {run_id}): {e}")
        continue

    # 가장 높은 정확도의 모델을 Production으로 설정
    if best_model_version is not None:
        print(f"가장 높은 {evaluation_metric}를 가진 모델 Version {best_model_version}을 Production으로 설정합니다.")
        try:
            client.transition_model_version_stage(
                name=experiment_name,
                version=best_model_version,
                stage='Production'
            )
        except Exception as e:
            print(f"모델을 Production으로 설정하는 중 오류 발생: {e}")
    else:
        print("Production으로 설정할 모델을 찾지 못했습니다.")
    