import os
import mlflow


def stage_alias_first_second(evaluation_metric):
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='Experiments'")
    best_model_version = None
    best_metric_value = None

    second_model_version = None
    second_metric_value = None

    for version in versions:
        run_id = version.run_id
        model_name = version.tags.get('model_name', None)  # 'model_name'이 없을 경우를 대비
        if not model_name:
            continue  # model_name이 없는 경우는 건너뜁니다

        try:
            run_data = client.get_run(run_id).data
            metrics = run_data.metrics
            if evaluation_metric in metrics:
                metric_value = metrics[evaluation_metric]

                # 평가 메트릭이 'loss'인 경우 최소값이 더 좋음, 그 외에는 최대값이 더 좋음
                if evaluation_metric == 'loss':
                    if best_metric_value is None or metric_value < best_metric_value:
                        if best_metric_value is not None:  # 최초가 아니라면 second 값을 업데이트
                            second_metric_value = best_metric_value
                            second_model_version = best_model_version
                            second_model_name = best_model_name
                            second_run_id = best_run_id

                        best_metric_value = metric_value
                        best_model_version = version.version
                        best_run_id = run_id
                        best_model_name = model_name
                    elif second_metric_value is None or metric_value < second_metric_value:
                        second_metric_value = metric_value
                        second_model_version = version.version
                        second_model_name = model_name
                        second_run_id = run_id

                else:
                    if best_metric_value is None or metric_value > best_metric_value:
                        if best_metric_value is not None:  # 최초가 아니라면 second 값을 업데이트
                            second_metric_value = best_metric_value
                            second_model_version = best_model_version
                            second_model_name = best_model_name
                            second_run_id = best_run_id

                        best_metric_value = metric_value
                        best_model_version = version.version
                        best_run_id = run_id
                        best_model_name = model_name
                    elif second_metric_value is None or metric_value > second_metric_value:
                        second_metric_value = metric_value
                        second_model_version = version.version
                        second_model_name = model_name
                        second_run_id = run_id

        except Exception as e:
            print(f"런 메트릭을 가져오는 중 오류 발생 (run_id: {run_id}): {e}")
            continue

    # 가장 높은 메트릭 값을 가진 모델을 Production으로 설정
    if best_model_version is not None:
        print(f"최선의 {evaluation_metric}를 가진 모델 Version {best_model_version}을 Production으로 설정합니다.")
        try:
            # stage 모델 중 동일한 태그를 가진 모델이 있는지 확인
            staged_models = client.search_model_versions('name="Staged"')
            for staged_model in staged_models:
                if staged_model.tags.get('status') == 'First' and staged_model.tags.get('evaluation_metric') == evaluation_metric:
                    print(f"Staged에 등록된 모델 중 같은 평가 기준으로 1등인 모델이 있어 해당 버전은 삭제합니다.")
                    client.transition_model_version_stage(name='Staged', version=staged_model.version, stage='Archived')
            print("새로운 모델을 등록합니다.")
            model_uri = f"runs:/{best_run_id}/{best_model_name}"
            print(model_uri)
            mlflow.register_model(model_uri, name='Staged',
                                  tags={'model_name': best_model_name, 'status': 'First', 'evaluation_metric': evaluation_metric})
        except Exception as e:
            print(f"모델을 Staged로 설정하는 중 오류 발생: {e}")
    else:
        print("Production으로 설정할 모델을 찾지 못했습니다.")

    if second_model_version is not None:
        print(f"두 번째로 좋은 {evaluation_metric}를 가진 모델 Version {second_model_version}을 Production으로 설정합니다.")
        try:
            # stage 모델 중 동일한 태그를 가진 모델이 있는지 확인
            staged_models = client.search_model_versions('name="Staged"')
            for staged_model in staged_models:
                if staged_model.tags.get('status') == 'Second' and staged_model.tags.get('evaluation_metric') == evaluation_metric:
                    print(f"Staged에 등록된 모델 중 같은 평가 기준으로 동일한 모델이 있어 해당 버전은 삭제합니다.")
                    client.transition_model_version_stage(name='Staged', version=staged_model.version, stage='Archived')
            print("새로운 모델을 등록합니다.")
            model_uri = f"runs:/{second_run_id}/{second_model_name}"
            print(model_uri)
            mlflow.register_model(model_uri, name='Staged',
                                  tags={'model_name': second_model_name, 'status': 'Second', 'evaluation_metric': evaluation_metric})
        except Exception as e:
            print(f"모델을 Staged로 설정하는 중 오류 발생: {e}")
    else:
        print("Production으로 설정할 두 번째 모델을 찾지 못했습니다.")


def produce_alias(evaluation_metric):
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='Staged'")
    print(versions)
    for version in versions:
        if version.tags.get('evaluation_metric') == evaluation_metric and version.tags.get('status') == 'First':
            run_id = version.run_id
            model_name = version.tags.get('model_name')
            print(f'Production model로 {evaluation_metric}의 기준에서 최선의 모델인 {model_name}을 선정합니다.')
            client.transition_model_version_stage(name='Staged', version=version.version, stage='Archived')
            model_uri = f"runs:/{run_id}/{model_name}"
            print(model_uri)
            mlflow.register_model(model_uri, name='Production',
                                  tags={'model_name': model_name,'evaluation_metric': evaluation_metric})
