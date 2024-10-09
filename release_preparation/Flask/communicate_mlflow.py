import mlflow

# mlflow Model Registry에 Staged Name의 First Status 모델의 run_id를 가져오는 함수 
def get_staged_first_status_model_run_id():
    client = mlflow.tracking.MlflowClient()
    
    # "Production" Name에 모델들을 가져온다. 
    versions = client.search_model_versions("name='Staged'")
    
    # 만약 모델들이 없는 경우 처리
    if not versions:
        print("Error: Model Registry에 'Staged' 이름을 가진 모델이 없습니다.")
        return None  
    
    print(f"get_staged_first_status_model_run_id 함수 - Staged Name에서 가져온 모델 리스트 : {versions}")

    # Tags의 'status' 값이 'First'인 모델을 필터링
    first_status_model_versions = [v for v in versions if v.tags.get("status") == "First"]
    
    print(f"get_staged_first_status_model_run_id 함수 - Tags의 status 값이 First인 모델 : {first_status_model_versions}")

     # 첫 번째 'status'가 'First'인 모델의 run_id를 가져온다.
    run_id = first_status_model_versions[0].run_id
    
    print(f"get_staged_first_status_model_run_id 함수 - 모델의 run_id : {run_id}")
    
    return run_id


# mlflow Model Registry에 Staged Name의 First Status 모델의 uri를 가져오는 함수 
def get_staged_first_status_model_uri():
    client = mlflow.tracking.MlflowClient()
    
    # "Staged" Name에 모델들을 가져온다
    versions = client.search_model_versions("name='Staged'")
    
    # 만약 모델들이 없는 경우 처리
    if not versions:
        print("Error: Model Registry에 'Staged' 이름을 가진 모델이 없습니다.")
        return None  
    
    print(f"get_staged_first_status_model_run_uri 함수 - Staged Name에서 가져온 모델 리스트 : {versions}")

    # Tags의 'status' 값이 'First'인 모델을 필터링
    first_status_model_versions = [v for v in versions if v.tags.get("status") == "First"]
    
    print(f"get_staged_first_status_model_run_uri 함수 - Tags의 status 값이 First인 모델 : {first_status_model_versions}")
    
    # 모델 이름과 버전으로 모델 URI 생성
    model_name = first_status_model_versions[0].name
    model_version = first_status_model_versions[0].version
    model_uri = f"models:/{model_name}/{model_version}"
    
    return model_uri









# mlflow Model Registry에 Production Name에 모델의 run id를 가져오는 함수 
# def get_production_run_id():
#     client = mlflow.tracking.MlflowClient()
#     version = client.search_model_versions("name='Production'")

#     run_id = version[0].run_id
    
#     return run_id 

    
# mlflow Model Registry에 Production Name에 모델의 model_uri를 가져오는 함수 
# def get_production_model_uri():
#     client = mlflow.tracking.MlflowClient()
#     version = client.search_model_versions("name='Production'")

#     if len(version) > 1:
#         print("운영 모델이 2개 이상입니다.")
#     else:
#         run_id = version[0].run_id
#         model_name = version[0].tags.get('model_name')
#         model_uri=f"runs:/{run_id}/{model_name}"
#     return model_uri
