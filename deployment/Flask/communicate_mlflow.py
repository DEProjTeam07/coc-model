# mlflow Model Registry에 서비스에 사용될 모델을 가져오는 모듈 

import mlflow

# mlflow Model Registry에 Production Name에 가장 최근 모델의 run id를 가져오는 함수 
def get_production_run_id():
    client = mlflow.tracking.MlflowClient()
    
    # "Production" Name에 모델들을 가져온다. 
    versions = client.search_model_versions("name='Production'")
    
    # 만약 모델들이 없는 경우 처리
    if not versions:
        print("Error: Model Registry에 'Production' 이름을 가진 모델이 없습니다.")
        return None  
    
    # 'Registered at' (creation_timestamp) 기준으로 내림차순 정렬하여 최신 버전 선택 -> 사실 필요는 없음 혹시나 모를까봐 
    sorted_versions = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
    
    print(f"get_production_run_id 함수 -  Experiments Name에서 가져온 모델 리스트 : {sorted_versions}")

    # 가장 최근에 등록된 모델 버전의 run_id 반환
    run_id = sorted_versions[0].run_id
    
    print(f"get_production_run_id 함수 - 모델의 run_id : {run_id}")
    
    return run_id


# mlflow Model Registry에 Production Name에 모델의 model_uri를 가져오는 함수 
def get_production_model_uri():
    client = mlflow.tracking.MlflowClient()
    
    # "Production" Name에 모델들을 가져온다
    versions = client.search_model_versions("name='Production'")
    
    # 만약 모델들이 없는 경우 처리
    if not versions:
        print("Error: Model Registry에 'Production' 이름을 가진 모델이 없습니다.")
        return None  
    
    # 'Registered at' (creation_timestamp) 기준으로 내림차순 정렬하여 최신 버전 선택 -> 사실 필요는 없음 혹시나 모를까봐 
    sorted_versions = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
    
    print(f"get_production_model_uri 함수 - Experiments Name에서 가져온 모델 리스트 : {sorted_versions}")
    
    # 최신 버전의 모델 정보 가져오기
    latest_version = sorted_versions[0]
    
    print(f"get_production_model_uri 함수 - 가져온 모델 정보 : {latest_version}")
    
    # 모델 이름과 버전으로 모델 URI 생성
    model_name = latest_version.name
    model_version = latest_version.version
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
