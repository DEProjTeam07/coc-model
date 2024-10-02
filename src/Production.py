# mlflow Model Registry에 서비스에 사용될 모델을 가져오는 모듈 

import mlflow

# mlflow Model Registry에 Production Name에 모델의 run id를 가져오는 함수 
def get_run_model_info():
    client = mlflow.tracking.MlflowClient()
    version = client.search_model_versions("name='Production'")

    run_id = version[0].run_id
    
    return run_id 

    
# mlflow Model Registry에 Production Name에 모델의 model_uri를 가져오는 함수 
def production_model_info():
    client = mlflow.tracking.MlflowClient()
    version = client.search_model_versions("name='Production'")

    if len(version) > 1:
        print("운영 모델이 2개 이상입니다.")
    else:
        run_id = version[0].run_id
        model_name = version[0].tags.get('model_name')
        model_uri=f"runs:/{run_id}/{model_name}"
    return model_uri
