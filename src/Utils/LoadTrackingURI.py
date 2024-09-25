import requests
import mlflow
import os

def get_tracking_uri():
    try:
        response = requests.get("https://api.ipify.org?format=json")
        response.raise_for_status()
        ip_address = response.json().get('ip')
        print(ip_address)

        with open('/home/ubuntu/coc-model/.env','w') as f:
            f.write(f"AWS_BUCKET_NAME = '<  bucket name here .... >'\n")
            f.write(f"MLFLOW_TRACKING_URI=http://{ip_address}:5000\n")
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    except requests.RequestException as e:
        print(f'퍼블릭 ip를 가져오는 중에 오류가 발생했습니다. : {e}')
