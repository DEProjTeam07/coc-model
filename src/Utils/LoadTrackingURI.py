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
            f.write(f"AWS_BUCKET_NAME = 'deprojteam07-labeledrawdata'\n")
            f.write(f"MLFLOW_TRACKING_URI=http://{ip_address}:5000\n")
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    except requests.RequestException as e:
        print(f'Error fetching public IP: {e}')
