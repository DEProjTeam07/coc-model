from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import mlflow.pytorch
from dotenv import load_dotenv
import os

app = Flask(__name__)

# 모델 로드 (예: 모델 파일 경로, 디바이스 설정 필요)
#변수 로드
load_dotenv(dotenv_path= '/home/ubuntu/coc-model/.env',
            verbose= True,)
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)

model_name = 'Efficientnet_B1'
model_version = 2

model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")  # 이미지 로드 및 RGB로 변환
        image = transform(image).unsqueeze(0)  # 배치 차원 추가
        image = image.to(device)

        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        class_mapping = {0:"defective",1:"good"}
        result = class_mapping.get(predicted_class, 'unknown')

        return jsonify({"predicted_class": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
