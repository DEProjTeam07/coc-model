# 사용자 정의 모듈 
from Production import production_model_info

# 외부 모듈 
import torch
import torchvision.transforms as transforms
import mlflow.pytorch
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 웹 서비스 테스트 화면
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=["GET"])
def health():
    return jsonify({'status':'healthy'}), 200 

# 사용자가 이미지를 업로드해서 mlflow에 로드된 모델을 활용하여 추론 값을 반환하는 함수
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    model_uri = production_model_info()
    model_uri = str(model_uri)

    model = mlflow.pytorch.load_model(model_uri)

    # 이미지를 로드 및 RGB로 변환하고 추론을 시작한다.
    try:
        image = Image.open(file.stream).convert("RGB")  # 이미지 로드 및 RGB로 변환
        image = transform(image).unsqueeze(0)  # 배치 차원 추가
        image = image.to(device)

        with torch.no_grad():
            output = model(image)  # 모델 추론
            predicted_class = torch.argmax(output, dim=1).item()  # 예측된 클래스 가져오기

        class_mapping = {0: "defective", 1: "good"}
        result = class_mapping.get(predicted_class, 'unknown')

        return jsonify({"predicted_class": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=15002, debug=True)


