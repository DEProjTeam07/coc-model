'''
Flask 서버 코드 
predict 함수는 mlflow model registry에서 등록된 model를 load해서 추론값을 return 하는 기능을 가진다. 
'''
import torch
import torchvision.transforms as transforms
import mlflow.pytorch
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Flask 객체를 만든다. 
app = Flask(__name__)

# mlflow model registy에 등록된 모델을 load하기 전에 set_tracking_uri를 명시해야 한다. 
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 모델명과 model_version을 명시한다. 
model_name = "Test"
model_version = 1

# mlflow에 있는 모델을 로드한다. 
model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델을 훈련 모드가 아닌 평가 모드로 전환하여 테스트 및 실제 서비스 환경에서 사용할 수 있도록 한다. 
model.eval()

# 사용자로부터 실제 이미지가 들어오면 이를 숫자화 하기 위한 코드 
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 웹 서비스 테스트 화면 
@app.route('/')
def index():
    return render_template('index.html')

# 사용자가 이미지를 업로드해서 mlflow에 로드된 모델을 활용하여 나온 추론값을 return 하는 함수 
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

        class_mapping = {0:"defective",
                         1:"good"}
        result = class_mapping.get(predicted_class, 'unknown')

        return jsonify({"predicted_class": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", 
            port=5002, 
            debug=True) # reload 옵션 설정 
