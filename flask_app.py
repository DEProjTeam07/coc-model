# 사용자 정의 모듈 
from src.Production import get_run_model_info, production_model_info

# 외부 모듈 
import torch
import torchvision.transforms as transforms
import mlflow.pytorch
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Flask 객체 생성
app = Flask(__name__)

# device 설정 (CUDA 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mlflow model registry에 등록된 모델을 로드하기 전에 set_tracking_uri를 명시해야 한다.
mlflow.set_tracking_uri("http://127.0.0.1:15000")

# 전역 변수로 모델 및 run_id를 저장
model = None
current_run_id = None

# mlflow Model Registry에 있는 Production Name을 가진 모델을 가져오는 함수 
def load_model():
    global model, current_run_id
    new_run_id = get_run_model_info()  # mlflow에서 새로운 run_id를 가져옴
    
    # 전역 변수에 있는 current_run_id와 지금 mlflow Model Registry에 있는 Production Name을 가진 모델의 run_id와 같은지 확인한다. 
    # 만약 다르면 model를 다시 로드하고 전역 변수에 업데이트 한다. 
    if current_run_id != new_run_id:
        print(f"run_id가 변경되었습니다. 새 모델을 로드합니다. (새 run_id: {new_run_id})")
        model_uri = production_model_info()  # 새로운 모델 URI 가져오기
        model = mlflow.pytorch.load_model(model_uri=model_uri)  # 새로운 모델 로드
        model.eval()  # 모델을 평가 모드로 설정
        current_run_id = new_run_id  # 전역 변수에 새로운 run_id 저장
    else:
        print(f"현재 로드된 모델이 최신 모델입니다. (run_id: {current_run_id})")

# 처음 서버 시작 시 모델을 로드
load_model()

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 웹 서비스 테스트 화면
@app.route('/')
def index():
    return render_template('index.html')

# 사용자가 이미지를 업로드해서 mlflow에 로드된 모델을 활용하여 추론 값을 반환하는 함수
@app.route("/predict", methods=["POST"])
def predict():
    # 사용자가 이미지를 업로드했을 때 이미지의 내용을 체크한다. 
    # 만약 문제가 생기면 문제가 있으면 return 한다. 
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # 혹시 mlflow Model Registry에 Production Name을 가진 모델이 달라졌는지 확인한다. 
    # 만약 모델이 달라졌으면 전역 변수에 있는 model과 run id를 변경한다. 
    load_model()

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
    app.run(host="0.0.0.0", port=5002, debug=True)