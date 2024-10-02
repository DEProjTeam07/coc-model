import mlflow
import torch
from torchvision import transforms
from PIL import Image

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = mlflow.pytorch.load_model(context.artifacts['model'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_image(self, file):
        image = Image.open(file.stream).convert("RGB")  # 이미지 로드 및 RGB로 변환
        image = self.transform(image).unsqueeze(0)  # 배치 차원 추가
        image = image.to(self.device)
        return image
    
    def predict(self, context, file):
        image_tensor = self.preprocess_image(file)
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        class_mapping = {0:"defective",
                         1:"good"}
        result = class_mapping.get(predicted_class, 'unknown')

        return result