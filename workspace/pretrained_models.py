from torchvision import transforms, datasets, models
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights


# 3. Pretrained 모델 불러오기 + 드롭아웃 추가
#model = models.resnet18(pretrained=True)

# 4. 모델 GPU로 이동
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. 손실 함수 및 옵티마이저 설정 (weight decay 적용)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # weight decay 추가

# 6. 손실 및 정확도 추적을 위한 리스트
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# 7. 학습 및 평가 함수
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 8. 학습 단계
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 9. 평가 단계
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# 11. 손실 함수 및 정확도 그래프 그리기
def plot_loss_accuracy():
    epochs = range(1, len(train_losses) + 1)
    
    # 손실 함수 그래프
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Epoch')
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy per Epoch')
    
    plt.show()


# 10. 모델 학습
train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=20)

# 12. 결과 시각화
plot_loss_accuracy()#왜안됨???


from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. 이미지 전처리 정의 (모델 학습 시 사용한 전처리와 동일하게 설정)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. 이미지를 불러오고 전처리
def load_image(image_path):
    img = Image.open(image_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 배치 차원을 추가
    return img_tensor

# 3. 예측 함수
def predict_image(model, image_tensor):
    model.eval()  # 모델을 평가 모드로 전환
    image_tensor = image_tensor.to(device)  # 이미지를 GPU로 이동 (가능한 경우)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # 예측 클래스 반환

    return predicted.item()  # 예측 결과 반환

# 4. 클래스 레이블 정의 (0: Defective, 1: Good)
class_names = train_dataset.classes  # ImageFolder를 사용할 때 자동으로 클래스가 정의됨

# 5. 예측하기
image_path = "../workspace/Defective (66).jpg"  # 예측할 이미지 경로 설정
image_tensor = load_image(image_path)
predicted_class = predict_image(model, image_tensor)

# 6. 결과 출력
print(f"The model predicts the image is: {class_names[predicted_class]}")

# 7. 이미지 출력 (선택 사항)
img = Image.open(image_path)
plt.imshow(img)
plt.title(f'Predicted: {class_names[predicted_class]}')
plt.show()

