import torch
from tqdm.notebook import tqdm

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

def test_model(model,device, criterion, optimizer, train_loader, test_loader, save_file, num_epochs=10) :
    for epoch in range(num_epochs):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): #기울기 계산 비활성화
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs) #순전파
                loss = criterion(outputs, labels) #손실값 계산
                test_loss += loss.item() #현재 배치에서의 손실 추가

                _, predicted = torch.max(outputs, 1) #예측값 및 실제값
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    return test_losses, test_accuracies