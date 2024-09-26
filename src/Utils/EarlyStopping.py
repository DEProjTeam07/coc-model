import torch

class EarlyStopping:
    def __init__(self, patience=3, delta=0.001, min_loss=0.5, min_acc = 70):
        self.patience = patience
        self.delta = delta
        self.min_acc = min_acc
        self.min_loss = min_loss
        self.best_loss = None
        self.best_acc = None
        self.counter=0
        self.early_stop = False
        self.model_log_triggered = False
        
    def __call__(self, model, current_loss, current_acc):
        if current_loss <= self.min_loss and current_acc >= self.min_acc:
            print('조기 종료 조건을 만족하여 모델을 로그하고 학습을 종료합니다.')
            self.early_stop = True
            self.model_log_triggered = True
            return
        
        if self.best_loss is None or self.best_acc is None:
            self.best_loss = current_loss
            self.best_acc = current_acc
            #모델 로그하는 코드
        elif current_loss > self.best_loss - self.delta and current_acc < self.best_acc + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.best_acc = current_acc
            self.counter = 0