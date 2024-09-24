import torch

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        """
        :param patience: 개선이 없을 때 기다릴 에포크 수
        :param min_delta: 메트릭 향상을 위한 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_model = None
        self.early_stop = False

    def __call__(self, model, current_score):
        """
        :param model: PyTorch 모델
        :param current_score: 현재 에포크에서 계산된 평가 메트릭
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_model = model.state_dict()
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"No improvement in score for {self.counter} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_model = model.state_dict()
            self.counter = 0
            print(f"New best model found with score: {current_score:.4f}")

    def load_best_model(self, model):
        """
        최고의 모델 가중치로 모델을 복원
        :param model: PyTorch 모델
        """
        model.load_state_dict(self.best_model)
        return model
