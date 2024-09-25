import fire

from src.Train import TrainModel
from src.Utils.LoadTrackingURI import get_tracking_uri
from src.Utils.ProductionAlias import production_model_alias
get_tracking_uri()

#모델 학습
def train_model(dataset_version: str, model_type,
                model_version=None, optimizer_type='adam', epochs=10, learning_rate=0.001, batch_size=16):

    # 모델 학습 실행
    model = TrainModel(
        model_type=model_type,
        model_version=model_version,
        optimizer_type=optimizer_type,
        dataset_version=dataset_version,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

    model.run_training()
    
#학습된 모델의 성능 확인해서 운영 모델 결정
def production_alias(param):
    production_model_alias(param)

# #운영중인 모델 로드 - 성능이랑 등등
# def load_production_model(model_type):
#     pass

# # #운영중인 모델로 추론 서버 띄우기 
# # def load_inf_server():
# #     pass

if __name__ == "__main__":
    fire.Fire({
        'train':train_model,
        'production': production_alias
    })