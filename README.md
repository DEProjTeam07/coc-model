# 타이어 결함 이미지 분류하기
타이어 이미지를 받아 결함 여부를 판단해주는 모델을 관리합니다.

### 명령어 예시
```
python main.py Train --dataset_version split_1 --model_type cnn --optimizer_type adam --epochs 10 --learning_rate 0.01 --batch_size 32
python main.py Stage Loss
python main.py Produce Loss
python main.py ProductionInfo

```

## 모델링 아키텍처

![아키텍처](model_architecture.png)

## 1. 데이터 분석 및 실험

### 1. EDA

### 2. Dataset
데이터 레이크인 s3로부터 이미지를 받고, 디렉토리 구조 정보로부터 라벨 정보를 파싱하여 첨부
```
def _load_images(self):
        imgs = []
        for class_name in self.class_names:
            class_prefix = str(self.prefix)+str(class_name)+'/'
            response = self.s3_client.list_objects(Bucket=self.bucket_name, Prefix=class_prefix)
            for obj in response['Contents']:
                imgs.append((obj['Key'], self.class_to_idx[class_name]))
        return imgs
        
    def _load_image(self, key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        img_data = obj['Body'].read()
        img = Image.open(BytesIO(img_data))
        return img
```
버킷이름과 usage(train or test), prefix(데이터 버전)정보를 받아 이미지와 이미지의 라벨정보를 함께 반환

실험 내용(train or test)에 따라 이미지를 다르게 전처리한다. 


## 2. Model 구성
### 1. models

초기 모델 구성을 위해 이미지 처리에 많이 쓰이는 efficientnet 모델을 먼저 구성한다. 비교를 위해 b0, b1, b3 세개를 옵션으로 제공한다. 
```
 match self.version:
            case 0:
                self.transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
                self.effi = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            case 1:
                self.transform = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1.transforms()
                self.effi = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            case 2:
                self.transform = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()
                self.effi = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            case _:
                raise ValueError("Efficientnets에 존재하지 않는 버전입니다.")

```

### 2. 모델 학습 및 평가
에포크를 돌면서 모델을 학습하고 테스트하며 성능을 확인한다. 모델 성능을 확인하여 최종적으로 모델을 구성한다. 

### 3. mlflow 설정
구성된 모델을 바탕으로 예측을 해본다. 


