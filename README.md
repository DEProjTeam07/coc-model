# 타이어 결함 이미지 분류하기

mlops 프로세스


## 1. EDA

## 2. 초기 모델링
### 1. Dataset 구성
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

### 2. Model 구성
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

### 3. 모델 학습 및 평가
에포크를 돌면서 모델을 학습하고 테스트하며 성능을 확인한다. 모델 성능을 확인하여 최종적으로 모델을 구성한다. 
### 4. 예측
구성된 모델을 바탕으로 예측을 해본다. 

## 3. MLFlow
초기 모델 구성을 바탕으로 mlflow가 머신러닝 모델의 cicd를 관리할 수 있도록 모델 내에 메트릭을 수집하고 모델을 등록하는 코드를 이식한다. 

### 1. 설치하기
```
pip install mlflow
```
### 2. web ui 띄우기
```
mlflow ui --host 0.0.0.0 --port 5000
```
### 3. 파일 설정
모델 학습을 하는 과정에서 메트릭, 파라미터, 모델 아티팩트를 수집하고 모델 학습이 완료된 경우에 모델을 등록하므로 관련 코드를 작성한다. 

