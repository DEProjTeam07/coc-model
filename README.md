# 타이어 결함 이미지 분류하기
## 1. 개요
타이어 이미지를 받아 결함 여부를 판단해주는 모델을 관리합니다.
mlops level 2 아키텍처를 차용하여 모델을 프로덕현 환경에 안정적으로 배포하고, 자동화된 파이프라인으로 훈련, 배포, 모니터링 워크 플로우를 지원하는 체계를 구현하였습니다.

## 2. 아키텍처 개요
![아키텍처](model_architecture.png)

다음과 같은 컴포넌트로 구성하였습니다.
- **데이터 저장소**: 데이터가 버전 관리되며 실험에 사용할 수 있도록 구성됩니다. s3 를 사용하였습니다.
- **모델 훈련**: GitHub Actions를 사용하여 새로운 데이터를 통해 모델을 자동으로 훈련합니다.
- **모델 등록 및 추적**: MLflow를 사용하여 실험과 모델 성능을 기록하고 관리합니다.
- **모델 배포**: 모델을 Docker 컨테이너를 통해 프로덕션 환경에 배포합니다.
- **모델 모니터링**: 프로덕션 환경에서의 모델 성능을 모니터링하고 슬랙 알림을 통해 성능 저하에 대응합니다.

## 3. 기술 스택

| 역할 | 기술 스택 | 비고 |
|----|-----|-----|
|CI/CD| GitHub Actions| 자동화된 모델 훈련 및 배포를 위한 워크플로우 설정|
|실험 관리| MLflow| 모델 훈련 과정과 성능 지표 추적|
|데이터 저장| Amazon S3| 모델 아티팩트 저장소로 사용|
|데이터 저장| PostgreSQL, PGPool | 모델 파라미터, 메트릭 등 메타데이터 저장소로 사용|
|컨테이너| Docker |컨테이너화된 환경에서 모델을 배포|
|모델 훈련 및 서빙| PyTorch |모델 훈련과 추론을 위한 딥러닝 프레임워크|
|모델 모니터링| Prometheus, Grafana | 프로덕션에서의 모델 성능 모니터링|

## 4. 모델링 과정
결함 타이어 사진과 정상 타이어 사진 데이터를 대상으로 이진 분류 모델을 학습하고 평가합니다.

### 명령어 예시
```
python main.py Train --dataset_version split_1 --model_type cnn --optimizer_type adam --epochs 10 --learning_rate 0.01 --batch_size 32
```

### 1. Dataset 구성
데이터 레이크로부터 이미지를 받고, 디렉토리 구조 정보로부터 라벨 정보를 파싱하여 첨부
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

```
```
실험 내용(train or test)에 따라 이미지를 다르게 전처리한다. 

```
```
실험을 위해 데이터셋을 랜덤 스플릿하여 활용

### 2. Model 구성
#### 1. models
이진분류에 사용된 모델은 다음과 같다.
* Efficientnet_b0, Efficientnet_b1, Efficientnet_b2, 
* Resnet 18, Resnet 50
* TinyVgg
* CNN

#### 2. 모델 학습 및 평가
에포크를 돌면서 모델을 학습하고 테스트하며 성능을 확인한다. 학습 에포크마다 파라미터와 메트릭을 수집하여 메타 데이터 저장소인 postgres에 저장한다. 
```
```
조기종료 조건을 만족하면 모델을 등록하고 종료한다.
```
```

최소 임계값을 넘지 못한 실험의 모델은 등록하지 않는다. 
```
```
#### 3. 하이퍼파라미터 튜닝
매개변수로 하이퍼파라미터를 받고 그걸 토대로 학습을 진행.
```
```
학습 진행 전 입력된 파라미터 값이 유효한지 검증
```
```

## 5. 기능 구현

### 1. 새로운 데이터가 데이터 레이크에 저장되면 자동화 워크플로우 배포

```
python main.py Stage Loss
python main.py Produce Loss
python main.py ProductionInfo

```
