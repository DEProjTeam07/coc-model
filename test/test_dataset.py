import os
from PIL import Image
from unittest.mock import patch, MagicMock
from torch.utils.data import Dataset, DataLoader
from io import BytesIO
from torchvision import transforms
import numpy as np

class MockS3ImageDatasets(Dataset):
    def __init__(self, dataset_version, usage):
        self.dataset_version = dataset_version
        self.usage = usage
        self.prefix = f"{self.dataset_version}/{self.usage}/"
        
        # 환경 변수에서 S3 버킷 이름 불러오기 대신 하드코딩 (테스트용)
        self.bucket_name = "mock-bucket"
        
        # 이미지 변환 형태 지정
        self.transform = self._get_transform(usage)
        
        # 클래스 이름과 클래스 인덱스 매핑
        self.class_names = ['defective', 'good']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # S3 클라이언트 대신 목(mock) 객체 사용
        self.s3_client = MagicMock()
        self.imgs = self._load_images()

    def _get_transform(self, usage):
        transform_list = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if usage not in transform_list:
            raise ValueError("Transformation에 존재하지 않는 사용 형태입니다.")
        
        return transform_list[usage]

    # S3 대신 로컬에 있는 더미 이미지 리스트 로드
    def _load_images(self):
        imgs = []
        for class_name in self.class_names:
            # 로컬 데이터 대신 더미 이미지 리스트 생성
            for i in range(10):  # 각 클래스당 10개의 더미 이미지
                key = f"{self.prefix}{class_name}/dummy_image_{i}.jpg"
                imgs.append((key, self.class_to_idx[class_name]))
        return imgs
    
    # 더미 이미지를 로드 (S3 없이)
    def _load_image(self, key):
        # 랜덤한 이미지 데이터를 생성해 반환
        img_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_data)
        return img

    def __len__(self):
        return len(self.imgs)

    # 이미지와 라벨(클래스) 반환
    def __getitem__(self, idx):
        key, class_idx = self.imgs[idx]
        image = self._load_image(key)
        if self.transform:
            image = self.transform(image)
        return image, class_idx


# 목 데이터셋을 사용하는 DataLoader 생성
def build_set_loaders_mock(dataset_version):
    train_dataset = MockS3ImageDatasets(dataset_version=dataset_version, usage='train')
    test_dataset = MockS3ImageDatasets(dataset_version=dataset_version, usage='test')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    return train_dataset, test_dataset, train_loader, test_loader


# Pytest에서 사용될 테스트 케이스
def test_mock_s3_datasets():
    # Mock 데이터셋 로더 빌드
    train_dataset, test_dataset, train_loader, test_loader = build_set_loaders_mock(dataset_version="v1")

    # 데이터셋 길이 확인
    assert len(train_dataset) == 20  # 각 클래스마다 10개의 이미지
    assert len(test_dataset) == 20
    
    # 배치 크기 확인
    for images, labels in train_loader:
        assert images.shape == (16, 3, 256, 256)  # 배치 크기, 채널, 이미지 크기
        assert labels.shape == (16,)
        break  # 한 번만 확인

    for images, labels in test_loader:
        assert images.shape == (16, 3, 256, 256)
        assert labels.shape == (16,)
        break
