import os
import json
import boto3
import random
from PIL import Image
from dotenv import load_dotenv
from torch.utils.data import Dataset
from io import BytesIO
from torchvision import transforms
from torch.utils.data import DataLoader


class S3ImageDatasets(Dataset):
    def __init__(self, dataset_version, usage):
        self.dataset_version = dataset_version
        self.usage = usage
        self.prefix = f"{self.dataset_version}/{self.usage}/"
        
        load_dotenv(dotenv_path='.env', verbose=True)
        self.bucket_name = os.getenv('AWS_BUCKET_NAME')

        # 이미지 변환 형태 지정
        self.transform = self._get_transform(usage)
        
        # 클래스 이름과 클래스 인덱스 매핑
        self.class_names = ['defective', 'good']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        self.s3_client = boto3.client('s3')
        self.annotation_path = f"{self.dataset_version}/{self.usage}/{self.usage}.json"
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.annotation_path)
        json_content = response['Body'].read().decode('utf-8')
        json_data = json.loads(json_content)

        self.imgs = self._load_images(json_data)

        sample_size = int(len(self.imgs) * 0.5)
        self.imgs = random.sample(self.imgs, sample_size)

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

    # 이미지 리스트 로드
    def _load_images(self, json_data):
        imgs = []
        try:
            for item in json_data['annotation']['data']:
                file_path = item['FilePath']
                label = item['label_value']
                imgs.append((file_path, label))
        except Exception as e:
            print(f"Json 파일로부터 이미지 경로와 라벨을 로드하는 중 오류 발생 : {e}")
        return imgs
    
    # 로드된 이미지 리스트 중 이미지 하나씩 로드
    def _load_image(self, key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        img_data = obj['Body'].read()
        img = Image.open(BytesIO(img_data))
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


# 하나의 데이터 버전에 있는 데이터를 train과 test 셋으로 랜덤 스플릿
def build_set_loaders(dataset_version):
    train_dataset = S3ImageDatasets(dataset_version=dataset_version, usage='train')
    test_dataset = S3ImageDatasets(dataset_version=dataset_version, usage='test')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    return train_dataset, test_dataset, train_loader, test_loader
