import boto3
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO
from torchvision import transforms
from torch.utils.data import DataLoader


class S3ImageDatasets(Dataset):
    def __init__(self, bucket_name, version, usage):
        self.bucket_name = bucket_name
        self.version = version
        self.usage = usage
        self.prefix = str(self.version)+'/'+str(self.usage)+'/'
        
        #이미지 변환 형태 지정
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

        if usage not in list(transform_list.keys()):
            raise ValueError("Transformation에 존재하지 않는 사용 형태입니다.")

        self.transform = transform_list[self.usage]
        
        # 클래스 이름과 클래스 인덱스 매핑
        self.class_names = ['defective', 'good']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        self.s3_client = boto3.client('s3')
        self.imgs = self._load_images()

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

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        key, class_idx = self.imgs[idx]
        image = self._load_image(key)
        if self.transform:
            image = self.transform(image)
        return image, class_idx


def build_set_loaders(bucket_name, version):
    train_dataset = S3ImageDatasets(bucket_name=bucket_name,version=version,usage='train')
    test_dataset = S3ImageDatasets(bucket_name=bucket_name,version=version,usage='test')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    return train_dataset, test_dataset, train_loader, test_loader