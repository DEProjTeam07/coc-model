import boto3
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO
from torchvision import transforms


class S3ImageDatasets(Dataset):
    def __init__(self, bucket_name, usage):
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3')
        self.usage = usage

        self.image_keys = self.get_image_keys()
    
        transform_list = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if usage not in list(transform_list.keys()):
            raise ValueError("Transformation에 존재하지 않는 사용 형태입니다.")

        self.transform = transform_list[self.usage]

    def get_image_keys(self):
        image_keys = []
        obj_list = self.s3.list_objects(Bucket=self.bucket_name)
        for content in obj_list['Contents']:
            image_keys.append(content['Key'])
        return image_keys
    
    def __len__(self):
        return len(self.image_keys)
    
    def __getitem__(self, idx):
        image_key = self.image_keys[idx]
        response = self.s3.get_object(Bucket=self.bucket_name, Key=image_key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data))

        if self.transform:
            image = self.transform(image)

        return image