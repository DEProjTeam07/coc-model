from pathlib import Path
import os
from torchvision import transforms

# s3에서 데이터 받아오기



# 디렉토리 하위 파일 구조 확인
def print_filestruct(img_path):
    for dirpath, dirnames, filenames in os.walk(img_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# 이미지 변환
transform = transforms.Compose([    
                                transforms.Resize(size = (64,64)),
                                transforms.ToTensor()
                                ])
