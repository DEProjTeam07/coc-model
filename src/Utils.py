from pathlib import Path
import os
from torchvision import transforms
from collections import defaultdict

# s3에서 데이터 받아오기



# 디렉토리 하위 파일 구조 확인


directory_image_count = defaultdict(int)

for path in image_keys:
    parts = path.split('/')
    directory = '/'.join(parts[:-1])  # 마지막 부분(파일명)을 제외한 경로가 디렉터리
    directory_image_count[directory] += 1

for directory, count in directory_image_count.items():
    print(f"There are {count} images in '{directory}'.")



