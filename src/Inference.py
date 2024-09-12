from PIL import Image
import torch
from torchvision import transforms
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


transform = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

def get_image_paths(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

def predict_image(image_path, model,device):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

def infer_images_in_folder(folder_path, model,device):
    image_paths = get_image_paths(folder_path)
    results = []
    for image_path in image_paths:
        predicted_class = predict_image(image_path, model,device)
        results.append({
            'image_path': image_path,
            'predicted_class': predicted_class
        })
    return pd.DataFrame(results)

def save_to_parquet(df, output_file):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
    print(f'Results saved to {output_file}')