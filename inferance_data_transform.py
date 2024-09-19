import json
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

def image_to_array(image_path):
    image = Image.open(image_path)
    image = transform(image)
    return image.numpy().tolist()

def create_request_payload(image_path):
    image_array = image_to_array(image_path)
    return {
        "instances": [image_array]
    }
def save_payload_to_json(payload, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(payload, json_file, indent=4)


image_path = '/home/ubuntu/coc-model/inference_dataset/inference1.jpeg'
data = create_request_payload(image_path)
save_payload_to_json(data, '/home/ubuntu/app/data.json')