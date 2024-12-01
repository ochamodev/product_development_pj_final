'''
PREPROCESS PIPELINE
'''

# library import
from PIL import Image
import torch
import torchvision.transforms as transforms

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert to tensor and normalize to [0, 1]
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor