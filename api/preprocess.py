'''
PREPROCESS PIPELINE
'''

# library import
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

async def preprocess_image(contents):
    image = Image.open(BytesIO(contents)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert to tensor and normalize to [0, 1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor