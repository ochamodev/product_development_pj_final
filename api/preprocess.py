'''
PREPROCESS PIPELINE
'''

# library import
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

async def preprocess_image(image_file):
    contents = await image_file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert to tensor and normalize to [0, 1]
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor