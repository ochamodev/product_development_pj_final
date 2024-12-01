'''
PREPROCESS PIPELINE
'''

# library import
from PIL import Image
import numpy as np
import torch

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    
    # resize and normalize
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0

    # convert into tensor for pytorch models
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float()

    # convert into array for tf.keras models
    image_array_tf = np.expand_dims(image_array, axis=0)
    
    return image_array_tf, image_tensor