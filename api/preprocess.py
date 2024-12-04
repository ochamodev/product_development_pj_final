'''
PREPROCESS PIPELINE
'''

# library import
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

async def preprocess_image(contents):
    image = Image.open(BytesIO(contents)).convert("RGB")

    # preparar imagen al input shape que el modelo necesita
    input_size = (1024, 512)
    image_resized = image.resize(input_size, Image.LANCZOS)
    image_normalized = np.array(image_resized) / 255.0
    image_tensor = tf.convert_to_tensor(image_normalized, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension

    return image_tensor