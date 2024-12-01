'''
MODEL HANDLER
'''

import tensorflow as tf
import torch
import numpy as np

# Load TensorFlow/Keras models
keras_model1 = tf.keras.models.load_model("../model/fine_tune_resnet.keras")
keras_model2 = tf.keras.models.load_model("../model/fine_tune_resnet_epochs40.keras")

# Load PyTorch model
pt_model = torch.load("../model/waste_classification_model.pt")
pt_model.eval()

# Subcategory and category mapping
CLASS_MAPPING = {
    1: {"categoria": "Inorgánico", "nombre": "Aerosoles"},
    2: {"categoria": "Inorgánico", "nombre": "Latas de comida de aluminio"},
    3: {"categoria": "Inorgánico", "nombre": "Latas de refresco de aluminio"},
    4: {"categoria": "Orgánico", "nombre": "Cajas de cartón"},
    5: {"categoria": "Orgánico", "nombre": "Empaques de cartón"},
    6: {"categoria": "Inorgánico", "nombre": "Ropa"},
    7: {"categoria": "Orgánico", "nombre": "Residuos de café"},
    8: {"categoria": "Inorgánico", "nombre": "Cubiertos de plástico desechables"},
    9: {"categoria": "Orgánico", "nombre": "Cáscaras de huevo"},
    10: {"categoria": "Orgánico", "nombre": "Residuos de comida"},
    11: {"categoria": "Inorgánico", "nombre": "Botellas de vidrio para bebidas"},
    12: {"categoria": "Inorgánico", "nombre": "Envases cosméticos de vidrio"},
    13: {"categoria": "Inorgánico", "nombre": "Frascos de vidrio para alimentos"},
    14: {"categoria": "Orgánico", "nombre": "Revistas"},
    15: {"categoria": "Orgánico", "nombre": "Periódicos"},
    16: {"categoria": "Orgánico", "nombre": "Papel de oficina"},
    17: {"categoria": "Orgánico", "nombre": "Tazas de papel"},
    18: {"categoria": "Inorgánico", "nombre": "Tapas de plástico para tazas"},
    19: {"categoria": "Inorgánico", "nombre": "Botellas de detergente de plástico"},
    20: {"categoria": "Inorgánico", "nombre": "Envases de plástico para alimentos"},
    21: {"categoria": "Inorgánico", "nombre": "Bolsas de compras de plástico"},
    22: {"categoria": "Inorgánico", "nombre": "Botellas de refresco de plástico"},
    23: {"categoria": "Inorgánico", "nombre": "Popotes de plástico"},
    24: {"categoria": "Inorgánico", "nombre": "Bolsas de basura de plástico"},
    25: {"categoria": "Inorgánico", "nombre": "Botellas de agua de plástico"},
    26: {"categoria": "Inorgánico", "nombre": "Zapatos"},
    27: {"categoria": "Inorgánico", "nombre": "Latas de comida de acero"},
    28: {"categoria": "Inorgánico", "nombre": "Tazas de unicel (poliestireno)"},
    29: {"categoria": "Inorgánico", "nombre": "Envases de comida de unicel (poliestireno)"},
    30: {"categoria": "Orgánico", "nombre": "Bolsas de té"}
}

def map_to_category(class_id):
    mapping = CLASS_MAPPING.get(class_id)
    if mapping is None:
        return {"nombre": "Basura", "categoria": "Inorgánico"}
    
    return {
        "nombre": mapping["nombre"], 
        "categoria": mapping["categoria"]
    }
    
def predict(image_tensor):
    with torch.no_grad():
        # inference
        outputs = pt_model(image_tensor)
        class_id = torch.argmax(outputs).item() # get class index
    
    # map class to category and name
    return map_to_category(class_id)