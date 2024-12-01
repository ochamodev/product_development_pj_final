'''
MODEL HANDLER
'''

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
    1: {"classification": "Inorgánico", "type_of_material": "Aerosoles"},
    2: {"classification": "Inorgánico", "type_of_material": "Latas de comida de aluminio"},
    3: {"classification": "Inorgánico", "type_of_material": "Latas de refresco de aluminio"},
    4: {"classification": "Orgánico", "type_of_material": "Cajas de cartón"},
    5: {"classification": "Orgánico", "type_of_material": "Empaques de cartón"},
    6: {"classification": "Inorgánico", "type_of_material": "Ropa"},
    7: {"classification": "Orgánico", "type_of_material": "Residuos de café"},
    8: {"classification": "Inorgánico", "type_of_material": "Cubiertos de plástico desechables"},
    9: {"classification": "Orgánico", "type_of_material": "Cáscaras de huevo"},
    10: {"classification": "Orgánico", "type_of_material": "Residuos de comida"},
    11: {"classification": "Inorgánico", "type_of_material": "Botellas de vidrio para bebidas"},
    12: {"classification": "Inorgánico", "type_of_material": "Envases cosméticos de vidrio"},
    13: {"classification": "Inorgánico", "type_of_material": "Frascos de vidrio para alimentos"},
    14: {"classification": "Orgánico", "type_of_material": "Revistas"},
    15: {"classification": "Orgánico", "type_of_material": "Periódicos"},
    16: {"classification": "Orgánico", "type_of_material": "Papel de oficina"},
    17: {"classification": "Orgánico", "type_of_material": "Tazas de papel"},
    18: {"classification": "Inorgánico", "type_of_material": "Tapas de plástico para tazas"},
    19: {"classification": "Inorgánico", "type_of_material": "Botellas de detergente de plástico"},
    20: {"classification": "Inorgánico", "type_of_material": "Envases de plástico para alimentos"},
    21: {"classification": "Inorgánico", "type_of_material": "Bolsas de compras de plástico"},
    22: {"classification": "Inorgánico", "type_of_material": "Botellas de refresco de plástico"},
    23: {"classification": "Inorgánico", "type_of_material": "Popotes de plástico"},
    24: {"classification": "Inorgánico", "type_of_material": "Bolsas de basura de plástico"},
    25: {"classification": "Inorgánico", "type_of_material": "Botellas de agua de plástico"},
    26: {"classification": "Inorgánico", "type_of_material": "Zapatos"},
    27: {"classification": "Inorgánico", "type_of_material": "Latas de comida de acero"},
    28: {"classification": "Inorgánico", "type_of_material": "Tazas de unicel (poliestireno)"},
    29: {"classification": "Inorgánico", "type_of_material": "Envases de comida de unicel (poliestireno)"},
    30: {"classification": "Orgánico", "type_of_material": "Bolsas de té"}
}

def map_to_category(class_id):
    mapping = CLASS_MAPPING.get(class_id)
    if mapping is None:
        return {"type_of_material": "Basura", "classification": "Inorgánico"}
    
    return {
        "type_of_material": mapping["type_of_material"], 
        "classification": mapping["classification"]
    }
    
def predict(image_tensor):
    with torch.no_grad():
        # inference
        outputs = pt_model(image_tensor)
        class_id = torch.argmax(outputs).item() # get class index
    
    # map class to category and name
    return map_to_category(class_id)