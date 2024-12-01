'''
MODEL HANDLER
'''

import torch
from torch import nn
import numpy as np


class WasteClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        from torchvision.models.feature_extraction import create_feature_extractor
        
        self.mobnet = mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.feature_extraction = create_feature_extractor(self.mobnet, return_nodes={'features.12': 'mob_feature'})
        self.conv1 = nn.Conv2d(576, 300, 3)
        self.fc1 = nn.Linear(10800, 30)
        self.dr = nn.Dropout()

    def forward(self, x):
        feature_layer = self.feature_extraction(x)['mob_feature']
        x = nn.functional.relu(self.conv1(feature_layer))
        x = x.flatten(start_dim=1)
        x = self.dr(x)
        output = self.fc1(x)
        return output


# Load PyTorch model
pt_model = WasteClassificationModel()
state_dict = torch.load("../model/waste_classification_model.pt", map_location=torch.device('cpu'), weights_only=True)
pt_model.load_state_dict(state_dict)
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