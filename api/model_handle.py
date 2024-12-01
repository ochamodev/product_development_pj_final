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
    1: "aerosol_cans", 2: "aluminum_food_cans", 3: "aluminum_soda_cans", 4: "cardboard_boxes", 5: "cardboard_packaging", 
    6: "clothing", 7: "coffee_grounds", 8: "disposable_plastic_cutlery", 9: "eggshells", 10: "food_waste", 
    11: "glass_beverage_bottles", 12: "glass_cosmetic_containers", 13: "glass_food_jars", 14: "magazines", 15: "newspaper", 
    16: "office_paper", 17: "paper_cups", 18: "plastic_cup_lids", 19: "plastic_detergent_bottles", 20: "plastic_food_containers", 
    21: "plastic_shopping_bags", 22: "plastic_soda_bottles", 23: "plastic_straws", 24: "plastic_trash_bags", 25: "plastic_water_bottles", 
    26: "shoes", 27: "steel_food_cans", 28: "styrofoam_cups", 29: "styrofoam_food_containers", 30: "tea_bags"
}

ORGANIC_CLASSES = ["Cardboard", "Paper", "Food Waste"]
INORGANIC_CLASSES = ["Plastic Bottle", "Glass Bottle", "Styrofoam"]