'''
MODEL HANDLER
'''
import numpy as np
import absl.logging
import tensorflow as tf


# ignorar warnings (no estamos entrenando el modelo, se suprimen los warnings de ese tipo)
absl.logging.set_verbosity(absl.logging.ERROR)

# cargar modelo
MODEL_PATH = "./model"
circularnet_model = tf.saved_model.load(MODEL_PATH)
model_fn = circularnet_model.signatures["serving_default"]

# mapeo de material a clasificación
CLASS_MAPPING = {
    1: {"classification": "Inorgánico", "type_of_material": "Residuos inorgánicos"},
    2: {"classification": "Inorgánico", "type_of_material": "Textiles"},
    3: {"classification": "Inorgánico", "type_of_material": "Caucho y cuero"},
    4: {"classification": "Orgánico", "type_of_material": "Madera"},
    5: {"classification": "Orgánico", "type_of_material": "Alimentos"},
    6: {"classification": "Inorgánico", "type_of_material": "Plásticos"},
    7: {"classification": "Orgánico", "type_of_material": "Restos de plantas"},
    8: {"classification": "Orgánico", "type_of_material": "Papel y Cartón"},
    9: {"classification": "Inorgánico", "type_of_material": "Vidrio"},
    10: {"classification": "Inorgánico", "type_of_material": "Metales"},
}

# funcion para predecir nuevo resultado
def predict(image_tensor):
    results = model_fn(inputs=image_tensor)
    detection_scores = results["detection_scores"].numpy()[0]
    detection_classes = results["detection_classes"].numpy()[0].astype(int)

    # devolver clase con el mayor confidence score
    max_index = np.argmax(detection_scores)
    predicted_class_id = detection_classes[max_index]
    # en caso no haya una class_id, devolver respuesta default
    predicted_label = CLASS_MAPPING.get(predicted_class_id, {"classification": "Inorgánico", "type_of_material": "Basura"})

    return predicted_label