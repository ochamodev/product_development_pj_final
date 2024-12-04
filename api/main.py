from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from api.model_handle import predict as predict_pytorch
from api.preprocess import preprocess_image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
import logging

logging.basicConfig(filename="api_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")

app = FastAPI()

# Configurar CORS
origins = [
    "http://localhost",  # Cambiar por los dominios permitidos
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para registrar solicitudes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        log_message = {
            "method": request.method,
            "url": request.url.path,
            "status_code": response.status_code,
            "duration": f"{duration:.2f}s"
        }
        logging.info(f"SUCCESS: {log_message}")
        return response
    except Exception as e:
        duration = time.time() - start_time
        log_message = {
            "method": request.method,
            "url": request.url.path,
            "error": str(e),
            "duration": f"{duration:.2f}s"
        }
        logging.error(f"ERROR: {log_message}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error"}
        )

# Ruta para consultar los logs
@app.get("/logs")
async def get_logs():
    try:
        with open("api_logs.log", "r") as log_file:
            logs = log_file.readlines()
        return {"logs": logs}
    except FileNotFoundError:
        return {"logs": []}

# Ruta de prueba para errores
@app.get("/error")
async def raise_error():
    raise ValueError("This is a test error!")

# Cargar modelo de Hugging Face
model_name = "yangy50/garbage-classification"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
hf_model = AutoModelForImageClassification.from_pretrained(model_name)
labels_hf = ['cartón', 'vidrio', 'metal', 'papel', 'plástico', 'basura']

# Endpoint para identificar con ambos modelos
@app.post("/identify")
async def identify(image: UploadFile) -> Dict[str, Dict[str, str]]:
    # Leer la imagen
    contents = await image.read()

    # Predicción con el modelo PyTorch
    image_tensor = await preprocess_image(contents)
    pytorch_result = predict_pytorch(image_tensor)

    # Predicción con el modelo de Hugging Face
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    inputs = feature_extractor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        logits = hf_model(**inputs).logits
    hf_predicted_label = logits.argmax(-1).item()
    hugging_face_result = {"type_of_material": labels_hf[hf_predicted_label]}

    # Retornar ambas predicciones
    return {
        "pytorch_model": pytorch_result,
        "hugging_face_model": hugging_face_result
    }

# Ejecutar el servidor (solo para pruebas locales, usar uvicorn para producción)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
