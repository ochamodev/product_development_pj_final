from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from preprocess import preprocess_image
from model_handle import predict

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

@app.post("/identify")
async def identify(image: UploadFile) -> Dict[str, str]:

    image_tensor = preprocess_image(image.file)
    result = predict(image_tensor)

    # Retornar la respuesta
    return result

# Ejecutar el servidor (solo para pruebas locales, usar uvicorn para producción)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
