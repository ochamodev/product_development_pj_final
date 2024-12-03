from io import BytesIO

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from api.preprocess import preprocess_image

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
@app.get("/logs")
async def get_logs():
    try:
        with open("api_logs.log", "r") as log_file:
            logs = log_file.readlines()
        return {"logs": logs}
    except FileNotFoundError:
        return {"logs": []}

@app.get("/error")
async def raise_error():
    raise ValueError("This is a test error!")

@app.post("/identify")
async def identify(image: UploadFile) -> Dict[str, str]:

    contents = await image.read()
    image_tensor = await preprocess_image(contents)
    result = predict(image_tensor)
    #output_path = f"uploaded_images/{image.filename}"
    #img = Image.open(BytesIO(contents))
    #img.save(output_path)

    # Retornar la respuesta
    return result

# Ejecutar el servidor (solo para pruebas locales, usar uvicorn para producción)
if __name__ == "__main__":
    import uvicorn
    from api.model_handle import predict
    uvicorn.run(app, host="0.0.0.0", port=8000)
