# Usa una imagen base completa para evitar problemas
FROM python:3.10-slim

# Configura el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY api/ ./api/
COPY model/ ./model/
COPY api/requirements.txt .

# Instala las dependencias
RUN pip install --upgrade pip && pip install --timeout=120 -r requirements.txt

# Crea la carpeta para subir imágenes
RUN mkdir -p /app/uploaded_images && chmod -R 777 /app/uploaded_images

# Expone el puerto 8000
EXPOSE 8000

# Comando para ejecutar el servidor
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
