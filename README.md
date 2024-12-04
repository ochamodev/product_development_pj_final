# ♻️ RecycAI 🌱

**RecycAI** es nuestra aplicación que utiliza AI para apoyar en la clasificación de residuos, contribuyendo a una Guatemala más limpia y sostenible. 🌍✨

---

## 🚀 Descripción

RecycAI es una aplicación que clasifica imágenes de residuos en **orgánicos** e **inorgánicos** y determina el tipo de material (plástico, papel, cartón, etc.). Esto busca facilitar la correcta separación de residuos y apoyar el cumplimiento de las nuevas leyes de reciclaje del país. 🌿

---

## 📂 Estructura

### 🖥️ `api/`
Backend de la aplicación, desarrollado con **FastAPI**. Aquí cargamos el modelo de clasificación y procesamos imágenes enviadas por los usuarios para devolver la categoría correspondiente ♻️

### 📁 `model/`
El modelo de clasificación entrenado y listo para usar. Desplegamos dos modelos, uno basado en [CircularNet]([https://github.com/ochamodev/pd_app_ui/tree/main](https://github.com/tensorflow/models/tree/master/official/projects/waste_identification_ml)) y otro basado en [Garbage Classification](https://huggingface.co/yangy50/garbage-classification) 🤖

### 📓 Notebooks
Una colección de notebooks con pruebas de los modelos de clasificación 📊

### 🐳 `Dockerfile`
Instrucciones para crear un contenedor Docker y desplegar la API 🛠️

### 🔗 Código de la Aplicación Flutter
El código fuente de la aplicación móvil creada en **Flutter** se encuentra en [este repositorio](https://github.com/ochamodev/pd_app_ui/tree/main). La aplicación permite a los usuarios tomar foto de sus residuos y enviarlas al backend para clasificación 📱

---

## 🛠️ Instalación y Uso

### 1️⃣ Requisitos
- Python 3.8+
- Docker 🐳
- FastAPI 🚀
- Flutter (app móvil) 📱

### 2️⃣ Ejecutar la API
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/recycai.git
   ```
2. Construye y ejecuta el contenedor Docker:
   ```bash
   docker build -t recycai-api .
   docker run -p 8000:8000 recycai-api
   ```
3. Accede a la API en: [http://localhost:8000/docs](http://localhost:8000/docs) para probar los endpoints 🌐

### 3️⃣ Clasificación de Residuos
Envía una imagen a través de la API para recibir la clasificación 📸

---

## 📝 Licencia

Este proyecto está bajo la licencia MIT 📜
