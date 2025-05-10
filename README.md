# Reconocimiento Facial en Tiempo Real con Flask, FaceNet y MTCNN

Este proyecto implementa un sistema completo de reconocimiento facial en tiempo real usando **Flask** para el backend, **FaceNet** para la generación de embeddings y **MTCNN** para la detección de rostros. La interfaz web interactúa con la cámara del navegador y permite detectar si un rostro pertenece a una persona registrada, o mostrar "Acceso Denegado" si es desconocido.

---

## 📦 Requisitos del entorno

### Python
- Requiere **Python 3.10.13**
  - Esta versión fue elegida por su compatibilidad con TensorFlow 2.13.0 y las versiones estables de Keras, MTCNN y otras dependencias. 
  - Versiones más nuevas de Python (como 3.11 o 3.12) no son compatibles directamente con algunas de estas bibliotecas (especialmente `tensorflow` y `keras_facenet`).

### Librerías clave
- `tensorflow==2.13.0`
  - Compatible con Python 3.10
  - Evita conflictos con versiones recientes y asegura soporte para CPU sin necesidad de configuraciones avanzadas de CUDA.
- `numpy==1.24.*`
  - Última versión de NumPy compatible con TensorFlow 2.13.
  - Versiones más recientes (1.25+) rompen compatibilidad con algunas operaciones internas de TensorFlow.
- `keras-facenet`
  - Biblioteca que encapsula FaceNet preentrenado con un pipeline simple de embeddings faciales.
- `mtcnn`
  - Detector de rostros que funciona bien con imágenes RGB en tiempo real.

---

## 🛠️ Instalación del entorno virtual

### 1. Clona el repositorio
```bash
git clone https://github.com/Jesolis14/face-recognition.git
cd face-recognition
```

### 2. Crea y activa un entorno virtual (venv)
```bash
# Crear entorno virtual
python3.10 -m venv venv

# Activar en Linux/macOS
source venv/bin/activate

# Activar en Windows
venv\Scripts\activate
```

### 3. Instala las dependencias
```bash
pip install -r requirements.txt
```

> Si no tienes `python3.10` disponible, puedes instalarlo con `pyenv`, `conda` o desde los binarios oficiales de [python.org](https://www.python.org/downloads/release/python-31013/).

---

## 📁 Estructura del proyecto

```
face-recognition/
├── app.py                 # Servidor Flask
├── generar_embeddings.py  # Script para generar embeddings desde imágenes
├── embeddings.pkl         # Base de datos serializada de embeddings faciales
├── templates/
│   └── index.html         # Interfaz web (HTML + JS)
├── static/                # (opcional) Archivos JS/CSS estáticos
├── data/                  # Carpeta con subcarpetas por persona y sus imágenes
│   └── NombrePersona/
│       ├── img1.jpg
│       └── img2.png
├── requirements.txt       # Lista de dependencias
└── README.md
```

---

## 📸 Flujo del sistema

### 1. Generar embeddings (una vez)
Ejecuta el script para convertir imágenes faciales en vectores de características:
```bash
python generar_embeddings.py
```
Esto generará el archivo `embeddings.pkl` que será usado por el servidor Flask.

### 2. Iniciar el servidor Flask
```bash
python app.py
```
Esto abrirá un servidor en `http://localhost:5000`.

### 3. Acceder desde el navegador
Abre `http://localhost:5000` y acepta el acceso a la cámara.

- Si el rostro es reconocido → mostrará “Bienvenido, Nombre”.
- Si el rostro es desconocido → mostrará “Acceso Denegado” y lo etiquetará en rojo.

---

## 📥 Recolectar datos

Organiza tus imágenes en `data/{nombre_persona}/*.jpg`. Por ejemplo:
```
data/
├── Maria/
│   ├── 1.jpg
│   └── 2.jpg
├── Juan/
│   ├── 1.jpg
│   └── selfie.png
```

Luego corre nuevamente `generar_embeddings.py` para actualizar la base.

---

## ✅ Consideraciones

- El sistema funciona mejor con rostros bien iluminados y de frente.
- Evita usar imágenes borrosas o demasiado pequeñas.
- Se recomienda capturar al menos **3 imágenes por persona** para mejorar precisión.
- Si se reciben imágenes con canal alfa (RGBA), el script las convierte a RGB automáticamente.


---

## 🧠 Créditos
- [Keras-FaceNet](https://github.com/nyoki-mtl/keras-facenet)
- [MTCNN](https://github.com/ipazc/mtcnn)
- Proyecto desarrollado por Jesús M. Solís Durán.

---

¿Tienes dudas? ¿Quieres extender el sistema con reconocimiento múltiple o autenticación biométrica? ¡Estoy para ayudarte!