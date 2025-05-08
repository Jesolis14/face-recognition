import os
import io
import base64
import requests
import numpy as np
import onnxruntime as rt
from flask import Flask, render_template, request, jsonify
from PIL import Image
from numpy.linalg import norm
import pickle
import cv2

# URL pública de tu modelo ONNX (sustituye por la tuya)
MODEL_URL = "https://mi-cdn.com/modelos/facenet.onnx"

# Descarga y carga el modelo ONNX (solo una vez en cold start)
_model_bytes = requests.get(MODEL_URL).content
_sess = rt.InferenceSession(_model_bytes)

def preprocess(img_array):
    """
    Redimensiona y normaliza la imagen para el modelo ONNX.
    Ajusta este preprocess al formato que tu modelo espera.
    """
    img = Image.fromarray(img_array).convert("RGB").resize((160, 160))
    x = np.array(img).astype(np.float32) / 255.0
    # Si tu modelo espera shape [1,3,160,160], descomenta la línea siguiente:
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x

def embed(img_array):
    """
    Ejecuta el modelo ONNX para obtener el embedding de la cara.
    Asegúrate de que 'input' coincida con el nombre real de la entrada en tu gráfico ONNX.
    """
    x = preprocess(img_array)
    return _sess.run(None, {"input": x})[0][0]

# Inicializa Flask y señala carpeta de plantillas
app = Flask(__name__, template_folder="../templates")

# Ruta al pickle de embeddings precomputados
emb_path = os.path.join(os.path.dirname(__file__), "../embeddings.pkl")
with open(emb_path, "rb") as f:
    base_datos = pickle.load(f)

# Carga del clasificador de rostros
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def reconocer_persona(img_array, base_datos, umbral=0.8):
    """
    Dado un recorte de cara, obtiene su embedding y lo compara
    contra la base de datos retornando el nombre y la distancia mínima.
    """
    embedding = embed(img_array)

    nombre_ident = "Desconocido"
    dist_min = float("inf")

    for nombre, emb_list in base_datos.items():
        for emb_base in emb_list:
            d = norm(embedding - emb_base)
            if d < dist_min and d < umbral:
                dist_min, nombre_ident = d, nombre

    return nombre_ident, dist_min

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/reconocer", methods=["POST"])
def reconocer():
    try:
        data = request.json
        image_data = data["image"].split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image_array = np.array(image)

        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        resultados = []
        for (x, y, w, h) in faces:
            face_img = image_array[y : y + h, x : x + w]
            nombre, distancia = reconocer_persona(face_img, base_datos)
            resultados.append({
                "nombre": nombre,
                "distancia": float(distancia),
                "bbox": [int(x), int(y), int(w), int(h)]
            })

        return jsonify(success=True, resultados=resultados)

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

if __name__ == "__main__":
    # Para desarrollo local con "python api/index.py"
    app.run(debug=True, host="0.0.0.0", port=5000)

