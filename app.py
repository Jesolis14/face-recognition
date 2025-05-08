from flask import Flask, render_template, request, jsonify
import os
import base64
import io
import time
import pickle
import numpy as np
from PIL import Image
import cv2
from keras_facenet import FaceNet
from numpy.linalg import norm
from mtcnn.mtcnn import MTCNN

app = Flask(__name__)

# Pre-carga de recursos (modelo, detector y embeddings)
def preload_resources():
    global embedder, detector, base_datos

    print("⏳ Cargando FaceNet…")
    t0 = time.time()
    embedder = FaceNet()
    print(f"✅ FaceNet cargado en {time.time() - t0:.2f}s")

    print("⏳ Cargando detector MTCNN…")
    detector = MTCNN()
    print("✅ Detector MTCNN cargado")

    print("⏳ Cargando embeddings…")
    with open("embeddings.pkl", "rb") as f:
        base_datos = pickle.load(f)
    print(f"✅ Embeddings cargadas para {len(base_datos)} identidades")

# Llamada de pre-carga al iniciar
preload_resources()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/reconocer", methods=["POST"])
def reconocer():
    try:
        print("🔔 /reconocer invocado")
        t_start = time.time()

        # Obtener imagen base64 de JSON
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)

        # Convertir a arreglo numpy
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        print(f"🔍 Imagen recibida: shape={arr.shape}, dtype={arr.dtype}")

        # Detección con MTCNN
        detecciones = detector.detect_faces(arr)
        print(f"🔔 MTCNN detectó {len(detecciones)} rostros")

        resultados = []
        for cara in detecciones:
            x, y, w, h = cara["box"]
            face_img = arr[y:y+h, x:x+w]
            nombre, distancia = reconocer_persona(face_img)
            resultados.append({
                "nombre": nombre,
                "distancia": float(distancia),
                "bbox": [int(x), int(y), int(w), int(h)]
            })

        print(f"🔔 Procesado en {time.time() - t_start:.2f}s")
        return jsonify(success=True, resultados=resultados)

    except Exception as e:
        print("❌ Error en /reconocer:", e)
        return jsonify(success=False, error=str(e)), 500


def reconocer_persona(img_array, umbral=0.8):
    # Ajustar tamaño para FaceNet
    face_resized = cv2.resize(img_array, (160, 160))
    embedding = embedder.embeddings([face_resized])[0]

    nombre_identificado = "Desconocido"
    distancia_minima = float("inf")
    for nombre, emb_list in base_datos.items():
        for emb_base in emb_list:
            d = norm(embedding - emb_base)
            if d < distancia_minima and d < umbral:
                distancia_minima = d
                nombre_identificado = nombre
    return nombre_identificado, distancia_minima

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Arrancando servidor en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

