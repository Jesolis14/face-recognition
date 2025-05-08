from flask import Flask, render_template, request, jsonify
import os, base64, io, time, pickle
import numpy as np
from PIL import Image
import cv2
from keras_facenet import FaceNet
from numpy.linalg import norm

app = Flask(__name__)

# 1) Pre-cargar recursos al inicio
def preload_resources():
    global embedder, face_cascade, base_datos

    print("⏳ Cargando FaceNet…")
    t0 = time.time()
    embedder = FaceNet()
    print(f"✅ FaceNet cargado en {time.time() - t0:.2f}s")

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"⏳ Cargando cascada desde {cascade_path}…")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("✅ Cascade de rostros cargada")

    print("⏳ Cargando embeddings…")
    with open("embeddings.pkl", "rb") as f:
        base_datos = pickle.load(f)
    print(f"✅ Embeddings cargadas para {len(base_datos)} identidades")

preload_resources()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reconocer", methods=["POST"])
def reconocer():
    try:
        print("🔔 /reconocer invocado")
        t_start = time.time()

        # 2) Parsear JSON con la imagen en base64
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        if "," in image_data:
            # quitar el header "data:image/…;base64,"
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)

        # 3) Abrir con PIL y pasar a numpy
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)

        # 4) Detección de rostros en escala de grises (RGB→GRAY)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print(f"🔔 Detectados {len(faces)} rostros")

        resultados = []
        for (x, y, w, h) in faces:
            face_img = arr[y : y + h, x : x + w]
            nombre, dist = reconocer_persona(face_img)
            resultados.append({
                "nombre": nombre,
                "distancia": float(dist),
                "bbox": [int(x), int(y), int(w), int(h)]
            })

        print(f"🔔 Procesado en {time.time() - t_start:.2f}s")
        return jsonify(success=True, resultados=resultados)

    except Exception as e:
        print("❌ Error en /reconocer:", e)
        return jsonify(success=False, error=str(e)), 500


def reconocer_persona(img_array, umbral=0.8):
    # Redimensionar al input de FaceNet y extraer embedding
    face_resized = cv2.resize(img_array, (160, 160))
    embedding = embedder.embeddings([face_resized])[0]

    # Comparar con la base de datos
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
    print(f"Arrancando en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

