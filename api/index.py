# api/index.py
from flask import Flask, render_template, request, jsonify
import os
import base64
import numpy as np
from PIL import Image
import io
from keras_facenet import FaceNet
from numpy.linalg import norm
import pickle
import cv2

app = Flask(__name__, template_folder="../templates")

# Carga FaceNet y tu base de embeddings
embedder = FaceNet()
with open("embeddings.pkl", "rb") as f:
    base_datos = pickle.load(f)

def reconocer_persona(img_array, base_datos, umbral=0.8):
    img = Image.fromarray(img_array).resize((160, 160)).convert("RGB")
    embedding = embedder.embeddings([np.array(img)])[0]

    nombre_identificado = "Desconocido"
    distancia_minima = float('inf')

    for nombre, emb_lista in base_datos.items():
        for emb_base in emb_lista:
            d = norm(embedding - emb_base)
            if d < distancia_minima and d < umbral:
                distancia_minima, nombre_identificado = d, nombre

    return nombre_identificado, distancia_minima

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/reconocer", methods=["POST"])
def reconocer():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image_array = np.array(image)

        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        resultados = []
        for (x, y, w, h) in faces:
            face_img = image_array[y:y+h, x:x+w]
            nombre, distancia = reconocer_persona(face_img, base_datos)
            resultados.append({
                'nombre': nombre,
                'distancia': float(distancia),
                'bbox': [int(x), int(y), int(w), int(h)]
            })

        return jsonify(success=True, resultados=resultados)

    except Exception as e:
        return jsonify(success=False, error=str(e))
