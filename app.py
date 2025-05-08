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

app = Flask(__name__)

# Cargar el modelo FaceNet y las embeddings
embedder = FaceNet()
with open("embeddings.pkl", "rb") as f:
    base_datos = pickle.load(f)

def reconocer_persona(img_array, base_datos, umbral=0.8):
    # Redimensionar la imagen a 160x160 (tamaño requerido por FaceNet)
    img = Image.fromarray(img_array).resize((160, 160))
    embedding = embedder.embeddings([np.array(img)])[0]

    nombre_identificado = "Desconocido"
    distancia_minima = float('inf')

    for nombre, emb_lista in base_datos.items():
        for emb_base in emb_lista:
            distancia = norm(embedding - emb_base)
            if distancia < distancia_minima and distancia < umbral:
                distancia_minima = distancia
                nombre_identificado = nombre

    return nombre_identificado, distancia_minima

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/reconocer", methods=["POST"])
def reconocer():
    try:
        # Recibir la imagen en base64 desde el cliente
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convertir a imagen
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Detectar rostros usando OpenCV
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        resultados = []
        
        # Si se detectan rostros, reconocer cada uno
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = image_array[y:y+h, x:x+w]
                nombre, distancia = reconocer_persona(face_img, base_datos)
                resultados.append({
                    'nombre': nombre,
                    'distancia': float(distancia),
                    'bbox': [int(x), int(y), int(w), int(h)]
                })
        
        return jsonify({
            'success': True,
            'resultados': resultados
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
