from keras_facenet import FaceNet
import numpy as np
from PIL import Image
import os
import pickle

embedder = FaceNet()

def obtener_embedding(ruta):
    img = Image.open(ruta).resize((160, 160))
    embedding = embedder.embeddings([np.array(img)])
    return embedding[0]

dataset_dir = "data"
embeddings = {}

for persona in os.listdir(dataset_dir):
    embeddings[persona] = []
    persona_dir = os.path.join(dataset_dir, persona)
    
    for imagen in os.listdir(persona_dir):
        ruta_img = os.path.join(persona_dir, imagen)
        emb = obtener_embedding(ruta_img)
        embeddings[persona].append(emb)

# Guarda embeddings generados para usarlos después
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings generados correctamente.")
