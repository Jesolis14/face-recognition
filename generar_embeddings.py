# Script para generar embeddings faciales a partir de un dataset usando FaceNet

from keras_facenet import FaceNet
import numpy as np
from PIL import Image
import os
import pickle

# Inicializar modelo FaceNet (usa MTCNN internamente para preprocesar)
embedder = FaceNet()

def obtener_embedding(ruta, force_rgb=False):
    """
    Dado el path de una imagen, devuelve su embedding facial de 512 dimensiones.

    Args:
        ruta (str): Ruta del archivo de imagen.
        force_rgb (bool): Si True, fuerza la conversión a RGB (útil para imágenes RGBA).

    Returns:
        np.ndarray: Vector de embedding facial.
    """
    img = Image.open(ruta)
    if force_rgb:
        img = img.convert('RGB')  # Eliminar canal alfa si lo tiene
    img = img.resize((160, 160))
    X = np.array(img)
    embedding = embedder.embeddings([X])
    return embedding[0]

# Directorio donde están organizadas las imágenes: data/NombrePersona/*.jpg
dataset_dir = "data"
embeddings = {}

# Iterar sobre cada carpeta de persona
for persona in os.listdir(dataset_dir):
    persona_dir = os.path.join(dataset_dir, persona)
    if not os.path.isdir(persona_dir):
        continue

    embeddings[persona] = []
    # Activar conversión RGB solo para casos conocidos problemáticos
    need_rgb = (persona.lower() == "ederlmira" or persona == "Edelmira")

    # Recorrer imágenes dentro de la carpeta de esa persona
    for imagen in os.listdir(persona_dir):
        if imagen.startswith('.') or not imagen.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        ruta_img = os.path.join(persona_dir, imagen)
        try:
            emb = obtener_embedding(ruta_img, force_rgb=need_rgb)
            embeddings[persona].append(emb)
        except Exception as e:
            print(f"❌ Error procesando {ruta_img}: {e}")

# Convertir los embeddings a listas serializables (no se puede guardar numpy directamente en pickle portable)
embeddings_serializables = {
    k: [e.tolist() for e in v] for k, v in embeddings.items()
}

# Guardar a archivo .pkl usando protocolo 4 para compatibilidad
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_serializables, f, protocol=4)

print("✅ Embeddings generados correctamente.")
