import numpy as np
from PIL import Image
from numpy.linalg import norm
from app.models.facenet import facenet_model


def generate_embeddings(image_file):
    """
    Recebe um arquivo de imagem e retorna os embeddings do rosto encontrado.
    """
    try:
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image).astype("float32")

        detections = facenet_model.extract(image_np, threshold=0.95)

        if not detections:
            return {"error": "Nenhum rosto detectado na imagem."}, 400

        embedding = detections[0]["embedding"]

        return {
            "embedding": np.array(embedding).tolist(),
            "box": detections[0]["box"]
        }, 200

    except Exception as e:
        return {"error": str(e)}, 500


def compare_embeddings(image1_file, image2_file, threshold=0.7):
    """
    Compara duas imagens e retorna a similaridade entre os embeddings.
    """
    try:
        emb1, status1 = generate_embeddings(image1_file)
        emb2, status2 = generate_embeddings(image2_file)

        if status1 != 200 or status2 != 200:
            return {"error": "Não foi possível extrair embeddings de uma das imagens."}, 400

        v1 = np.array(emb1["embedding"])
        v2 = np.array(emb2["embedding"])

        # Distância euclidiana
        distance = norm(v1 - v2)
        same_person = bool(distance < threshold)  # 👈 conversão para bool do Python
        
        return {
            "distance": float(distance),
            "same_person": same_person
        }, 200


    except Exception as e:
        return {"error": str(e)}, 500
