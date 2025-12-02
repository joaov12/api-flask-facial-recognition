import numpy as np
from PIL import Image
from numpy.linalg import norm
from app.models.facenet import facenet_model


def generate_embeddings(image_file):
    """
    Gera embeddings faciais a partir de uma imagem fornecida.

    Args:
        image_file (file-like): Arquivo de imagem enviado pelo usuário.

    Returns:
        tuple:
            - dict: Contém os embeddings do rosto detectado e a bounding box.
            - int: Código de status HTTP.
        
        Em caso de erro:
            - dict: Mensagem de erro.
            - int: Código de status HTTP (400 ou 500).

    Raises:
        Exception: Em caso de falha inesperada durante o processamento da imagem.
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
    Compara os embeddings gerados de duas imagens para verificar se representam a mesma pessoa.

    Args:
        image1_file (file-like): Arquivo da primeira imagem.
        image2_file (file-like): Arquivo da segunda imagem.
        threshold (float, optional): Valor limite para considerar que dois embeddings 
            pertencem à mesma pessoa. Quanto menor, mais rigorosa a comparação. 
            Default é 0.7.

    Returns:
        tuple:
            - dict: Contém a distância euclidiana entre os embeddings e um booleano 
              indicando se representam a mesma pessoa.
            - int: Código de status HTTP.
        
        Em caso de erro:
            - dict: Mensagem de erro.
            - int: Código de status HTTP (400 ou 500).

    Raises:
        Exception: Em caso de falha inesperada durante a comparação.
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
        same_person = bool(distance < threshold)
        
        return {
            "distance": float(distance),
            "same_person": same_person
        }, 200


    except Exception as e:
        return {"error": str(e)}, 500
