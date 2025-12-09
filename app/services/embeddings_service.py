import numpy as np
from PIL import Image
from numpy.linalg import norm
from models.facenet import get_facenet_model


def generate_embeddings(image_file):
    try:
        import os
        import cv2
        from PIL import Image
        import numpy as np

        # LOAD DO MODELO SOMENTE QUANDO O WORKER CHAMAR
        model = get_facenet_model()

        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image).astype(np.uint8)

        detections = model.extract(image_np, threshold=0.95)
        if not detections:
            return {"error": "Nenhum rosto detectado na imagem."}, 400

        image_copy = image_np.copy()
        all_boxes = []

        for det in detections:
            box = det["box"]
            all_boxes.append(box)

            try:
                cv2.rectangle(image_copy, box, (255, 0, 0), 3)
            except Exception:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)

        embedding = detections[0]["embedding"]

        # salvar imagem
        save_dir = "processed_faces"
        os.makedirs(save_dir, exist_ok=True)

        name = getattr(image_file, "filename", None) or "image.jpg"
        base, ext = os.path.splitext(name)
        out_name = f"{base}_processed{ext}"

        save_path = os.path.join(save_dir, out_name)
        Image.fromarray(image_copy).save(save_path)

        return {
            "embedding": np.array(embedding).tolist(),
            "boxes": all_boxes,
            "processed_image_path": save_path
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

def detect_and_search_faces(image_file, top_k=3):
    try:
        import os
        import cv2
        from PIL import Image
        import numpy as np
        from app.services.milvus_service import search_similar_faces

        model = get_facenet_model()

        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image).astype(np.uint8)

        detections = model.extract(image_np, threshold=0.95)
        if not detections:
            return {"error": "Nenhum rosto detectado na imagem."}, 400

        boxes = []
        matches = []
        distances = []
        image_copy = image_np.copy()

        for det in detections:
            box = det["box"]
            embedding = det["embedding"]

            boxes.append(box)

            best = search_similar_faces(embedding, top_k=top_k)
            if best:
                distances.append(best[0]["distance"])
                matches.append(best[0])
            else:
                distances.append(float("inf"))
                matches.append(None)

        winner_index = int(np.argmin(distances))
        winner_box = boxes[winner_index]
        winner_match = matches[winner_index]

        # desenhar
        for i, box in enumerate(boxes):
            color = (255, 0, 0) if i == winner_index else (0, 0, 255)
            try:
                cv2.rectangle(image_copy, box, color, 3)
            except:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 3)

        save_dir = "processed_faces"
        os.makedirs(save_dir, exist_ok=True)

        name = getattr(image_file, "filename", None) or "search.jpg"
        base, ext = os.path.splitext(name)
        out_name = f"{base}_search_processed{ext}"

        save_path = os.path.join(save_dir, out_name)
        Image.fromarray(image_copy).save(save_path)

        return {
            "processed_image_path": save_path,
            "boxes": boxes,
            "winner_box": winner_box,
            "winner_index": winner_index,
            "winner_match": winner_match
        }, 200

    except Exception as e:
        return {"error": str(e)}, 500
