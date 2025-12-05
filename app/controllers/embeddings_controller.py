from flask import Blueprint, request, jsonify
from app.services.embeddings_service import generate_embeddings, compare_embeddings

embeddings_bp = Blueprint("embeddings", __name__)

#  Rota para gerar embeddings de uma imagem
@embeddings_bp.route("/embeddings", methods=["POST"])
def embeddings():
    """
    Gera embeddings faciais a partir de uma imagem enviada.

    Request (multipart/form-data):
        image (file): Arquivo de imagem contendo um rosto.

    Returns:
        tuple:
            - dict: Contém o embedding gerado ou mensagem de erro.
            - int: Código de status HTTP.

    Raises:
        Exception: Propaga erros internos caso ocorram durante o processamento.
    """
    if "image" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado. Use o campo 'image'."}), 400

    image_file = request.files["image"]
    result, status = generate_embeddings(image_file)
    return jsonify(result), status


#  Rota para comparar duas imagens
@embeddings_bp.route("/compare", methods=["POST"])
def compare():
    """
    Compara duas imagens faciais para verificar similaridade entre embeddings.

    Request (multipart/form-data):
        image1 (file): Primeira imagem contendo um rosto.
        image2 (file): Segunda imagem contendo um rosto.

    Returns:
        tuple:
            - dict: Estrutura contendo:
                - distance (float): Distância euclidiana entre embeddings.
                - same_person (bool): True se a distância for menor que o threshold.
                - error (str, optional): Caso a extração de embeddings falhe.
            - int: Código de status HTTP.

    Raises:
        Exception: Caso ocorra falha interna no processo de comparação.
    """ 
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Envie duas imagens nos campos 'image1' e 'image2'."}), 400

    image1 = request.files["image1"]
    image2 = request.files["image2"]

    result, status = compare_embeddings(image1, image2)
    return jsonify(result), status