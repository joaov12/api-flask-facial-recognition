from flask import Blueprint, request, jsonify
from app.services.embeddings_service import generate_embeddings, compare_embeddings

embeddings_bp = Blueprint("embeddings", __name__)

#  Rota para gerar embeddings de uma imagem
@embeddings_bp.route("/embeddings", methods=["POST"])
def embeddings():
    if "image" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado. Use o campo 'image'."}), 400

    image_file = request.files["image"]
    result, status = generate_embeddings(image_file)
    return jsonify(result), status


#  Rota para comparar duas imagens
@embeddings_bp.route("/compare", methods=["POST"])
def compare():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Envie duas imagens nos campos 'image1' e 'image2'."}), 400

    image1 = request.files["image1"]
    image2 = request.files["image2"]

    result, status = compare_embeddings(image1, image2)
    return jsonify(result), status
