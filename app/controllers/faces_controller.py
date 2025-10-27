from flask import Blueprint, request, jsonify
from app.services.embeddings_service import generate_embeddings
from app.services.milvus_service import insert_face, search_similar_faces
import traceback

faces_bp = Blueprint("faces", __name__)

# ============================================================
# 🔹 Rota 1 - Registrar suspeito (imagem associada)
# ============================================================
@faces_bp.route("/faces/register", methods=["POST"])
def register_face():
    """
    Recebe uma imagem associada a um suspeito e salva o embedding no Milvus.
    """
    try:
        # Verifica se imagem foi enviada
        if "image" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado. Use o campo 'image'."}), 400

        image_file = request.files["image"]
        suspect_id = request.form.get("suspect_id")

        if not suspect_id:
            return jsonify({"error": "Campo 'suspect_id' é obrigatório."}), 400

        # Gera o embedding usando o serviço existente
        embedding_result, status = generate_embeddings(image_file)
        if status != 200:
            return jsonify(embedding_result), status

        embedding = embedding_result["embedding"]

        # Insere o embedding no Milvus
        face_id = insert_face(suspect_id=suspect_id, embedding=embedding, is_query=False)

        return jsonify({
            "message": "Face registrada com sucesso.",
            "face_id": face_id,
            "suspect_id": suspect_id
        }), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


# ============================================================
# 🔹 Rota 2 - Buscar suspeitos semelhantes (imagem anônima)
# ============================================================
@faces_bp.route("/faces/search", methods=["POST"])
def search_faces():
    """
    Recebe uma imagem (não associada) e retorna suspeitos semelhantes.
    Também salva a face de busca no Milvus com flag 'is_query=True'.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado. Use o campo 'image'."}), 400

        image_file = request.files["image"]
        top_k = int(request.form.get("top_k", 5))

        # Gera o embedding da imagem recebida
        embedding_result, status = generate_embeddings(image_file)
        if status != 200:
            return jsonify(embedding_result), status

        embedding = embedding_result["embedding"]

        # Busca faces semelhantes no Milvus
        matches = search_similar_faces(embedding, top_k=top_k)

        # Salva também essa face no Milvus (como consulta anônima)
        query_face_id = insert_face(suspect_id=None, embedding=embedding, is_query=True)

        return jsonify({
            "query_face_id": query_face_id,
            "matches": matches
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500
