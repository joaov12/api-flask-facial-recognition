from flask import Blueprint, request, jsonify
from app.services.embeddings_service import generate_embeddings
from app.services.milvus_service import *
import boto3
from io import BytesIO
import json, traceback, requests

faces_bp = Blueprint("faces", __name__)

# ============================================================
# 🔹 Rota 1 - Registrar suspeito (imagem associada)
# ============================================================
@faces_bp.route("/faces/register", methods=["POST"])
def register_face():
    """
    Registra uma face de suspeito.
    Fonte da imagem:
      - Upload local (campo 'image')
      - S3 privado via boto3 (campo 's3_path': s3://bucket/key)
    """
    try:
        data = request.get_json(silent=True) or request.form
        suspect_id = data.get("suspect_id")
        metadata_raw = data.get("metadata")
        s3_path = data.get("s3_path")  # ✅ único campo do S3

        if not suspect_id:
            return jsonify({"error": "Campo 'suspect_id' é obrigatório."}), 400

        # 🔹 Metadados opcionais
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            metadata = {"raw": metadata_raw}

        image_file = None

        # =====================================================
        # 🔹 1️⃣ Upload local (form-data)
        # =====================================================
        if "image" in request.files:
            image_file = request.files["image"]

        # =====================================================
        # 🔹 2️⃣ Imagem do S3 via boto3 (s3://bucket/key)
        # =====================================================
        elif s3_path:
            if not s3_path.startswith("s3://"):
                return jsonify({"error": "Formato inválido em 's3_path'. Use s3://bucket/key"}), 400

            # 🔹 Quebra o caminho em bucket e key
            parts = s3_path.replace("s3://", "").split("/", 1)
            if len(parts) != 2:
                return jsonify({"error": "s3_path deve conter bucket e key"}), 400

            bucket, key = parts
            metadata["s3_source"] = s3_path

            # Apenas para ambiente local
            s3 = boto3.client(
                "s3",
                aws_access_key_id="AKIAZVDHMZWLMJNM2UNJ",
                aws_secret_access_key="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",
                region_name="us-east-2"
            )

            buffer = BytesIO()
            s3.download_fileobj(bucket, key, buffer)
            buffer.seek(0)
            buffer.name = key.split("/")[-1]
            image_file = buffer

        else:
            return jsonify({
                "error": "Envie uma imagem (campo 'image') ou um 's3_path'."
            }), 400

        # =====================================================
        # 🔹 Gera embedding
        # =====================================================
        embedding_result, status = generate_embeddings(image_file)
        if status != 200:
            return jsonify(embedding_result), status

        embedding = embedding_result["embedding"]

        # =====================================================
        # 🔹 Insere no Milvus
        # =====================================================
        face_id = insert_face(
            suspect_id=suspect_id,
            embedding=embedding,
            is_query=False,
            metadata=metadata,
            s3_path=s3_path if s3_path else None
        )

        return jsonify({
            "message": "Face registrada com sucesso.",
            "face_id": face_id,
            "suspect_id": suspect_id,
            "s3_path": s3_path,
            "source": "s3" if s3_path else "upload"
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
    Recebe uma imagem (upload local ou via S3) e retorna suspeitos semelhantes.
    Também salva a face de busca no Milvus com flag 'is_query=True'.
    """
    try:
        import boto3
        from io import BytesIO
        import json

        # Aceita tanto JSON (S3) quanto form-data (upload)
        data = request.get_json(silent=True) or request.form

        s3_path = data.get("s3_path")  # ✅ caminho completo no formato s3://bucket/key
        top_k = int(data.get("top_k", 5))
        image_file = None

        # =====================================================
        # 🔹 1️⃣ Upload local (form-data)
        # =====================================================
        if "image" in request.files:
            image_file = request.files["image"]

        # =====================================================
        # 🔹 2️⃣ Imagem no S3 (via boto3)
        # =====================================================
        elif s3_path:
            if not s3_path.startswith("s3://"):
                return jsonify({"error": "Formato inválido em 's3_path'. Use s3://bucket/key"}), 400

            # Divide o caminho em bucket e key
            parts = s3_path.replace("s3://", "").split("/", 1)
            if len(parts) != 2:
                return jsonify({"error": "s3_path deve conter bucket e key"}), 400

            bucket, key = parts

            # Apenas para ambiente local (use variáveis de ambiente em prod)
            s3 = boto3.client(
                "s3",
                aws_access_key_id="AKIAZVDHMZWLMJNM2UNJ",
                aws_secret_access_key="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",
                region_name="us-east-2"
            )

            buffer = BytesIO()
            s3.download_fileobj(bucket, key, buffer)
            buffer.seek(0)
            buffer.name = key.split("/")[-1]
            image_file = buffer

        else:
            return jsonify({
                "error": "Envie uma imagem (campo 'image') ou um 's3_path'."
            }), 400

        # =====================================================
        # 🔹 Gera o embedding da imagem
        # =====================================================
        embedding_result, status = generate_embeddings(image_file)
        if status != 200:
            return jsonify(embedding_result), status

        embedding = embedding_result["embedding"]

        # =====================================================
        # 🔹 Busca as faces mais semelhantes (somente is_query=False)
        # =====================================================
        matches = search_similar_faces(embedding, top_k=top_k)

        # =====================================================
        # 🔹 Salva essa face de busca no Milvus (is_query=True)
        # =====================================================
        query_face_id = insert_face(
            suspect_id=None,
            embedding=embedding,
            is_query=True,
            metadata={"s3_path": s3_path} if s3_path else {},
            s3_path=s3_path
        )

        return jsonify({
            "query_face_id": query_face_id,
            "source": "s3" if s3_path else "upload",
            "matches": matches
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


# ============================================================
# 🔹 Rota 3 - Listar todos os suspeitos e suas faces
# ============================================================
@faces_bp.route("/faces/suspects", methods=["GET"])
def list_all_suspects():
    """
    Lista todos os suspeitos cadastrados e suas faces associadas.
    Inclui s3_path e source.
    """
    try:
        from app.services.milvus_service import connect_milvus, COLLECTION_NAME
        from pymilvus import Collection, utility
        import json

        connect_milvus()

        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"error": f"Collection '{COLLECTION_NAME}' não existe."}), 404

        collection = Collection(COLLECTION_NAME)
        collection.load()

        results = collection.query(
            expr="is_query == false",
            output_fields=[
                "face_id", "suspect_id", "timestamp", "metadata", "s3_path"
            ]
        )

        suspects = {}
        for face in results:
            sid = face["suspect_id"]

            try:
                face["metadata"] = json.loads(face["metadata"])
            except Exception:
                pass

            face["source"] = "s3" if face.get("s3_path") else "upload"

            if sid not in suspects:
                suspects[sid] = []
            suspects[sid].append(face)

        return jsonify({
            "total_suspects": len(suspects),
            "suspects": [
                {"suspect_id": sid, "faces": faces}
                for sid, faces in suspects.items()
            ]
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500
    


# ============================================================
# 🔹 Rota 4 - Listar todas as faces de um suspeito específico
# ============================================================
@faces_bp.route("/faces/suspects/<int:suspect_id>", methods=["GET"])
def list_faces_by_suspect(suspect_id):
    """
    Lista todas as faces associadas a um suspeito específico.
    Inclui s3_path e source.
    """
    try:
        from app.services.milvus_service import connect_milvus, COLLECTION_NAME
        from pymilvus import Collection, utility
        import json

        connect_milvus()

        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"error": f"Collection '{COLLECTION_NAME}' não existe."}), 404

        collection = Collection(COLLECTION_NAME)
        collection.load()

        expr = f"is_query == false and suspect_id == {suspect_id}"
        results = collection.query(
            expr=expr,
            output_fields=["face_id", "timestamp", "metadata", "s3_path"]
        )

        if not results:
            return jsonify({
                "suspect_id": suspect_id,
                "faces": [],
                "message": "Nenhuma face encontrada para este suspeito."
            }), 404

        for r in results:
            try:
                r["metadata"] = json.loads(r["metadata"])
            except Exception:
                pass
            r["source"] = "s3" if r.get("s3_path") else "upload"

        return jsonify({
            "suspect_id": suspect_id,
            "faces": results,
            "total_faces": len(results)
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


# ============================================================
# 🔹 Rota 5 - Listar todos os embeddings cadastrados
# ============================================================
@faces_bp.route("/faces/all", methods=["GET"])
def list_all_faces():
    """
    Lista todos os embeddings cadastrados no Milvus (suspeitos + queries).
    Suporta limite opcional via query param (?limit=100).
    Agora exibe também o campo 's3_path' e 'source'.
    """
    try:
        from app.services.milvus_service import connect_milvus, COLLECTION_NAME
        from pymilvus import Collection, utility
        import json

        connect_milvus()

        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"error": f"Collection '{COLLECTION_NAME}' não existe."}), 404

        collection = Collection(COLLECTION_NAME)
        collection.load()

        limit = int(request.args.get("limit", 1000))

        results = collection.query(
            expr="",
            limit=limit,
            output_fields=[
                "face_id", "suspect_id", "is_query",
                "timestamp", "metadata", "s3_path"
            ]
        )

        # 🔹 Processa metadados e define o campo "source"
        for r in results:
            try:
                r["metadata"] = json.loads(r["metadata"])
            except Exception:
                pass
            r["source"] = "s3" if r.get("s3_path") else "upload"

        return jsonify({
            "total_faces": len(results),
            "faces": results
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500



# ============================================================
# 🔹 Rota 6 - Remove um embedding específico
# ============================================================
@faces_bp.route("/faces/delete/<int:face_id>", methods=["DELETE"])
def delete_face(face_id):
    """
    Remove um embedding específico do Milvus.
    """
    try:
        from app.services.milvus_service import connect_milvus, COLLECTION_NAME
        from pymilvus import Collection, utility

        connect_milvus()

        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"error": f"Collection '{COLLECTION_NAME}' não existe."}), 404

        collection = Collection(COLLECTION_NAME)
        collection.load()

        expr = f"face_id == {face_id}"
        delete_result = collection.delete(expr)

        return jsonify({
            "message": f"Face com ID {face_id} removida (se existente).",
            "delete_count": getattr(delete_result, 'delete_count', None)
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


# ============================================================
# 🔹 Rota 7 - Limpa toda a collection do Milvus
# ============================================================
@faces_bp.route("/faces/clear", methods=["DELETE"])
def clear_collection():
    """
    Remove toda a collection 'faces' do Milvus (uso administrativo).
    """
    try:
        from app.services.milvus_service import connect_milvus, COLLECTION_NAME
        from pymilvus import utility

        connect_milvus()

        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"message": "Collection já inexistente."}), 200

        utility.drop_collection(COLLECTION_NAME)
        return jsonify({"message": f"Collection '{COLLECTION_NAME}' removida com sucesso."}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


# ============================================================
# 🔹 Rota 8 - Editar uma face específica
# ============================================================
@faces_bp.route("/faces/update/<int:face_id>", methods=["PUT"])
def update_face(face_id):
    """
    Atualiza o 'suspect_id' e/ou os 'metadata' de uma face específica no Milvus.
    """
    try:
        from app.services.milvus_service import connect_milvus, COLLECTION_NAME
        from pymilvus import Collection, utility
        import json

        data = request.get_json(silent=True) or request.form
        new_suspect_id = data.get("suspect_id")
        metadata_raw = data.get("metadata")

        connect_milvus()

        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"error": f"Collection '{COLLECTION_NAME}' não existe."}), 404

        collection = Collection(COLLECTION_NAME)
        collection.load()

        # 🔹 Busca os dados existentes (sem embedding)
        existing = collection.query(
            expr=f"face_id == {face_id}",
            output_fields=[
                "face_id", "suspect_id", "timestamp",
                "is_query", "metadata", "s3_path"
            ]
        )

        if not existing:
            return jsonify({"error": f"Face ID {face_id} não encontrada."}), 404

        current = existing[0]

        # 🔹 Busca o embedding separadamente
        emb_result = collection.query(
            expr=f"face_id == {face_id}",
            output_fields=["embedding"]
        )
        if not emb_result or "embedding" not in emb_result[0]:
            return jsonify({"error": "Não foi possível recuperar o embedding da face."}), 500

        embedding = emb_result[0]["embedding"]

        # 🔹 Determina novos valores
        updated_suspect_id = int(new_suspect_id) if new_suspect_id is not None else current["suspect_id"]

        # 🔹 Merge de metadados
        try:
            new_metadata = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            new_metadata = {"raw": metadata_raw}

        try:
            old_metadata = json.loads(current["metadata"])
        except Exception:
            old_metadata = {}

        merged_metadata = {**old_metadata, **new_metadata}

        # ============================================================
        # 🔹 1️⃣ Deleta o registro antigo
        # ============================================================
        collection.delete(expr=f"face_id == {face_id}")
        collection.flush()

        # ============================================================
        # 🔹 2️⃣ Reinsere com os novos dados
        # ============================================================
        data_insert = [
            [face_id],
            [updated_suspect_id],
            [embedding],
            [current.get("timestamp", 0)],
            [current.get("is_query", False)],
            [json.dumps(merged_metadata)],
            [current.get("s3_path", "")]
        ]

        collection.insert(data_insert)
        collection.flush()
        collection.load()

        return jsonify({
            "message": f"Face {face_id} atualizada com sucesso.",
            "face_id": face_id,
            "suspect_id": updated_suspect_id,
            "metadata": merged_metadata
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


# ============================================================
# 🔹 Rota 9 - Remove todas as faces associadas a um suspect_id
# ============================================================
@faces_bp.route("/faces/delete/suspect/<int:suspect_id>", methods=["DELETE"])
def delete_faces_by_suspect(suspect_id):
    """
    Remove todas as faces associadas a um suspeito específico (suspect_id).
    """
    try:
        from app.services.milvus_service import connect_milvus, COLLECTION_NAME
        from pymilvus import Collection, utility

        connect_milvus()

        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"error": f"Collection '{COLLECTION_NAME}' não existe."}), 404

        collection = Collection(COLLECTION_NAME)
        collection.load()

        # 🔹 Verifica se há registros do suspeito
        existing = collection.query(
            expr=f"suspect_id == {suspect_id}",
            output_fields=["face_id"]
        )

        if not existing:
            return jsonify({
                "message": f"Nenhuma face encontrada para suspect_id {suspect_id}."
            }), 404

        face_ids = [f["face_id"] for f in existing]
        collection.delete(expr=f"suspect_id == {suspect_id}")
        collection.flush()

        return jsonify({
            "message": f"Todas as faces associadas ao suspect_id {suspect_id} foram removidas.",
            "deleted_faces": face_ids,
            "total_deleted": len(face_ids)
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500
