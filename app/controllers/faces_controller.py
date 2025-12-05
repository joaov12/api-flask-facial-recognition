from flask import Blueprint, request, jsonify
from app.services.embeddings_service import generate_embeddings
from app.services.milvus_service import *
import boto3
from io import BytesIO
import json, traceback, requests
from redis import Redis
from app.workers import process_register_face
from rq import Queue
from rq.job import Job

faces_bp = Blueprint("faces", __name__)

# Configura conexão e fila Redis
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(redis_url)
queue = Queue("faces_register_queue", connection=redis_conn)

# ============================================================
#  Rota 1 - Registrar suspeito (imagem associada)
# ============================================================
@faces_bp.route("/faces/register", methods=["POST"])
def register_face():
    """
    Registra uma face de suspeito no sistema.

    A imagem pode vir de duas fontes:
        - Upload local (campo `image`)
        - Caminho no S3 (campo `s3_path` no formato s3://bucket/key)

    O registro funciona de duas formas:
        - Upload local → processamento imediato (gera embedding e salva no Milvus)
        - S3 → processamento assíncrono via Redis + RQ

    Request Body (JSON ou form-data):
        suspect_id (int | str): ID do suspeito (obrigatório).
        metadata (str, optional): JSON contendo informações extras.
        image (file, optional): Imagem enviada por upload.
        s3_path (str, optional): Caminho S3 no formato s3://bucket/key.

    Returns:
        tuple:
            - dict: Informações sobre o processamento ou job criado.
            - int: Código de status HTTP.

    Raises:
        Exception: Em caso de erro interno durante processamento ou envio do job.
    """
    try:
        data = request.get_json(silent=True) or request.form
        suspect_id = data.get("suspect_id")
        metadata_raw = data.get("metadata")
        s3_path = data.get("s3_path")

        if not suspect_id:
            return jsonify({"error": "Campo 'suspect_id' é obrigatório."}), 400

        #  Metadados opcionais
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            metadata = {"raw": metadata_raw}

        # =====================================================
        # Caso: Upload local
        # =====================================================
        if "image" in request.files:
            image_file = request.files["image"]

            embedding_result, status = generate_embeddings(image_file)
            if status != 200:
                return jsonify(embedding_result), status

            embedding = embedding_result["embedding"]

            face_id = insert_face(
                suspect_id=suspect_id,
                embedding=embedding,
                is_query=False,
                metadata=metadata,
                s3_path=None
            )

            return jsonify({
                "message": "Face registrada com sucesso (upload local).",
                "face_id": face_id,
                "suspect_id": suspect_id,
                "source": "upload"
            }), 201

        # =====================================================
        # Caso: Imagem no S3 (via boto3 + Redis)
        # =====================================================
        elif s3_path:
            if not s3_path.startswith("s3://"):
                return jsonify({"error": "Formato inválido em 's3_path'. Use s3://bucket/key"}), 400

            #  Envia tarefa com função real
            job = queue.enqueue(
                process_register_face,
                suspect_id,
                s3_path,
                metadata,
                result_ttl=3600,
                failure_ttl=3600
            )

            return jsonify({
                "message": "Processamento enviado para fila Redis.",
                "job_id": job.get_id(),
                "status": job.get_status(),
                "suspect_id": suspect_id,
                "s3_path": s3_path,
                "source": "s3"
            }), 202

        else:
            return jsonify({
                "error": "Envie uma imagem (campo 'image') ou um 's3_path'."
            }), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500



# ============================================================
#  Rota 2 - Realiza a busca de rostos na imagem enviada
# ============================================================
@faces_bp.route("/faces/search", methods=["POST"])
def search_faces():
    """
    Realiza a busca de rostos na imagem enviada, podendo operar de duas formas:
    (1) Processamento imediato (upload local)
    (2) Processamento assíncrono via S3 + Redis/RQ

    Fluxo geral:
        - O cliente envia uma imagem ou um caminho S3.
        - Se a imagem for enviada diretamente, a busca é processada na hora.
        - Se o caminho S3 for enviado, o processamento é delegado ao worker assíncrono.

    -------------------------------
    CASO 1 — Upload local (processamento imediato)
    -------------------------------
    Quando o cliente envia um arquivo em `multipart/form-data`:
        - A imagem é processada em tempo real pelo método `detect_and_search_faces()`.
        - O sistema detecta múltiplos rostos, computa embeddings e faz a busca no Milvus.
        - Identifica o rosto com maior match (`winner_match`).
        - NÃO realiza upload para o S3 neste modo.
        - Retorna diretamente:
            - bounding boxes
            - rosto vencedor
            - dados do match
            - caminho local da imagem processada
            - metadata do suspeito (opcional, se existir)
            - original_s3 / processed_s3 = None
            - original_url / processed_url = None

    -------------------------------
    CASO 2 — Busca via S3 (processamento assíncrono)
    -------------------------------
    Quando o cliente envia:
        {
            "s3_path": "s3://bucket/key",
            "top_k": 5
        }

        - A requisição é enviada para a fila Redis através de um job RQ.
        - O worker executará `process_search_face_worker`, que:
            - Baixa a imagem do S3
            - Detecta rostos
            - Busca no Milvus
            - Gera imagem processada com boxes coloridas
            - Faz upload da imagem de saída para o S3
            - Retorna as URLs e dados da análise

        A resposta imediata do endpoint será:
            {
                "message": "Busca enviada para a fila.",
                "job_id": "...",
                "status": "...",
                "s3_path": "...",
                "source": "s3"
            }

        O resultado final deve ser consultado via `/jobs/status/<job_id>`.

    Args:
        Nenhum parâmetro direto via função.
        Espera-se no body:
            - image (FileStorage): imagem enviada diretamente (opcional)
            - s3_path (str): caminho S3 (opcional, formato s3://bucket/key)
            - top_k (int): quantidade de matches retornados pelo Milvus (default: 5)

    Returns:
        Response (Flask JSON):
            - 200: processamento local concluído
            - 202: job enviado para fila (modo S3)
            - 400: requisição inválida (sem image e sem s3_path)
            - 500: erro interno

    Observações:
        - Apenas UM dos parâmetros deve ser enviado: "image" OU "s3_path".
        - O caminho S3 deve obrigatoriamente seguir o padrão `s3://bucket/key`.
        - O retorno do processamento via worker incluirá URLs públicas da imagem processada.
        - A integração com o Milvus ocorre dentro de `detect_and_search_faces()` e
          `process_search_face_worker()`.
    """
    try:
        from app.services.embeddings_service import detect_and_search_faces
        from app.workers import process_search_face_worker

        data = request.get_json(silent=True) or request.form
        s3_path = data.get("s3_path")
        top_k = int(data.get("top_k", 5))

        # =====================================================
        # 🟢 CASO 1: Upload local — processamento IMEDIATO
        # =====================================================
        if "image" in request.files:
            image_file = request.files["image"]

            result, status = detect_and_search_faces(image_file, top_k=top_k)
            if status != 200:
                return jsonify(result), status

            winner = result.get("winner_match")
            suspect_metadata = None
            if winner and winner.get("suspect_id") is not None:
                suspect_metadata = get_suspect_metadata(winner["suspect_id"])
                print(">>> SUSPECT_METADATA ===>", suspect_metadata)

            return jsonify({
                **result,
                "original_s3": None,
                "processed_s3": None,
                "original_url": None,
                "processed_url": None,
                "suspect_metadata": suspect_metadata
            }), 200

        # =====================================================
        # 🟡 CASO 2: Busca via S3 — enviar para Redis
        # =====================================================
        if s3_path:
            if not s3_path.startswith("s3://"):
                return jsonify({"error": "Formato inválido para 's3_path'. Use s3://bucket/key"}), 400

            job = queue.enqueue(
                process_search_face_worker,
                s3_path,
                top_k,
                result_ttl=3600,
                failure_ttl=3600
            )

            return jsonify({
                "message": "Busca enviada para a fila.",
                "job_id": job.get_id(),
                "status": job.get_status(),
                "s3_path": s3_path,
                "source": "s3"
            }), 202

        return jsonify({"error": "Forneça 'image' ou 's3_path'."}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500



# ============================================================
#  Rota 3 - Listar todos os suspeitos e suas faces
# ============================================================
@faces_bp.route("/faces/suspects", methods=["GET"])
def list_all_suspects():
    """
    Lista todos os suspeitos cadastrados no Milvus com as faces associadas.

    Cada registro inclui:
        - face_id
        - suspect_id
        - timestamp
        - metadata (convertido de JSON)
        - s3_path
        - source (s3 ou upload)

    Returns:
        tuple:
            - dict: Estrutura contendo suspeitos e suas faces.
            - int: Código de status HTTP.

    Raises:
        Exception: Caso haja erro ao acessar a collection do Milvus.
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
#  Rota 4 - Listar todas as faces de um suspeito específico
# ============================================================
@faces_bp.route("/faces/suspects/<int:suspect_id>", methods=["GET"])
def list_faces_by_suspect(suspect_id):
    """
    Lista todas as faces associadas a um suspeito específico.

    Args:
        suspect_id (int): ID do suspeito consultado.

    Returns:
        tuple:
            - dict: Detalhes das faces encontradas, incluindo:
                - face_id
                - timestamp
                - metadata
                - s3_path
                - source
            - int: Código de status HTTP.

    Raises:
        Exception: Em caso de falhas de acesso ao banco vetorial (Milvus).
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
#  Rota 5 - Listar todos os embeddings cadastrados
# ============================================================
@faces_bp.route("/faces/all", methods=["GET"])
def list_all_faces():
    """
    Lista todas as faces cadastradas no Milvus, incluindo suspeitos e buscas.

    Query Params:
        limit (int, optional): Máximo de registros retornados. Default = 1000.

    Cada item retorna:
        - face_id
        - suspect_id
        - is_query
        - timestamp
        - metadata
        - s3_path
        - source (s3 | upload)

    Returns:
        tuple:
            - dict: Estrutura contendo todas as faces.
            - int: Código de status HTTP.

    Raises:
        Exception: Em caso de falha ao acessar Milvus.
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

        #  Processa metadados e define o campo "source"
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
#  Rota 6 - Remove um embedding específico
# ============================================================
@faces_bp.route("/faces/delete/<int:face_id>", methods=["DELETE"])
def delete_face(face_id):
    """
    Remove uma face específica da collection do Milvus.

    Args:
        face_id (int): ID da face a ser removida.

    Returns:
        tuple:
            - dict: Informações sobre a operação.
            - int: Código HTTP.

    Raises:
        Exception: Em caso de erro no acesso ao Milvus ou remoção.
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
#  Rota 7 - Limpa toda a collection do Milvus
# ============================================================
@faces_bp.route("/faces/clear", methods=["DELETE"])
def clear_collection():
    """
    Remove completamente a collection 'faces' do Milvus.

    Uso exclusivo para fins administrativos.

    Returns:
        tuple:
            - dict: Mensagem de confirmação.
            - int: Código HTTP.

    Raises:
        Exception: Caso ocorra falha durante a operação no Milvus.
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
#  Rota 8 - Editar uma face específica
# ============================================================
@faces_bp.route("/faces/update/<int:face_id>", methods=["PUT"])
def update_face(face_id):
    """
    Atualiza parcialmente um registro facial existente no Milvus.

    Pode alterar:
        - suspect_id
        - metadata (merge entre antigo e novo)
        - mantém: embedding, timestamp, s3_path, is_query

    Args:
        face_id (int): ID da face a ser atualizada.

    Body:
        suspect_id (int, optional): Novo ID do suspeito.
        metadata (str, optional): JSON de metadados a serem mesclados.

    Returns:
        tuple:
            - dict: Dados atualizados.
            - int: Código HTTP.

    Raises:
        Exception: Caso não encontre a face ou falhe na reinserção.
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

        #  Busca os dados existentes (sem embedding)
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

        #  Busca o embedding separadamente
        emb_result = collection.query(
            expr=f"face_id == {face_id}",
            output_fields=["embedding"]
        )
        if not emb_result or "embedding" not in emb_result[0]:
            return jsonify({"error": "Não foi possível recuperar o embedding da face."}), 500

        embedding = emb_result[0]["embedding"]

        #  Determina novos valores
        updated_suspect_id = int(new_suspect_id) if new_suspect_id is not None else current["suspect_id"]

        #  Merge de metadados
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
        #  Deleta o registro antigo
        # ============================================================
        collection.delete(expr=f"face_id == {face_id}")
        collection.flush()

        # ============================================================
        #  Reinsere com os novos dados
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
#  Rota 9 - Remove todas as faces associadas a um suspect_id
# ============================================================
@faces_bp.route("/faces/delete/suspect/<int:suspect_id>", methods=["DELETE"])
def delete_faces_by_suspect(suspect_id):
    """
    Remove todas as faces associadas a um suspeito específico.

    Args:
        suspect_id (int): ID do suspeito alvo.

    Returns:
        tuple:
            - dict: IDs removidos e quantidade total.
            - int: Código HTTP.

    Raises:
        Exception: Em falhas durante a operação no Milvus.
    """ 
    try:
        from app.services.milvus_service import connect_milvus, COLLECTION_NAME
        from pymilvus import Collection, utility

        connect_milvus()

        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"error": f"Collection '{COLLECTION_NAME}' não existe."}), 404

        collection = Collection(COLLECTION_NAME)
        collection.load()

        #  Verifica se há registros do suspeito
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


# ============================================================
#  Rota 10 - Pegar status do job
# ============================================================
@faces_bp.route("/jobs/status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """
    Recupera o status de um job enfileirado no Redis/RQ e retorna o resultado
    produzido pelo worker responsável pela tarefa (registro de face ou busca facial).

    Este endpoint permite que o cliente consulte periodicamente o status de um
    processamento assíncrono iniciado anteriormente, retornando informações como:
    - Status atual (queued, started, finished, failed)
    - Horário de início e término
    - Resultado retornado pelo worker (caso já concluído)
    - Informações sobre erros, se houver

    Fluxo do processamento:
        1. Conecta ao Redis usando a URL configurada na aplicação.
        2. Tenta localizar o job em uma das filas usadas pelo sistema
           ("faces_register_queue" ou "faces_search_queue").
        3. Caso o job não exista, retorna erro 404.
        4. Caso o job exista, monta um payload contendo:
            - ID do job
            - status
            - timestamps (início/término)
            - mensagem de erro, se o job falhou
        5. Se o job estiver em execução ou aguardando, retorna os metadados sem resultado.
        6. Quando o job já finalizou:
            - Se o worker retornou uma string → erro interno, retorna 500.
            - Se retornou algo que não seja dict → erro de formato.
            - Caso seja válido, retorna o conteúdo completo em `response["result"]`.

    Args:
        job_id (str):
            ID do job gerado pelo RQ no momento em que a tarefa foi enfileirada.

    Returns:
        Response (Flask JSON):
            Estrutura semelhante ao exemplo abaixo:

            {
                "job_id": "abc123",
                "status": "finished",
                "started_at": "...",
                "ended_at": "...",
                "error": null,
                "result": {
                    ... conteúdo retornado pelo worker ...
                }
            }

            - Código HTTP 200 para sucesso.
            - Código 404 caso o job não exista.
            - Código 500 para erros internos ou retorno inválido do worker.

    Observações:
        - O campo "result" só é incluído quando o job finaliza com sucesso.
        - Workers são responsáveis por retornar um dict válido; caso retornem outro tipo,
          o endpoint rejeita o formato e retorna erro 500.
        - O endpoint NÃO manipula ou transforma o conteúdo retornado pelo worker, apenas
          repassa diretamente para o cliente.
        - Útil para integrações com front-end ou outros serviços que precisam acompanhar
          tarefas assíncronas como registro e busca facial.
    """
    try:
        from redis import Redis
        from rq.job import Job
        import os

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_conn = Redis.from_url(redis_url)

        # Buscar o job nas filas
        job = None
        for queue_name in ["faces_register_queue", "faces_search_queue"]:
            try:
                job = Job.fetch(job_id, connection=redis_conn)
                break
            except:
                pass

        if job is None:
            return jsonify({"error": "Job não encontrado"}), 404

        response = {
            "job_id": job.id,
            "status": job.get_status(),
            "started_at": str(job.started_at) if job.started_at else None,
            "ended_at": str(job.ended_at) if job.ended_at else None,
            "error": str(job.exc_info) if job.is_failed and job.exc_info else None
        }

        # Job ainda executando
        if job.result is None:
            return jsonify(response), 200

        raw_result = job.result

        # Caso o worker retorne string (erro interno)
        if isinstance(raw_result, str):
            response["error"] = raw_result
            return jsonify(response), 500

        # Se o worker retornou algo inválido
        if not isinstance(raw_result, dict):
            response["error"] = f"Worker retornou tipo inválido: {type(raw_result).__name__}"
            response["raw_result"] = str(raw_result)
            return jsonify(response), 500

        # Caso normal: devolver exatamente o que o worker retornou
        response["result"] = raw_result
        return jsonify(response), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500
