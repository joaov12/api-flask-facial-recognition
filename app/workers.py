from app.services.milvus_service import insert_face, search_similar_faces
from app.services.embeddings_service import generate_embeddings
import boto3
import config
from io import BytesIO
import traceback

def process_register_face(suspect_id, s3_path, metadata=None):
    """
    Processa o registro de uma face: baixa a imagem do S3, gera o embedding
    e salva o registro no Milvus. Ao finalizar, notifica o Java via webhook.

    Args:
        suspect_id (int or str): ID do suspeito associado à face registrada.
        s3_path (str): Caminho completo no formato s3://bucket/key da imagem a ser processada.
        metadata (dict, optional): Metadados adicionais relacionados à face. Default é None.

    Returns:
        dict: Informações do registro criado, contendo:
            - message (str): Mensagem de sucesso.
            - face_id (int): ID da face armazenada no Milvus.
            - suspect_id (int): ID do suspeito.
            - s3_path (str): Caminho da imagem no S3.

    Raises:
        Exception: Caso ocorra erro no download da imagem, geração do embedding
            ou inserção no Milvus.
    """
    face_id = None
    
    try:
        print(f"[Worker] Processando {s3_path} (suspect_id={suspect_id})")

        # Quebra o caminho s3://bucket/key
        if not s3_path.startswith("s3://"):
            raise ValueError("Caminho S3 inválido. Use o formato s3://bucket/key")

        parts = s3_path.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError("Caminho S3 inválido. Deve conter bucket e key")

        bucket, key = parts
        print(f"[Worker] Baixando do bucket '{bucket}' com key '{key}'...")

        # Baixa a imagem do S3 para a memória
        s3 = boto3.client(
            "s3",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )

        buffer = BytesIO()
        s3.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        buffer.name = key.split("/")[-1]  # nome do arquivo, útil se o modelo usa extensão
        print(f"[Worker] Download concluído ({len(buffer.getvalue())} bytes).")

        # Gera o embedding com a imagem em memória
        embedding_result, status = generate_embeddings(buffer)
        if status != 200:
            raise Exception(f"Falha ao gerar embedding: {embedding_result}")

        embedding = embedding_result["embedding"]

        # Insere no Milvus
        face_id = insert_face(
            suspect_id=int(suspect_id),
            embedding=embedding,
            is_query=False,
            metadata=metadata,
            s3_path=s3_path
        )

        print(f"[Worker] ✅ Face {face_id} inserida com sucesso (suspect_id={suspect_id})")

        # 🆕 Notifica o Java que o processamento foi concluído com sucesso
        notify_java_completion(
            suspect_id=suspect_id,
            face_id=face_id,
            s3_path=s3_path,
            status="completed"
        )

        return {
            "message": "Face registrada com sucesso.",
            "face_id": int(face_id),
            "suspect_id": int(suspect_id),
            "s3_path": s3_path
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Worker] ❌ Erro ao processar {s3_path}: {e}")
        
        # 🆕 Notifica o Java que o processamento falhou
        notify_java_completion(
            suspect_id=suspect_id,
            face_id=face_id,
            s3_path=s3_path,
            status="failed",
            error=str(e)
        )
        
        raise e


def process_search_face(s3_path=None, top_k=5):
    """
    Processa a busca de uma face: baixa a imagem do S3, gera o embedding,
    encontra faces semelhantes no Milvus e registra a consulta como is_query=True.

    Args:
        s3_path (str, optional): Caminho da imagem no formato s3://bucket/key.
            Necessário para realizar a busca. Default é None.
        top_k (int, optional): Número máximo de correspondências retornadas.
            Default é 5.

    Returns:
        dict: Resultado da busca contendo:
            - query_face_id (int): ID da face da consulta inserida no Milvus.
            - matches (list): Lista de correspondências encontradas.
            - s3_path (str): Caminho da imagem usada na busca.
            - source (str): Indica que a busca veio do S3.

    Raises:
        Exception: Se ocorrer erro no download da imagem, geração do embedding
            ou durante a busca no Milvus.
    """
    import boto3
    from io import BytesIO

    try:
        print(f"[Worker]  Processando busca (S3 path={s3_path})")

        # Valida caminho S3
        if not s3_path or not s3_path.startswith("s3://"):
            raise ValueError("Caminho S3 inválido. Use o formato s3://bucket/key")

        parts = s3_path.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError("Caminho S3 inválido. Deve conter bucket e key")

        bucket, key = parts
        print(f"[Worker]  Baixando imagem do bucket '{bucket}' com key '{key}'...")

        # Baixa a imagem
        s3 = boto3.client(
            "s3",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )

        buffer = BytesIO()
        s3.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        buffer.name = key.split("/")[-1]
        print(f"[Worker]  Download concluído ({len(buffer.getvalue())} bytes).")

        # Gera embedding
        embedding_result, status = generate_embeddings(buffer)
        if status != 200:
            raise Exception(f"Falha ao gerar embedding: {embedding_result}")

        embedding = embedding_result["embedding"]

        # Busca semelhantes no Milvus
        matches = search_similar_faces(embedding, top_k=int(top_k))

        # Salva face da busca (is_query=True)
        query_face_id = insert_face(
            suspect_id=None,
            embedding=embedding,
            is_query=True,
            metadata={"s3_path": s3_path},
            s3_path=s3_path
        )

        print(f"[Worker]  Busca concluída. {len(matches)} correspondências encontradas.")

        # Retorna resultado simples
        return {
            "query_face_id": int(query_face_id),
            "matches": matches,
            "s3_path": s3_path,
            "source": "s3"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Worker]  Erro ao processar busca: {e}")
        raise e
    

def notify_java_completion(suspect_id, face_id, s3_path, status, error=None):
    """
    Notifica o backend Java que o processamento da face foi concluído
    (com sucesso ou falha) via webhook HTTP.

    Args:
        suspect_id (int or str): ID do suspeito.
        face_id (int or None): ID da face inserida no Milvus (None se falhou).
        s3_path (str): Caminho da imagem no S3.
        status (str): Status do processamento ('completed' ou 'failed').
        error (str, optional): Mensagem de erro caso status seja 'failed'.

    Returns:
        None
    """
    import requests
    from rq import get_current_job
    
    # Obtém o job_id do RQ (se disponível)
    job = get_current_job()
    job_id = job.get_id() if job else None
    
    # URL do webhook configurada no arquivo de config
    webhook_url = config.JAVA_WEBHOOK_URL
    
    # Monta o payload do webhook
    payload = {
        "suspectId": int(suspect_id),
        "jobId": job_id,
        "status": status,
        "message": "Processamento concluído com sucesso" if status == "completed" else f"Erro: {error}",
        "faceId": int(face_id) if face_id else None,
        "s3Path": s3_path
    }
    
    try:
        print(f"[Worker] 🔔 Enviando webhook para Java: {webhook_url}")
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10,  # timeout de 10 segundos
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print(f"[Worker] ✅ Webhook enviado com sucesso: {response.status_code}")
        else:
            print(f"[Worker] ⚠️ Webhook retornou status inesperado: {response.status_code}")
            print(f"[Worker] Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"[Worker] ⚠️ Timeout ao enviar webhook para Java")
    except requests.exceptions.ConnectionError:
        print(f"[Worker] ⚠️ Erro de conexão ao enviar webhook para Java")
    except Exception as e:
        print(f"[Worker] ⚠️ Erro inesperado ao enviar webhook: {e}")
        # Não propaga a exceção para não interromper o job principal