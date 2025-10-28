from app.services.milvus_service import insert_face, search_similar_faces
from app.services.embeddings_service import generate_embeddings
import boto3
from io import BytesIO
import traceback

def process_register_face(suspect_id, s3_path, metadata=None):
    """
    Worker: baixa imagem do S3, gera embedding e salva no Milvus.
    """
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
            aws_access_key_id="xxx",
            aws_secret_access_key="xxx",
            region_name="us-east-2"
        )

        buffer = BytesIO()
        s3.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        buffer.name = key.split("/")[-1]  # nome do arquivo, útil se o modelo usa extensão
        print(f"[Worker] Download concluído ({len(buffer.getvalue())} bytes).")

        #  Gera o embedding com a imagem em memória
        embedding_result, status = generate_embeddings(buffer)
        if status != 200:
            raise Exception(f"Falha ao gerar embedding: {embedding_result}")

        embedding = embedding_result["embedding"]

        #  Insere no Milvus
        face_id = insert_face(
            suspect_id=int(suspect_id),
            embedding=embedding,
            is_query=False,
            metadata=metadata,
            s3_path=s3_path
        )

        print(f"[Worker] ✅ Face {face_id} inserida com sucesso (suspect_id={suspect_id})")

        return {
            "message": "Face registrada com sucesso.",
            "face_id": int(face_id),
            "suspect_id": int(suspect_id),
            "s3_path": s3_path
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Worker]  Erro ao processar {s3_path}: {e}")
        raise e


def process_search_face(s3_path=None, top_k=5):
    """
    Worker: busca suspeitos semelhantes (via S3 ou upload).
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
            aws_access_key_id="xxx",
            aws_secret_access_key="xxx",
            region_name="xxx"
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