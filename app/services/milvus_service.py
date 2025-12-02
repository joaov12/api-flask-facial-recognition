from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import time
import uuid
import os

# Nome fixo da collection
COLLECTION_NAME = "faces"

# ============================================================
# Conexão com o servidor Milvus
# ============================================================
def connect_milvus():
    """Estabelece conexão com o servidor Milvus.

    Esta função tenta conectar ao servidor Milvus usando as variáveis de ambiente
    `MILVUS_HOST` e `MILVUS_PORT`. Caso a conexão falhe, uma exceção é lançada.

    Exemplos:
        >>> connect_milvus()
        [Milvus] Tentando conectar em milvus:19530 ...
        [Milvus] ✅ Conectado ao servidor Milvus em milvus:19530

    Environment Variables:
        MILVUS_HOST (str): Endereço do servidor Milvus. Default: "127.0.0.1".
        MILVUS_PORT (str): Porta do servidor Milvus. Default: "19530".

    Raises:
        Exception: Se não for possível estabelecer a conexão com o servidor Milvus.
    """
    milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
    milvus_port = os.getenv("MILVUS_PORT", "19530")

    print(f"[Milvus] Tentando conectar em {milvus_host}:{milvus_port} ...")

    connections.connect("default", host=milvus_host, port=milvus_port)

    print(f"[Milvus] ✅ Conectado ao servidor Milvus em {milvus_host}:{milvus_port}")


# ============================================================
#  Criação da Collection
# ============================================================
def create_collection_if_not_exists(dim=512):
    """
    Cria a collection 'faces' no Milvus caso ela ainda não exista,
    incluindo o índice vetorial para o campo de embeddings.

    Args:
        dim (int, optional): Dimensão dos embeddings faciais. Default é 512.

    Returns:
        Collection: Instância da collection existente ou recém-criada.
    """
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="face_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="suspect_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
        FieldSchema(name="is_query", dtype=DataType.BOOL),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="s3_path", dtype=DataType.VARCHAR, max_length=512)  # 🆕 novo campo
    ]

    schema = CollectionSchema(fields, description="Banco de embeddings faciais")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("[Milvus] 🆕 Collection 'faces' criada com sucesso com campo 's3_path'.")

    #  Cria índice vetorial
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("[Milvus] 🧩 Índice vetorial criado (IVF_FLAT, L2).")

    return collection


# ============================================================
#  Inserção de uma face
# ============================================================
def insert_face(suspect_id, embedding, is_query=False, metadata=None, s3_path=None):
    """
    Insere um registro facial na collection 'faces', incluindo o embedding,
    informações do suspeito, metadados e o caminho no S3.

    Args:
        suspect_id (int): ID do suspeito associado ao embedding.
        embedding (list[float]): Vetor de características faciais.
        is_query (bool, optional): Marca a face como consulta ou registro. Default é False.
        metadata (dict or str, optional): Metadados adicionais do registro. Default é None.
        s3_path (str, optional): Caminho do arquivo armazenado no S3. Default é None.

    Returns:
        int: O ID incremental da face inserida.
    """ 
    connect_milvus()
    collection = create_collection_if_not_exists(dim=len(embedding))

    if not collection.indexes:
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    total_count = collection.num_entities
    face_id = int(total_count) + 1
    timestamp = int(time.time())

    data = [
        [face_id],
        [int(suspect_id) if suspect_id else 0],
        [embedding],
        [timestamp],
        [is_query],
        [str(metadata or {})],
        [s3_path or ""]  # 🆕 salva o path do S3
    ]

    collection.insert(data)
    collection.flush()
    collection.load()

    print(f"[Milvus] ✅ Face inserida (face_id={face_id}, s3_path={s3_path})")
    return face_id



# ============================================================
#  Busca de faces semelhantes
# ============================================================
def search_similar_faces(embedding, top_k=3):
    """
    Realiza a busca vetorial no Milvus retornando as faces mais semelhantes
    ao embedding fornecido, considerando apenas registros marcados como
    `is_query=False` (faces cadastradas).

    Args:
        embedding (list[float]): Vetor de características usado como consulta.
        top_k (int, optional): Número máximo de resultados retornados. Default é 3.

    Returns:
        list[dict]: Lista de correspondências contendo:
            - face_id (int): ID da face encontrada.
            - suspect_id (int): ID do suspeito associado.
            - distance (float): Distância L2 entre os embeddings.

        Caso nenhuma face válida exista, retorna uma lista vazia.

    Raises:
        Exception: Caso a collection 'faces' não exista.
    """
    connect_milvus()

    if not utility.has_collection(COLLECTION_NAME):
        raise Exception(f"Collection '{COLLECTION_NAME}' não existe.")

    collection = Collection(COLLECTION_NAME)
    collection.load()

    #  Filtra apenas embeddings de suspeitos cadastrados
    # Retorna face_ids válidos para busca
    registered_faces = collection.query(
        expr="is_query == false",
        output_fields=["face_id"]
    )

    if not registered_faces:
        print("[Milvus] ⚠️ Nenhuma face registrada encontrada.")
        return []

    valid_face_ids = [int(f["face_id"]) for f in registered_faces]

    #  Busca vetorial apenas entre os registros válidos
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["suspect_id", "is_query"],
        expr=f"face_id in {valid_face_ids}"
    )

    matches = []
    for hits in results:
        for hit in hits:
            if not hit.entity.get("is_query"):  # reforço extra
                matches.append({
                    "face_id": int(hit.id),
                    "suspect_id": hit.entity.get("suspect_id"),
                    "distance": hit.distance
                })

    print(f"[Milvus] 🔍 {len(matches)} resultados encontrados (somente cadastrados).")
    return matches


