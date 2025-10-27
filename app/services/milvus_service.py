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
# 🔹 Conexão com o servidor Milvus
# ============================================================
def connect_milvus(host="127.0.0.1", port="19530"):
    """
    Tenta conectar ao Milvus rodando no Docker (localhost:19530).
    Se falhar, ativa o modo embutido (Milvus Lite, para fallback local).
    """
    try:
        connections.connect("default", host=host, port=port)
        print("[Milvus] ✅ Conectado ao servidor local.")
    except Exception as e:
        print(f"[Milvus] ⚠️ Falha ao conectar ao servidor local ({e}).")
        print("[Milvus] ▶ Iniciando em modo embutido (Milvus Lite)...")

        # Caminho absoluto do arquivo local (sem 'file://')
        db_path = os.path.abspath("milvus_lite.db")

        # Conecta ao modo Lite usando o caminho direto (sem prefixo)
        connections.connect("default", uri=db_path)
        print(f"[Milvus] 💾 Rodando em modo Lite (banco: {db_path})")


# ============================================================
# 🔹 Criação da Collection
# ============================================================
def create_collection_if_not_exists(dim=512):
    """
    Cria a collection 'faces' se ainda não existir e cria o índice vetorial.
    """
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="face_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="suspect_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
        FieldSchema(name="is_query", dtype=DataType.BOOL),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024)
    ]

    schema = CollectionSchema(fields, description="Banco de embeddings faciais")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("[Milvus] 🆕 Collection 'faces' criada com sucesso (face_id INT64).")

    # 🔹 Cria índice vetorial (obrigatório para buscas)
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    collection.create_index(field_name="embedding", index_params=index_params)
    print("[Milvus] 🧩 Índice vetorial criado (IVF_FLAT, L2).")

    return collection


# ============================================================
# 🔹 Inserção de uma face
# ============================================================
def insert_face(suspect_id, embedding, is_query=False, metadata=None):
    """
    Insere uma face no Milvus com ID numérico incremental.
    """
    connect_milvus()
    collection = create_collection_if_not_exists(dim=len(embedding))

    # Garante que o índice exista
    if not collection.indexes:
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("[Milvus] 🧩 Índice criado automaticamente na collection existente.")

    # 🔹 Gera o próximo ID manualmente
    total_count = collection.num_entities
    face_id = int(total_count) + 1  # incremental
    timestamp = int(time.time())

    data = [
        [face_id],
        [int(suspect_id) if suspect_id else 0],
        [embedding],
        [timestamp],
        [is_query],
        [str(metadata or {})]
    ]

    collection.insert(data)
    collection.flush()
    collection.load()
    print(f"[Milvus] ✅ Face inserida (face_id={face_id}, is_query={is_query})")

    return face_id


# ============================================================
# 🔹 Busca de faces semelhantes
# ============================================================
def search_similar_faces(embedding, top_k=3):
    """
    Busca as faces mais semelhantes no banco vetorial.
    Garante que a collection esteja realmente carregada no servidor.
    """
    connect_milvus()

    if not utility.has_collection(COLLECTION_NAME):
        raise Exception(f"Collection '{COLLECTION_NAME}' não existe.")

    collection = Collection(COLLECTION_NAME)

    # 🔹 Garante que está carregada de fato
    try:
        load_state = utility.get_load_state(COLLECTION_NAME)
        if load_state != "Loaded":
            print(f"[Milvus] ⚙️ Collection não estava carregada (estado: {load_state}). Tentando carregar...")
            collection.load()
            # Espera um pouquinho para o Milvus indexar
            time.sleep(2)
    except Exception as e:
        print(f"[Milvus] ⚠️ Erro ao checar estado da collection: {e}")
        collection.load()
        time.sleep(2)

    # 🔹 Executa busca
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["suspect_id", "is_query"]
    )

    matches = []
    for hits in results:
        for hit in hits:
            matches.append({
                "face_id": int(hit.id),
                "suspect_id": hit.entity.get("suspect_id"),
                "distance": hit.distance
            })


    print(f"[Milvus] 🔍 {len(matches)} resultados encontrados.")
    return matches


