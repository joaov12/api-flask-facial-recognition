# Projeto de Reconhecimento Facial — Parte IA/Python

## 🚀 Como executar o projeto localmente

### **1️⃣ — Ter o Docker Desktop em execução**
Certifique-se de que o **Docker Desktop** está rodando no seu computador.  
> *https://docs.docker.com/desktop/setup/install/windows-install/*

---

### **2️⃣ — Criar o arquivo `config.py` na raiz do projeto**

Crie seu próprio arquivo `config.py` na raiz do projeto, com as credenciais.

#### 🧩 Exemplo de estrutura do `config.py`:
```python
# Tudo que precisa são as credenciais da AWS.

AWS_ACCESS_KEY_ID = "sua_access_key_aqui"
AWS_SECRET_ACCESS_KEY = "sua_secret_key_aqui"
AWS_REGION = "us-east-1"
```

---

### **3️⃣ — Subir o ambiente Docker**

Abra um terminal na **raiz do projeto** e execute o comando abaixo para construir e iniciar todos os serviços:

```bash
docker-compose up --build
```
O processo pode demorar alguns minutos, na primeira vez.

Após o build, o ambiente completo será iniciado automaticamente, incluindo:
- 🧠 **API Flask** (`facial_api`)  
- ⚙️ **Worker de filas** (`facial_worker`)  
- 🗄️ **Redis**  
- 📦 **Milvus**  
- 🔑 **Etcd**  
- ☁️ **MinIO**


---
## 💡 Outro comandos

- Para encerrar todos os containers:
    ```bash
    docker-compose down
    ```

- Para **reiniciar apenas a API** (sem rebuildar tudo):  
  ```bash
  docker-compose restart api
  ```

- Para **limpar volumes** e dados persistentes (Redis, Milvus, etc.):  
  ```bash
  docker-compose down -v
  ```

---


### 🧩 Descrição dos serviços

- 🧠 **API Flask (`facial_api`)**  
  Serviço principal da aplicação.  
  Responsável por receber requisições HTTP, processar imagens faciais, interagir com o banco vetorial (Milvus) e enfileirar tarefas no Redis.

- ⚙️ **Worker de filas (`facial_worker`)**  
  Executa as tarefas assíncronas enviadas pela API (como geração de embeddings faciais, inserção e busca no Milvus).  
  Utiliza o **Redis** como gerenciador de filas (RQ - Redis Queue).

- 🗄️ **Redis**  
  Banco de dados em memória utilizado para gerenciamento de filas e cache.  
  Armazena os jobs criados pela API e processados pelo Worker.

- 📦 **Milvus**  
  Banco de dados vetorial especializado em buscas de similaridade entre embeddings (vetores).  
  É onde ficam armazenados os embeddings das faces cadastradas e consultadas.

- 🔑 **Etcd**  
  Serviço auxiliar utilizado internamente pelo Milvus para controle de configuração, registro de nós e coordenação de serviços distribuídos.

- ☁️ **MinIO**  
  Armazenamento de objetos compatível com o S3 da AWS.  
  Utilizado para guardar imagens, arquivos e outros dados binários do sistema.

---

# 🧪 Testando os endpoints no Postman

Abaixo estão os três principais endpoints para testar o funcionamento da API facial.

---

### **1️⃣ — Registrar face já associada a um suspeito (S3)**  
Fluxo de registro de um novo suspeito.

**Método:** `POST`  
**Endpoint:** `http://127.0.0.1:5000/faces/register`  
**Body (raw / JSON):**
```json
{
  "s3_path": "s3://apijava-qrcode/João Gabriel.png_1763379356782",
  "suspect_id": 1,
  "metadata": {
    "origem": "S3",
    "operador": "Jose Antonio"
  }
}
```

---

### **2️⃣ — Buscar rostos semelhantes (S3)**  
Fluxo de busca facial a partir de uma imagem no S3.

**Método:** `POST`  
**Endpoint:** `http://127.0.0.1:5000/faces/search`  
**Body (raw / JSON):**
```json
{
  "s3_path": "s3://apijava-qrcode/João Gabriel.png_1763379356782",
  "top_k": 5
}
```

---

### **3️⃣ — Listar suspeitos registrados**  

**Método:** `GET`  
**Endpoint:** `http://127.0.0.1:5000/faces/suspects`

---

💡 **Dica:**  
Todos os endpoints devem ser testados com o ambiente Docker em execução, após rodar:
```bash
docker-compose up --build
```
