# Executando a Aplicação

Este projeto utiliza **Flask (Python)** com **Redis** e **Milvus** para processamento e armazenamento vetorial de embeddings faciais.  
Siga os passos abaixo para iniciar tudo do zero.  

---

## 📦 **1. Redis — Fila de Mensagens**

Baixe a imagem e rode o container Redis:

```bash
docker pull redis:7-alpine
docker run -d --name redis-local -p 6379:6379 redis:7-alpine
```

✅ Redis estará rodando em `localhost:6379`

---

## 🧩 **2. Milvus — Banco Vetorial**

Em uma terminal na pasta onde está o arquivo `docker-compose.yml` , execute:

```bash
docker compose up -d
```

✅ Milvus estará disponível em `localhost:19530`

---

## 🐍 **3. Python — Aplicação Flask e Worker**

Com dois terminal na raiz do projeto(uma para cada comando), inicie:

### 🔹 requirements.txt
```bash
python -m pip install -r requirements.txt
```

### 🔹 API Flask
```bash
python main.py
```

### 🔹 Worker (fila Redis)
Em outro terminal, rode:
```bash
python run_worker.py
```

✅ A API Flask ficará escutando as requisições.  
✅ O Worker processará as tarefas enfileiradas (registro e busca de faces).

---

## 🧠 **Resumo dos Serviços**

| Serviço | Função | Porta |
|----------|--------|-------|
| 🧠 Flask API | Recebe e enfileira requisições | 5000 |
| ⚙️ Worker | Processa tarefas (Redis) | — |
| 📦 Redis | Fila de mensagens | 6379 |
| 🧩 Milvus | Banco vetorial | 19530 |

---
