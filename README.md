# API Flask - Facial Recognition (NEXUS)

API em Flask responsável por expor serviços de reconhecimento facial 
usando embeddings (Facenet).  

## Estrutura
- **app/**: código principal
  - controllers/: rotas
  - services/: lógica de negócio/modelo
  - models/: arquivos de IA
  - core/: configurações
  - tests/: testes

# 🚀 Como rodar o projeto NEXUS (API Flask Facial Recognition)

## 📦 Pré-requisitos
- **Python 3.10** (instale pela [Microsoft Store] no Windows).  
- **VSCode** (recomendado).  

---

## ⚙️ Passo a passo para rodar

1. **Instale as dependências**  
   No terminal integrado do VSCode:
   ```bash
   pip install -r requirements.txt
   ```

2. **Inicie a API**
   ```bash
   python main.py
   ```
   A aplicação vai rodar em:  
   👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Teste o endpoint de embeddings**
   ```bash
   python app/tests/test_embeddings.py
   ```

4. **Teste o endpoint de comparação**
   ```bash
   python app/tests/test_compare.py
   ```

---

## 📌 Observações
- Coloque as imagens de teste (`exemplo1.jpg`, `exemplo2.jpg`, `exemplo3.jpg`) dentro da pasta `app/tests/`.  
- O endpoint `/embeddings` gera os embeddings de uma imagem.  
- O endpoint `/compare` compara duas imagens e retorna a distância e se são a mesma pessoa.  