# ============================================================
# Arquivo de exemplo — NÃO coloque credenciais reais aqui!
# Cada desenvolvedor deve criar seu próprio config.py baseado
# neste arquivo.
# ============================================================
import os

AWS_ACCESS_KEY_ID = "SUA_ACCESS_KEY_AQUI"
AWS_SECRET_ACCESS_KEY = "SUA_SECRET_KEY_AQUI"
AWS_REGION = "us-east-2"  # ou a região que você usa

JAVA_WEBHOOK_URL = os.getenv(
    "JAVA_WEBHOOK_URL", 
    "http://host.docker.internal:8080/api/nexus/webhooks/suspect-processed"
)