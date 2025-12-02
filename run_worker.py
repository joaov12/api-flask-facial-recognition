import os
from rq import SimpleWorker, Queue
from redis import Redis

# Lê o REDIS_URL da variável de ambiente ou usa localhost como fallback
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(redis_url)

# Filas que o worker vai ouvir
listen = ['faces_register_queue', 'faces_search_queue']

def run_worker():
    print(f"[Worker] Iniciado. Conectado em: {redis_url}")
    print(f"[Worker] Ouvindo filas: {listen}")
    
    queues = [Queue(name, connection=redis_conn) for name in listen]
    worker = SimpleWorker(queues, connection=redis_conn)
    
    # burst=False => fica ouvindo continuamente
    worker.work(burst=False)

if __name__ == '__main__':
    run_worker()
