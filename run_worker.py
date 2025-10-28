from rq import SimpleWorker, Queue
from redis import Redis

listen = ['faces_register_queue', 'faces_search_queue']
redis_conn = Redis(host='localhost', port=6379, db=0)

def run_worker():
    print(f"[Worker] 🚀 Iniciado (modo compatível com Windows). Ouvindo: {listen}")
    queues = [Queue(name, connection=redis_conn) for name in listen]
    worker = SimpleWorker(queues, connection=redis_conn)
    worker.work(burst=False)

if __name__ == '__main__':
    run_worker()
