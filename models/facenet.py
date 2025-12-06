import threading
from keras_facenet import FaceNet

_model = None
_lock = threading.Lock()

def get_facenet_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                print("[FaceNet] Carregando modelo...")
                _model = FaceNet()
                print("[FaceNet] Modelo carregado.")
    return _model
