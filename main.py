from flask import Flask
from app.controllers.embeddings_controller import embeddings_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(embeddings_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
