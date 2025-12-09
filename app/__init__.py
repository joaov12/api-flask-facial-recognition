from flask import Flask
from app.controllers.faces_controller import faces_bp


def create_app():
    app = Flask(__name__)

    # Registrar todos os blueprints
    app.register_blueprint(faces_bp)

    return app
