import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from io import BytesIO
from PIL import Image
from app.services.embeddings_service import generate_embeddings, compare_embeddings


@pytest.fixture
def mock_image_file():
    """Cria um arquivo de imagem mock em memória"""
    img = Image.new('RGB', (160, 160), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_generate_embeddings_success(mock_image_file):
    """Testa geração de embeddings com sucesso"""
    with patch('app.services.embeddings_service.facenet_model.extract') as mock_extract:
        mock_embedding = np.random.rand(128).tolist()
        mock_extract.return_value = [
            {
                "embedding": mock_embedding,
                "box": [10, 20, 100, 150]
            }
        ]
        
        result, status = generate_embeddings(mock_image_file)
        
        assert status == 200
        assert "embedding" in result
        assert "box" in result
        assert len(result["embedding"]) == 128


def test_generate_embeddings_no_face_detected(mock_image_file):
    """Testa quando nenhum rosto é detectado"""
    with patch('app.services.embeddings_service.facenet_model.extract') as mock_extract:
        mock_extract.return_value = []
        
        result, status = generate_embeddings(mock_image_file)
        
        assert status == 400
        assert "error" in result
        assert "Nenhum rosto detectado" in result["error"]


def test_generate_embeddings_invalid_file():
    """Testa com arquivo inválido"""
    invalid_file = BytesIO(b"not an image")
    
    result, status = generate_embeddings(invalid_file)
    
    assert status == 500
    assert "error" in result


def test_compare_embeddings_same_person(mock_image_file):
    """Testa comparação de embeddings da mesma pessoa"""
    with patch('app.services.embeddings_service.facenet_model.extract') as mock_extract:
        same_embedding = np.random.rand(128).tolist()
        mock_extract.return_value = [
            {
                "embedding": same_embedding,
                "box": [10, 20, 100, 150]
            }
        ]
        
        result, status = compare_embeddings(mock_image_file, mock_image_file, threshold=0.7)
        
        assert status == 200
        assert "distance" in result
        assert "same_person" in result
        assert result["same_person"] is True
        assert result["distance"] < 0.1


def test_compare_embeddings_different_person(mock_image_file):
    """Testa comparação de embeddings de pessoas diferentes"""
    with patch('app.services.embeddings_service.facenet_model.extract') as mock_extract:
        emb1 = np.random.rand(128).tolist()
        emb2 = np.random.rand(128).tolist()
        
        mock_extract.side_effect = [
            [{"embedding": emb1, "box": [10, 20, 100, 150]}],
            [{"embedding": emb2, "box": [10, 20, 100, 150]}]
        ]
        
        result, status = compare_embeddings(mock_image_file, mock_image_file, threshold=0.5)
        
        assert status == 200
        assert "distance" in result
        assert "same_person" in result
        assert isinstance(result["same_person"], bool)


def test_compare_embeddings_extraction_fails(mock_image_file):
    """Testa comparação quando extração falha em uma das imagens"""
    with patch('app.services.embeddings_service.generate_embeddings') as mock_gen:
        mock_gen.side_effect = [
            ({"error": "Falha na extração"}, 500),
            ({"embedding": [0.1] * 128, "box": [10, 20, 100, 150]}, 200)
        ]
        
        result, status = compare_embeddings(mock_image_file, mock_image_file)
        
        assert status == 400
        assert "error" in result
