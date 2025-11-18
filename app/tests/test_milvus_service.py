import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from app.services.milvus_service import (
    insert_face,
    search_similar_faces,
    create_collection_if_not_exists,
    connect_milvus
)


@pytest.fixture
def mock_embedding():
    """Cria um embedding mock"""
    return np.random.rand(512).tolist()


@pytest.fixture
def mock_collection():
    """Cria um mock da collection do Milvus"""
    collection = Mock()
    collection.num_entities = 5
    collection.indexes = [Mock()]
    return collection


def test_connect_milvus_success():
    """Testa conexão bem-sucedida ao Milvus"""
    with patch('app.services.milvus_service.connections.connect') as mock_connect:
        connect_milvus(host="127.0.0.1", port="19530")
        
        mock_connect.assert_called_once_with("default", host="127.0.0.1", port="19530")


def test_connect_milvus_fallback_lite():
    """Testa fallback para Milvus Lite quando servidor não está disponível"""
    with patch('app.services.milvus_service.connections.connect') as mock_connect:
        mock_connect.side_effect = [Exception("Connection failed"), None]
        
        connect_milvus()
        
        assert mock_connect.call_count == 2


def test_create_collection_if_not_exists():
    """Testa criação da collection"""
    with patch('app.services.milvus_service.utility.has_collection') as mock_has:
        with patch('app.services.milvus_service.Collection') as mock_collection_class:
            mock_has.return_value = False
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            
            result = create_collection_if_not_exists(dim=512)
            
            assert result is not None
            mock_collection.create_index.assert_called_once()


def test_create_collection_already_exists():
    """Testa quando collection já existe"""
    with patch('app.services.milvus_service.utility.has_collection') as mock_has:
        with patch('app.services.milvus_service.Collection') as mock_collection_class:
            mock_has.return_value = True
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            
            result = create_collection_if_not_exists(dim=512)
            
            assert result is not None


def test_insert_face_success(mock_embedding):
    """Testa inserção de face com sucesso"""
    with patch('app.services.milvus_service.connect_milvus'):
        with patch('app.services.milvus_service.create_collection_if_not_exists') as mock_create:
            mock_collection = Mock()
            mock_collection.num_entities = 5
            mock_collection.indexes = [Mock()]
            mock_create.return_value = mock_collection
            
            result = insert_face(
                suspect_id=123,
                embedding=mock_embedding,
                is_query=False,
                metadata={"name": "John"},
                s3_path="s3://bucket/face.jpg"
            )
            
            assert result is not None
            assert result == 6  # num_entities + 1
            mock_collection.insert.assert_called_once()
            mock_collection.flush.assert_called_once()


def test_insert_face_with_none_suspect_id(mock_embedding):
    """Testa inserção com suspect_id None"""
    with patch('app.services.milvus_service.connect_milvus'):
        with patch('app.services.milvus_service.create_collection_if_not_exists') as mock_create:
            mock_collection = Mock()
            mock_collection.num_entities = 5
            mock_collection.indexes = [Mock()]
            mock_create.return_value = mock_collection
            
            result = insert_face(
                suspect_id=None,
                embedding=mock_embedding,
                s3_path="s3://bucket/face.jpg"
            )
            
            assert result is not None
            mock_collection.insert.assert_called_once()


def test_search_similar_faces_success(mock_embedding):
    """Testa busca de faces semelhantes"""
    with patch('app.services.milvus_service.connect_milvus'):
        with patch('app.services.milvus_service.utility.has_collection') as mock_has:
            with patch('app.services.milvus_service.Collection') as mock_collection_class:
                mock_has.return_value = True
                mock_collection = Mock()
                mock_collection.query.return_value = [
                    {"face_id": 1},
                    {"face_id": 2},
                    {"face_id": 3}
                ]
                
                mock_hit = Mock()
                mock_hit.id = 1
                mock_hit.distance = 0.15
                mock_hit.entity.get.return_value = 123
                mock_collection.search.return_value = [[mock_hit]]
                
                mock_collection_class.return_value = mock_collection
                
                result = search_similar_faces(mock_embedding, top_k=3)
                
                assert isinstance(result, list)
                mock_collection.query.assert_called_once()
                mock_collection.search.assert_called_once()


def test_search_similar_faces_no_registered():
    """Testa busca quando não há faces registradas"""
    mock_embedding = np.random.rand(512).tolist()
    
    with patch('app.services.milvus_service.connect_milvus'):
        with patch('app.services.milvus_service.utility.has_collection') as mock_has:
            with patch('app.services.milvus_service.Collection') as mock_collection_class:
                mock_has.return_value = True
                mock_collection = Mock()
                mock_collection.query.return_value = []
                mock_collection_class.return_value = mock_collection
                
                result = search_similar_faces(mock_embedding)
                
                assert result == []
                mock_collection.query.assert_called_once()


def test_search_similar_faces_collection_not_exists():
    """Testa busca quando collection não existe"""
    mock_embedding = np.random.rand(512).tolist()
    
    with patch('app.services.milvus_service.connect_milvus'):
        with patch('app.services.milvus_service.utility.has_collection') as mock_has:
            mock_has.return_value = False
            
            with pytest.raises(Exception) as exc_info:
                search_similar_faces(mock_embedding)
            
            assert "não existe" in str(exc_info.value)
