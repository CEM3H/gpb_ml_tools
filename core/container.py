from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .utils.serialization import (
    serialize_container,
    deserialize_container
)

@dataclass
class ModelMetadata:
    """Класс для хранения метаданных модели"""
    model_id: str
    model_name: str
    author: str
    target_description: str
    features_description: Dict[str, str]
    train_tables: List[str]
    created_at: str
    
class ModelContainer:
    """Класс для хранения модели и всех связанных артефактов"""
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.preprocessing_pipeline: Optional[Pipeline] = None
        self.model: Optional[BaseEstimator] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self.metrics: Dict[str, float] = {}
        self.calibration_model: Optional[BaseEstimator] = None
        
    def save(self, path: str):
        """
        Сохранение контейнера в файл
        
        Args:
            path: путь к файлу для сохранения
        """
        container_bytes = serialize_container(self)
        with open(path, 'wb') as f:
            f.write(container_bytes)
    
    @classmethod
    def load(cls, path: str) -> 'ModelContainer':
        """
        Загрузка контейнера из файла
        
        Args:
            path: путь к файлу для загрузки
            
        Returns:
            ModelContainer: загруженный контейнер
        """
        with open(path, 'rb') as f:
            container_bytes = f.read()
        return deserialize_container(container_bytes) 