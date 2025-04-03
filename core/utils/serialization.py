import pickle
import json
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

class NumpyEncoder(json.JSONEncoder):
    """Кодировщик для сериализации numpy типов в JSON"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def serialize_model(model: BaseEstimator) -> bytes:
    """Сериализация модели"""
    return pickle.dumps(model)

def deserialize_model(model_bytes: bytes) -> BaseEstimator:
    """Десериализация модели"""
    return pickle.loads(model_bytes)

def serialize_pipeline(pipeline: Pipeline) -> bytes:
    """Сериализация пайплайна"""
    return pickle.dumps(pipeline)

def deserialize_pipeline(pipeline_bytes: bytes) -> Pipeline:
    """Десериализация пайплайна"""
    return pickle.loads(pipeline_bytes)

def serialize_metadata(metadata: Dict[str, Any]) -> str:
    """Сериализация метаданных в JSON"""
    return json.dumps(metadata, cls=NumpyEncoder)

def deserialize_metadata(metadata_str: str) -> Dict[str, Any]:
    """Десериализация метаданных из JSON"""
    return json.loads(metadata_str)

def serialize_container(container: Any) -> bytes:
    """Сериализация контейнера модели"""
    return pickle.dumps(container)

def deserialize_container(container_bytes: bytes) -> Any:
    """Десериализация контейнера модели"""
    return pickle.loads(container_bytes) 