import json
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import joblib
import hashlib
import os
from pathlib import Path

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
    """
    Безопасная сериализация модели с использованием joblib
    
    Parameters
    ----------
    model : BaseEstimator
        Модель для сериализации
        
    Returns
    -------
    bytes
        Сериализованные данные модели
    """
    return joblib.dumps(model)

def deserialize_model(model_bytes: bytes) -> BaseEstimator:
    """
    Безопасная десериализация модели с использованием joblib
    
    Parameters
    ----------
    model_bytes : bytes
        Сериализованные данные модели
        
    Returns
    -------
    BaseEstimator
        Десериализованная модель
        
    Raises
    ------
    ValueError
        Если данные не являются валидной сериализованной моделью
    """
    try:
        return joblib.loads(model_bytes)
    except Exception as e:
        raise ValueError(f"Ошибка десериализации модели: {str(e)}")

def serialize_pipeline(pipeline: Pipeline) -> bytes:
    """
    Безопасная сериализация пайплайна с использованием joblib
    
    Parameters
    ----------
    pipeline : Pipeline
        Пайплайн для сериализации
        
    Returns
    -------
    bytes
        Сериализованные данные пайплайна
    """
    return joblib.dumps(pipeline)

def deserialize_pipeline(pipeline_bytes: bytes) -> Pipeline:
    """
    Безопасная десериализация пайплайна с использованием joblib
    
    Parameters
    ----------
    pipeline_bytes : bytes
        Сериализованные данные пайплайна
        
    Returns
    -------
    Pipeline
        Десериализованный пайплайн
        
    Raises
    ------
    ValueError
        Если данные не являются валидным сериализованным пайплайном
    """
    try:
        return joblib.loads(pipeline_bytes)
    except Exception as e:
        raise ValueError(f"Ошибка десериализации пайплайна: {str(e)}")

def serialize_metadata(metadata: Dict[str, Any]) -> str:
    """
    Сериализация метаданных в JSON
    
    Parameters
    ----------
    metadata : Dict[str, Any]
        Метаданные для сериализации
        
    Returns
    -------
    str
        JSON строка с метаданными
    """
    return json.dumps(metadata, cls=NumpyEncoder)

def deserialize_metadata(metadata_str: str) -> Dict[str, Any]:
    """
    Десериализация метаданных из JSON
    
    Parameters
    ----------
    metadata_str : str
        JSON строка с метаданными
        
    Returns
    -------
    Dict[str, Any]
        Десериализованные метаданные
        
    Raises
    ------
    ValueError
        Если строка не является валидным JSON
    """
    try:
        return json.loads(metadata_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка десериализации метаданных: {str(e)}")

def serialize_container(container: Any) -> bytes:
    """
    Безопасная сериализация контейнера модели с использованием joblib
    
    Parameters
    ----------
    container : Any
        Контейнер для сериализации
        
    Returns
    -------
    bytes
        Сериализованные данные контейнера
    """
    return joblib.dumps(container)

def deserialize_container(container_bytes: bytes) -> Any:
    """
    Безопасная десериализация контейнера модели с использованием joblib
    
    Parameters
    ----------
    container_bytes : bytes
        Сериализованные данные контейнера
        
    Returns
    -------
    Any
        Десериализованный контейнер
        
    Raises
    ------
    ValueError
        Если данные не являются валидным сериализованным контейнером
    """
    try:
        return joblib.loads(container_bytes)
    except Exception as e:
        raise ValueError(f"Ошибка десериализации контейнера: {str(e)}") 