"""
Функции для расчета метрик моделей регрессии.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score


def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Рассчитывает базовые метрики качества модели регрессии.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred: Предсказанные значения
    
    Returns:
        Dict: Словарь с метриками
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }
    
    return metrics


def calculate_cross_val_metrics(model, X: pd.DataFrame, y: pd.Series, 
                               cv: int = 5, metric: str = 'r2') -> Dict[str, float]:
    """
    Рассчитывает метрики с использованием кросс-валидации.
    
    Args:
        model: Обученная модель с методами fit и predict
        X: Признаки
        y: Целевая переменная
        cv: Количество фолдов для кросс-валидации
        metric: Метрика для расчета ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error')
        
    Returns:
        Dict: Словарь с метриками кросс-валидации (среднее и стандартное отклонение)
    """
    # Приведение метрик к формату sklearn
    metric_map = {
        'r2': 'r2',
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'median_ae': 'neg_median_absolute_error',
        'explained_variance': 'explained_variance'
    }
    
    scoring = metric_map.get(metric, metric)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Для отрицательных метрик инвертируем знак
    if scoring.startswith('neg_'):
        cv_scores = -cv_scores
    
    return {
        f'cv_{metric}_mean': cv_scores.mean(),
        f'cv_{metric}_std': cv_scores.std()
    }


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Вычисляет остатки (residuals) для оценки качества регрессионной модели.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred: Предсказанные значения
        
    Returns:
        np.ndarray: Массив остатков
    """
    return y_true - y_pred 