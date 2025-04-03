"""
Функции для расчета метрик моделей классификации.
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score


def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Рассчитывает базовые метрики качества модели классификации.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred: Предсказанные значения
        y_pred_proba: Предсказанные вероятности (для бинарной классификации - вероятности класса 1)
    
    Returns:
        Dict: Словарь с метриками
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }
    
    # Добавляем метрики, требующие вероятности
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            # В случае, если невозможно рассчитать ROC AUC (например, при наличии только одного класса)
            metrics['roc_auc'] = np.nan
            metrics['average_precision'] = np.nan
    
    return metrics


def calculate_cross_val_metrics(model, X: pd.DataFrame, y: pd.Series, 
                               cv: int = 5, metric: str = 'roc_auc') -> Dict[str, float]:
    """
    Рассчитывает метрики с использованием кросс-валидации.
    
    Args:
        model: Обученная модель с методами fit и predict/predict_proba
        X: Признаки
        y: Целевая переменная
        cv: Количество фолдов для кросс-валидации
        metric: Метрика для расчета ('roc_auc', 'accuracy', 'f1', 'precision', 'recall')
        
    Returns:
        Dict: Словарь с метриками кросс-валидации (среднее и стандартное отклонение)
    """
    scoring = metric
    if metric == 'roc_auc' and len(np.unique(y)) <= 1:
        # При наличии только одного класса ROC AUC не определен
        return {
            f'cv_{metric}_mean': np.nan,
            f'cv_{metric}_std': np.nan
        }
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        f'cv_{metric}_mean': cv_scores.mean(),
        f'cv_{metric}_std': cv_scores.std()
    }


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Генерирует матрицу ошибок.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred: Предсказанные значения
        
    Returns:
        Tuple: (матрица ошибок, список уникальных классов)
    """
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Генерирует подробный отчет о метриках по классам.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred: Предсказанные значения
        
    Returns:
        Dict: Словарь с метриками по классам
    """
    return classification_report(y_true, y_pred, output_dict=True) 