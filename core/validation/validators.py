from typing import Dict
import pandas as pd
from .base import ModelValidator
import numpy as np
from ..metrics.classification import calculate_basic_metrics, calculate_cross_val_metrics

class DefaultModelValidator(ModelValidator):
    """Реализация стандартного валидатора модели"""
    def __init__(self, cv: int = 5):
        super().__init__(validation_params={'cv': cv})
        self.cv = cv
        
    def validate(self, model, X, y):
        """
        Валидация модели
        
        Args:
            model: Обученная модель
            X: Признаки
            y: Целевая переменная
        
        Returns:
            Dict: Метрики качества модели
        """
        # Рассчитываем метрики на всем наборе данных
        y_pred = model.predict(X)
        y_pred_proba = None
        
        # Пробуем получить вероятностные предсказания, если модель поддерживает
        try:
            y_pred_proba = model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError, ValueError):
            pass
            
        # Рассчитываем базовые метрики
        metrics = calculate_basic_metrics(y, y_pred, y_pred_proba)
        
        # Рассчитываем метрики с использованием кросс-валидации
        try:
            cv_metrics = calculate_cross_val_metrics(model, X, y, cv=self.cv, metric='roc_auc')
            metrics.update(cv_metrics)
        except (AttributeError, ValueError):
            # В случае ошибок при кросс-валидации (например, для регрессии)
            # или если модель не поддерживает необходимые методы
            metrics.update({
                'cv_roc_auc_mean': np.nan,
                'cv_roc_auc_std': np.nan
            })
        
        return metrics 