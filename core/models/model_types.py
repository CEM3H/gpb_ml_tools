from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.calibration import CalibratedClassifierCV
from .base import BaseModel

class RandomForestModel(BaseModel):
    """Реализация Random Forest модели"""
    def __init__(self, model_params: Dict = None):
        super().__init__(model_params)
        self.model = RandomForestClassifier(**(self.hyperparameters or {}))
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
        
    def get_feature_importance(self) -> Dict[str, float]:
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))

class CatBoostModel(BaseModel):
    """Реализация CatBoost модели"""
    def __init__(self, hyperparameters: dict = None):
        super().__init__()
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        
        # По умолчанию работаем с задачей классификации
        self.task = 'classification'
        
        # Удаляем task_type из гиперпараметров
        if 'task_type' in self.hyperparameters and self.hyperparameters['task_type'] == 'classification':
            self.hyperparameters.pop('task_type')
        
        self.model = CatBoostClassifier(**self.hyperparameters)
        self.calibration_model = None
    
    def get_params(self, deep=True):
        """Метод для sklearn API"""
        return {'hyperparameters': self.hyperparameters}
    
    def fit(self, X, y):
        """
        Обучение модели
        
        Args:
            X: Признаки для обучения
            y: Целевая переменная
        
        Returns:
            self
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Предсказание значений
        
        Args:
            X: Признаки для предсказания
        
        Returns:
            Предсказанные значения
        """
        if self.calibration_model is not None:
            return self.calibration_model.predict(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей
        
        Args:
            X: Признаки для предсказания
        
        Returns:
            Предсказанные вероятности
        """
        if self.task != 'classification':
            raise ValueError("Метод predict_proba доступен только для задач классификации")
        
        if self.calibration_model is not None:
            return self.calibration_model.predict_proba(X)
        return self.model.predict_proba(X)
    
    def calibrate(self, X, y):
        """
        Калибровка вероятностей модели
        
        Args:
            X: Признаки для калибровки
            y: Целевая переменная
        """
        if self.task != 'classification':
            raise ValueError("Калибровка доступна только для задач классификации")
        
        calibrator = CalibratedClassifierCV(
            estimator=self.model,
            method='isotonic',
            cv='prefit'
        )
        calibrator.fit(X, y)
        self.calibration_model = calibrator
        return self
    
    def get_feature_importance(self, feature_names=None):
        """
        Получение важности признаков
        
        Args:
            feature_names: Названия признаков
            
        Returns:
            Словарь с важностью признаков
        """
        importances = self.model.get_feature_importance()
        
        if feature_names is None:
            return {f"feature_{i}": importance for i, importance in enumerate(importances)}
        else:
            return {name: importance for name, importance in zip(feature_names, importances)} 