from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from .base import FeatureSelector

class RandomForestFeatureSelector(FeatureSelector):
    """Реализация селектора признаков на основе Random Forest"""
    def __init__(self, threshold: float = 0.3, blacklist: Optional[List[str]] = None):
        super().__init__(
            selection_params={'threshold': threshold},
            blacklist=blacklist
        )
        self.threshold = threshold
        self.selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100),
            threshold=threshold
        )
        self.selected_features = None
        self.importance_scores = None
        
    def _fit(self, df: pd.DataFrame, target: pd.Series) -> 'RandomForestFeatureSelector':
        """
        Внутренний метод для обучения селектора
        
        Args:
            df: DataFrame с признаками
            target: Целевая переменная
            
        Returns:
            self: Обученный селектор
        """
        self.selector.fit(df, target)
        self.selected_features = df.columns[self.selector.get_support()].tolist()
        self.importance_scores = self.selector.estimator_.feature_importances_
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование данных с использованием отобранных признаков
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            DataFrame с отобранными признаками
        """
        if not self.selected_features:
            raise ValueError("Сначала необходимо вызвать метод fit")
        return df[self.selected_features]
        
    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        Обучение и применение селектора признаков
        
        Args:
            df: DataFrame с признаками
            target: Целевая переменная
            
        Returns:
            DataFrame с отобранными признаками
        """
        return self._fit(df, target).transform(df)

class CorrelationFeatureSelector(FeatureSelector):
    """Реализация селектора признаков на основе корреляций"""
    def __init__(
        self,
        correlation_threshold: float = 0.95,
        blacklist: Optional[List[str]] = None
    ):
        super().__init__(
            selection_params={'correlation_threshold': correlation_threshold},
            blacklist=blacklist
        )
        self.correlation_threshold = correlation_threshold
        
    def _fit(self, df: pd.DataFrame, target: pd.Series) -> 'CorrelationFeatureSelector':
        """
        Внутренний метод для обучения селектора
        
        Args:
            df: DataFrame с признаками
            target: Целевая переменная
            
        Returns:
            self: Обученный селектор
        """
        # Вычисляем корреляционную матрицу
        corr_matrix = df.corr().abs()
        
        # Находим верхний треугольник матрицы
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Находим признаки для удаления
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]
        
        # Сохраняем оставшиеся признаки
        self.selected_features = [col for col in df.columns if col not in to_drop]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование данных с использованием отобранных признаков
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            DataFrame с отобранными признаками
        """
        if not self.selected_features:
            raise ValueError("Сначала необходимо вызвать метод fit")
        return df[self.selected_features]
        
    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        Обучение и применение селектора признаков
        
        Args:
            df: DataFrame с признаками
            target: Целевая переменная
            
        Returns:
            DataFrame с отобранными признаками
        """
        return self._fit(df, target).transform(df) 