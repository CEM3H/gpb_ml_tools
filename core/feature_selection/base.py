from typing import Dict, List, Optional
import pandas as pd

class FeatureSelector:
    """Класс для отбора признаков"""
    def __init__(self, selection_params: Dict, blacklist: Optional[List[str]] = None):
        self.selection_params = selection_params
        self.selected_features: List[str] = []
        self.blacklist = blacklist or []
        
    def fit(self, df: pd.DataFrame, target: pd.Series) -> 'FeatureSelector':
        """
        Обучение селектора признаков
        
        Args:
            df: DataFrame с признаками
            target: Целевая переменная
            
        Returns:
            self: Обученный селектор
        """
        # Исключаем признаки из черного списка
        available_features = [col for col in df.columns if col not in self.blacklist]
        df = df[available_features]
        return self._fit(df, target)
        
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
        return self.fit(df, target).transform(df)
        
    def _fit(self, df: pd.DataFrame, target: pd.Series) -> 'FeatureSelector':
        """
        Внутренний метод для обучения селектора
        
        Args:
            df: DataFrame с признаками
            target: Целевая переменная
            
        Returns:
            self: Обученный селектор
        """
        raise NotImplementedError("Метод _fit должен быть реализован в дочернем классе")
        
    def _select_features(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Внутренний метод для отбора признаков"""
        pass 