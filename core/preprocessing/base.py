from typing import Dict
import pandas as pd
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """Класс для предобработки данных"""
    def __init__(self):
        self.pipeline: Pipeline = Pipeline([])
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обучение и применение пайплайна предобработки"""
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение пайплайна предобработки"""
        pass 