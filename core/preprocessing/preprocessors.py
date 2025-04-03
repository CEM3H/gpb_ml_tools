from typing import Dict, List
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from .base import DataPreprocessor

class DefaultPreprocessor(DataPreprocessor):
    """Реализация стандартного препроцессора"""
    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        super().__init__()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.pipeline = self._create_pipeline()
        
    def _create_pipeline(self) -> Pipeline:
        """Создание пайплайна предобработки"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', LabelEncoder())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, self.numeric_features),
                ('categorical', categorical_transformer, self.categorical_features)
            ]
        )
        
        return Pipeline([
            ('preprocessor', preprocessor)
        ])
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обучение и применение препроцессора"""
        return pd.DataFrame(
            self.pipeline.fit_transform(df),
            columns=self.numeric_features + self.categorical_features
        )
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение препроцессора"""
        return pd.DataFrame(
            self.pipeline.transform(df),
            columns=self.numeric_features + self.categorical_features
        ) 