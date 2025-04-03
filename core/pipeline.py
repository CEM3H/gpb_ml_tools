from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

from .container import ModelContainer, ModelMetadata
from .data.base import DataLoader
from .preprocessing.base import DataPreprocessor
from .feature_selection.base import FeatureSelector
from .validation.base import ModelValidator
from .models.base import BaseModel
from .metrics.classification import calculate_basic_metrics, calculate_cross_val_metrics
from .reporting.word import WordReportGenerator

class ModelPipeline:
    """Класс для управления пайплайном разработки модели"""
    
    def __init__(
        self,
        data_loader: DataLoader,
        preprocessor: DataPreprocessor,
        feature_selector: FeatureSelector,
        model: BaseModel,
        validator: ModelValidator
    ):
        """
        Инициализация пайплайна
        
        Args:
            data_loader: загрузчик данных
            preprocessor: препроцессор
            feature_selector: селектор признаков
            model: модель
            validator: валидатор
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.model = model
        self.validator = validator
        
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.container = None
        
    def load_data(self) -> None:
        """Загрузка данных"""
        self.data = self.data_loader.load_data()
        
    def preprocess_data(self, test_size: float = 0.2, target_column: str = 'target') -> None:
        """Препроцессинг данных"""
        if self.data is None:
            raise ValueError("Сначала необходимо загрузить данные")
            
        # Разделяем данные на признаки и целевую переменную
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Разделяем данные на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Применяем препроцессинг
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        
    def select_features(self) -> None:
        """Отбор признаков"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Сначала необходимо обработать данные")
            
        self.X_train = self.feature_selector.fit_transform(self.X_train, self.y_train)
        self.X_test = self.feature_selector.transform(self.X_test)
        
    def train_model(self) -> None:
        """Обучение модели"""
        if self.X_train is None or self.X_test is None:
            raise ValueError("Сначала необходимо выбрать признаки")
            
        self.model.fit(self.X_train, self.y_train)
        
    def calibrate_model(self) -> None:
        """Калибровка модели"""
        if self.model is None:
            raise ValueError("Сначала необходимо обучить модель")
            
        self.model.calibrate(self.X_train, self.y_train)
        
    def validate_model(self) -> Dict[str, float]:
        """Валидация модели"""
        if self.model is None:
            raise ValueError("Сначала необходимо обучить модель")
            
        return self.validator.validate(self.model, self.X_test, self.y_test)
        
    def create_container(self, metadata: ModelMetadata) -> None:
        """Создание контейнера модели"""
        if self.model is None:
            raise ValueError("Сначала необходимо обучить модель")
            
        self.container = ModelContainer(metadata)
        self.container.model = self.model
        self.container.preprocessing_pipeline = self.preprocessor.pipeline
        self.container.calibration_model = self.model.calibration_model
        self.container.feature_importance = self.model.get_feature_importance()
        self.container.metrics = self.validate_model()
        
    def run_pipeline(self, metadata: ModelMetadata) -> ModelContainer:
        """
        Запуск полного пайплайна
        
        Args:
            metadata: метаданные модели
            
        Returns:
            ModelContainer: контейнер с обученной моделью
        """
        self.load_data()
        self.preprocess_data()
        self.select_features()
        self.train_model()
        self.calibrate_model()
        self.create_container(metadata)
        return self.container
        
    def generate_report(self, output_path: str) -> str:
        """
        Генерация отчета о модели в формате MS Word
        
        Args:
            output_path: путь для сохранения отчета
            
        Returns:
            str: путь к сохраненному отчету
        """
        if self.container is None:
            raise ValueError("Сначала необходимо создать контейнер модели")
            
        report_generator = WordReportGenerator(self.container)
        return report_generator.generate_report(self.X_test, self.y_test, output_path) 