import pytest
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from core.pipeline import ModelPipeline
from core.container import ModelMetadata
from core.data.data_loaders import FileDataLoader
from core.preprocessing.preprocessors import DefaultPreprocessor
from core.feature_selection.selectors import RandomForestFeatureSelector
from core.validation.validators import DefaultModelValidator
from core.models.model_types import CatBoostModel

@pytest.fixture(scope="session")
def feature_names():
    """Фикстура для имен признаков"""
    return [f'feature_{i}' for i in range(20)]

@pytest.fixture(scope="session")
def test_data(feature_names):
    """Фикстура для тестовых данных"""
    # Генерируем тестовые данные
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Создаем DataFrame
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    # Сохраняем тестовые данные
    data.to_csv('test_data.csv', index=False)
    
    yield data
    
    # Очистка после тестов
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')

@pytest.fixture(scope="session")
def data_loader(test_data):
    """Фикстура для загрузчика данных"""
    class TestDataLoader:
        def __init__(self, test_data):
            self.test_data = test_data
            
        def load_data(self, test_size=0.2, random_state=42):
            return self.test_data
            
    return TestDataLoader(test_data)

@pytest.fixture(scope="session")
def preprocessor(feature_names):
    """Фикстура для препроцессора"""
    return DefaultPreprocessor(
        numeric_features=feature_names,
        categorical_features=[]
    )

@pytest.fixture(scope="session")
def feature_selector():
    """Фикстура для селектора признаков"""
    return RandomForestFeatureSelector(
        threshold=0.01,
        blacklist=['target']
    )

@pytest.fixture(scope="session")
def validator():
    """Фикстура для валидатора"""
    return DefaultModelValidator(cv=3)

@pytest.fixture(scope="session")
def model():
    """Фикстура для создания модели"""
    return CatBoostModel(hyperparameters={
        'task_type': 'classification',
        'loss_function': 'Logloss',
        'random_seed': 42
    })

@pytest.fixture(scope="session")
def metadata(feature_names):
    """Фикстура для метаданных"""
    return ModelMetadata(
        model_id='test_model_001',
        model_name='Test Model',
        author='Test Author',
        target_description='Binary classification target',
        features_description={f: f'Feature {i}' for i, f in enumerate(feature_names)},
        train_tables=['test_data.csv'],
        created_at='2024-04-01'
    )

@pytest.fixture(scope="session")
def pipeline(data_loader, preprocessor, feature_selector, validator, model):
    """Фикстура для пайплайна"""
    return ModelPipeline(
        data_loader=data_loader,
        preprocessor=preprocessor,
        feature_selector=feature_selector,
        validator=validator,
        model=model
    )

def test_load_data(pipeline, test_data):
    """Тест загрузки данных"""
    pipeline.load_data()
    
    assert pipeline.data is not None
    assert isinstance(pipeline.data, pd.DataFrame)
    assert len(pipeline.data) == len(test_data)

def test_preprocess_data(pipeline):
    """Тест предобработки данных"""
    pipeline.load_data()
    pipeline.preprocess_data()
    
    assert pipeline.X_train is not None
    assert pipeline.X_test is not None

def test_select_features(pipeline, test_data):
    """Тест выбора признаков"""
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.select_features()
    
    assert pipeline.X_train is not None
    assert pipeline.X_test is not None
    assert len(pipeline.X_train.columns) <= len(test_data.columns) - 1

def test_train_model(pipeline):
    """Тест обучения модели"""
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.select_features()
    pipeline.train_model()
    
    assert pipeline.model is not None

def test_calibrate_model(pipeline):
    """Тест калибровки модели"""
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.select_features()
    pipeline.train_model()
    pipeline.calibrate_model()
    
    assert pipeline.model.calibration_model is not None

def test_create_container(pipeline, metadata):
    """Тест создания контейнера"""
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.select_features()
    pipeline.train_model()
    pipeline.calibrate_model()
    pipeline.create_container(metadata)
    
    assert pipeline.container is not None
    assert pipeline.container.model is not None
    assert pipeline.container.preprocessing_pipeline is not None
    assert pipeline.container.calibration_model is not None
    assert pipeline.container.feature_importance is not None
    assert pipeline.container.metrics is not None

def test_full_pipeline(pipeline, metadata):
    """Тест полного пайплайна"""
    container = pipeline.run_pipeline(metadata)
    
    assert container is not None
    assert container.model is not None
    assert container.preprocessing_pipeline is not None
    assert container.calibration_model is not None
    assert container.feature_importance is not None
    assert container.metrics is not None
    
    # Проверяем, что метрики имеют разумные значения
    assert container.metrics['roc_auc'] > 0.5
    if not np.isnan(container.metrics['cv_roc_auc_mean']):
        assert container.metrics['cv_roc_auc_mean'] > 0.5

if __name__ == '__main__':
    unittest.main() 