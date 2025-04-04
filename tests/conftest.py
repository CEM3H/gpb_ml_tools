import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from core.data.data_loaders import FileDataLoader
from core.preprocessing.preprocessors import DefaultPreprocessor
from core.feature_selection.selectors import RandomForestFeatureSelector
from core.validation.validators import DefaultModelValidator
from core.models.model_types import CatBoostModel
from core.container import ModelMetadata

@pytest.fixture(scope="session")
def test_data():
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
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
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
def feature_names(test_data):
    """Фикстура для имен признаков"""
    return [col for col in test_data.columns if col != 'target']

@pytest.fixture(scope="session")
def data_loader():
    """Фикстура для загрузчика данных"""
    return FileDataLoader()

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
    """Фикстура для модели"""
    return CatBoostModel({
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 3,
        'random_seed': 42,
        'verbose': False
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