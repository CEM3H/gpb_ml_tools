import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from core.data.data_loaders import FileDataLoader
from core.preprocessing.preprocessors import DefaultPreprocessor
from core.feature_selection.selectors import RandomForestFeatureSelector
from core.validation.validators import DefaultModelValidator
from core.models.model_types import CatBoostModel

@pytest.fixture
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

@pytest.fixture
def data_loader():
    """Фикстура для загрузчика данных"""
    return FileDataLoader({
        'file_path': 'test_data.csv',
        'file_type': 'csv'
    })

@pytest.fixture
def preprocessor():
    """Фикстура для препроцессора"""
    feature_names = [f'feature_{i}' for i in range(20)]
    return DefaultPreprocessor(
        numeric_features=feature_names,
        categorical_features=[]
    )

@pytest.fixture
def feature_selector():
    """Фикстура для селектора признаков"""
    return RandomForestFeatureSelector(
        threshold=0.05,
        blacklist=['target']
    )

@pytest.fixture
def validator():
    """Фикстура для валидатора"""
    return DefaultModelValidator(cv=3)

@pytest.fixture
def model():
    """Фикстура для модели"""
    return CatBoostModel({
        'task_type': 'classification',
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 3,
        'random_seed': 42,
        'verbose': False,
        'loss_function': 'Logloss'
    })

def test_data_loader(data_loader, test_data):
    """Тест загрузчика данных"""
    # Загрузка данных
    data = data_loader.load_data()
    
    # Проверка результата
    assert isinstance(data, pd.DataFrame)
    assert data.shape == test_data.shape
    assert all(data.columns == test_data.columns)

def test_preprocessor(preprocessor, test_data):
    """Тест препроцессора"""
    X = test_data.drop(columns=['target'])
    y = test_data['target']
    
    # Обучение препроцессора
    preprocessor.fit_transform(X)
    
    # Преобразование данных
    X_transformed = preprocessor.transform(X)
    
    # Проверка результата
    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] == X.shape[0]
    assert not X_transformed.isna().any().any()

def test_feature_selector(feature_selector, test_data):
    """Тест селектора признаков"""
    X = test_data.drop(columns=['target'])
    y = test_data['target']
    
    # Обучение селектора
    feature_selector.fit(X, y)
    
    # Отбор признаков
    X_selected = feature_selector.transform(X)
    
    # Проверка результата
    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.shape[0] == X.shape[0]
    assert X_selected.shape[1] <= X.shape[1]
    
    # Проверка fit_transform
    X_selected_2 = feature_selector.fit_transform(X, y)
    assert isinstance(X_selected_2, pd.DataFrame)
    assert X_selected_2.shape[0] == X.shape[0]
    assert X_selected_2.shape[1] <= X.shape[1]

def test_validator(validator, model, test_data):
    """Тест валидатора"""
    X = test_data.drop(columns=['target'])
    y = test_data['target']
    
    # Обучение модели
    model.fit(X, y)
    
    # Тест validate
    metrics = validator.validate(model, X, y)
    assert isinstance(metrics, dict)
    assert 'roc_auc' in metrics
    assert 'cv_roc_auc_mean' in metrics
    assert 'cv_roc_auc_std' in metrics
    assert 'average_precision' in metrics
    
    # Проверка разумности метрик
    assert 0 <= metrics['roc_auc'] <= 1
    if not np.isnan(metrics['cv_roc_auc_mean']):
        assert 0 <= metrics['cv_roc_auc_mean'] <= 1
    if not np.isnan(metrics['cv_roc_auc_std']):
        assert 0 <= metrics['cv_roc_auc_std'] <= 1
    assert 0 <= metrics['average_precision'] <= 1

def test_model(model, test_data):
    """Тест модели"""
    X = test_data.drop(columns=['target'])
    y = test_data['target']
    
    # Обучение модели
    model.fit(X, y)
    
    # Предсказание
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Проверка результатов
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_proba, np.ndarray)
    assert len(y_pred) == len(y)
    assert y_proba.shape == (len(y), 2)
    
    # Калибровка
    model.calibrate(X, y)
    assert model.calibration_model is not None
    
    # Важность признаков
    feature_importance = model.get_feature_importance()
    assert isinstance(feature_importance, dict)
    assert len(feature_importance) == X.shape[1] 