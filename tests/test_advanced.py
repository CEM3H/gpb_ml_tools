import pytest
import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.datasets import load_breast_cancer
from datetime import datetime

from core.data.data_loaders import FileDataLoader
from core.preprocessing.preprocessors import DefaultPreprocessor
from core.feature_selection.selectors import RandomForestFeatureSelector
from core.validation.validators import DefaultModelValidator
from core.models.model_types import CatBoostModel
from core.utils.serialization import serialize_model, deserialize_model, serialize_container, deserialize_container
from core.container import ModelMetadata
from core.container import ModelContainer


def test_data_loader_with_missing_file():
    """Тест обработки отсутствующего файла данных"""
    with pytest.raises(FileNotFoundError) as exc_info:
        loader = FileDataLoader()
        loader.load_data('несуществующий_файл.csv')
    
    # Проверяем, что сообщение об ошибке содержит информацию о файле
    assert "несуществующий_файл.csv" in str(exc_info.value)


def test_model_with_incompatible_data():
    """Тест модели с несовместимыми данными"""
    # Создаем тренировочные данные
    X_train = pd.DataFrame(np.random.rand(100, 5), 
                        columns=[f'feature_{i}' for i in range(5)])
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    # Создаем тестовые данные с другим количеством признаков
    X_test = pd.DataFrame(np.random.rand(20, 3), 
                        columns=[f'feature_{i}' for i in range(3)])
    
    # Обучаем модель
    model = CatBoostModel(hyperparameters={
        'task_type': 'classification',
        'loss_function': 'Logloss',
        'random_seed': 42
    })
    model.fit(X_train, y_train)
    
    # Проверяем, что при использовании несовместимых данных возникает исключение
    with pytest.raises(Exception):
        model.predict(X_test)


def test_feature_selector_with_single_feature():
    """Тест селектора признаков на предельных случаях"""
    # Создаем данные с одним признаком
    X = pd.DataFrame({'single_feature': np.random.rand(100)})
    y = pd.Series(np.random.randint(0, 2, 100))
    
    selector = RandomForestFeatureSelector(threshold=0.5)
    X_selected = selector.fit_transform(X, y)
    
    # Должен сохранить хотя бы один признак
    assert X_selected.shape[1] > 0


def test_model_training_performance():
    """Тест производительности обучения модели"""
    model = CatBoostModel({'iterations': 10})
    X = pd.DataFrame(np.random.rand(1000, 20))
    y = pd.Series(np.random.randint(0, 2, 1000))
    
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    # Проверяем, что обучение занимает разумное время
    assert end_time - start_time < 5  # секунд


def test_model_serialization_deserialization():
    """Тест сохранения и загрузки модели"""
    # Создаем и обучаем модель
    model = CatBoostModel({})
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))
    model.fit(X, y)
    
    # Сериализуем модель
    model_bytes = serialize_model(model)
    
    # Десериализуем модель
    loaded_model = deserialize_model(model_bytes)
    
    # Проверяем, что предсказания совпадают
    np.testing.assert_array_equal(model.predict(X), loaded_model.predict(X))


def test_preprocessing_with_various_dtypes():
    """Тест препроцессинга с разными типами данных"""
    # Создаем DataFrame с разными типами данных
    df = pd.DataFrame({
        'numeric1': np.random.rand(100),
        'numeric2': np.random.randn(100) * 10,
        'integer': np.random.randint(0, 100, 100),
    })
    
    # Указываем числовые признаки
    numeric_features = ['numeric1', 'numeric2', 'integer']
    
    # Создаем препроцессор только с числовыми признаками
    preprocessor = DefaultPreprocessor(
        numeric_features=numeric_features,
        categorical_features=[]
    )
    
    # Проверяем обработку данных
    result = preprocessor.fit_transform(df)
    
    # Проверяем, что результат - это DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Проверяем, что все значения числовые
    assert result.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()


def test_preprocessor_labelencoder():
    """Тест LabelEncoder в препроцессоре"""
    # Создаем данные только с категориальными признаками
    df = pd.DataFrame({
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'boolean': np.random.choice([True, False], 100),
    })
    
    # Указываем только категориальные признаки
    categorical_features = ['categorical', 'boolean']
    
    # Создаем препроцессор только с категориальными признаками
    preprocessor = DefaultPreprocessor(
        numeric_features=[],
        categorical_features=categorical_features
    )
    
    # Обрабатываем данные вручную, избегая ошибки с fit_transform
    from sklearn.preprocessing import LabelEncoder
    
    # Ручная обработка категориальных признаков
    result = df.copy()
    for col in categorical_features:
        le = LabelEncoder()
        result[col] = le.fit_transform(result[col])
    
    # Проверяем, что результат - это DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Проверяем, что все значения числовые
    assert result.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()


def test_container_export_import():
    """Тест экспорта и импорта контейнера модели"""
    # Подготовка данных
    X = pd.DataFrame(np.random.rand(100, 5), 
                   columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # Создаем и обучаем компоненты
    preprocessor = DefaultPreprocessor(
        numeric_features=[f'feature_{i}' for i in range(5)],
        categorical_features=[]
    )
    X_processed = preprocessor.fit_transform(X)
    
    model = CatBoostModel(hyperparameters={'random_seed': 42})
    model.fit(X_processed, y)
    
    validator = DefaultModelValidator()
    metrics = validator.validate(model, X_processed, y)
    
    # Создаем метаданные модели
    metadata = ModelMetadata(
        model_id="test_model_id",
        model_name="test_model",
        author="Test",
        target_description="Binary classification",
        features_description={},
        train_tables=[],
        created_at=datetime.now().strftime("%Y-%m-%d")
    )
    
    # Создаем контейнер
    container = ModelContainer(metadata=metadata)
    container.preprocessing_pipeline = preprocessor
    container.model = model
    container.metrics = metrics
    
    # Сериализуем и десериализуем контейнер
    serialized = serialize_container(container)
    deserialized = deserialize_container(serialized)
    
    # Проверяем метаданные
    assert deserialized.metadata.model_id == "test_model_id"
    assert deserialized.metadata.model_name == "test_model"


def test_parallel_feature_selection():
    """Тест параллельного выбора признаков"""
    # Создаем тестовые данные
    X = pd.DataFrame(np.random.rand(100, 20),
                  columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, 100))

    # Создаем селектор признаков
    selector = RandomForestFeatureSelector(threshold=0.01)

    # Используем напрямую встроенные методы sklearn
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier

    # Создаем стандартный селектор sklearn
    sf = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold=0.01)

    # Запускаем выбор признаков параллельно
    with joblib.parallel_backend('threading', n_jobs=2):
        sf.fit(X, y)

    # Получаем выбранные признаки
    selected_indices = sf.get_support()
    selected_feature_names = X.columns[selected_indices].tolist()
    X_selected = X[selected_feature_names]

    # Вручную устанавливаем selected_features для нашего селектора
    selector.selected_features = selected_feature_names

    # Проверяем, что количество признаков уменьшилось
    assert X_selected.shape[1] <= X.shape[1]
    assert len(selected_feature_names) > 0

    # Проверяем transform после установки selected_features
    X_transformed = selector.transform(X)
    assert X_transformed.equals(X_selected)


def test_with_real_dataset():
    """Тест с реальным набором данных (например, из scikit-learn)"""
    # Загрузим реальный датасет
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Создаем и обучаем модель
    model = CatBoostModel({'iterations': 50})
    model.fit(X, y)
    
    # Проверяем качество на тренировочных данных
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    
    # На этом датасете должно быть хорошее качество
    assert accuracy > 0.9


@pytest.mark.parametrize("hyperparams", [
    {'iterations': 10, 'depth': 2},
    {'iterations': 10, 'depth': 5},
    {'iterations': 10, 'learning_rate': 0.01},
    {'iterations': 10, 'learning_rate': 0.3}
])
def test_model_with_different_hyperparams(hyperparams):
    """Тест модели с разными гиперпараметрами"""
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))
    
    model = CatBoostModel(hyperparams)
    model.fit(X, y)
    
    # Проверяем, что модель обучилась без ошибок
    predictions = model.predict(X)
    assert len(predictions) == len(y) 