Тестирование
===========

Общие принципы
------------

Библиотека GPB использует pytest для тестирования. Тесты организованы в соответствии со структурой библиотеки:

.. code-block:: text

    tests/
    ├── __init__.py
    ├── conftest.py
    ├── data_loading/
    │   ├── __init__.py
    │   ├── test_base.py
    │   ├── test_file_loader.py
    │   ├── test_postgres_loader.py
    │   └── test_impala_loader.py
    ├── data_validation/
    │   ├── __init__.py
    │   ├── test_base.py
    │   ├── test_schema.py
    │   └── test_validators/
    │       ├── __init__.py
    │       ├── test_numeric.py
    │       ├── test_string.py
    │       └── test_datetime.py
    ├── feature_selection/
    │   ├── __init__.py
    │   ├── test_base.py
    │   ├── test_statistical/
    │   │   ├── __init__.py
    │   │   ├── test_variance.py
    │   │   ├── test_correlation.py
    │   │   └── test_mutual_info.py
    │   └── test_model_based/
    │       ├── __init__.py
    │       ├── test_rfe.py
    │       └── test_importance.py
    └── preprocessing/
        ├── __init__.py
        ├── test_base.py
        ├── test_numeric/
        │   ├── __init__.py
        │   ├── test_scalers.py
        │   └── test_transformers.py
        ├── test_categorical/
        │   ├── __init__.py
        │   ├── test_encoders.py
        │   └── test_transformers.py
        └── test_imputation/
            ├── __init__.py
            ├── test_simple.py
            └── test_knn.py

Запуск тестов
-----------

Для запуска всех тестов:

.. code-block:: bash

    pytest

Для запуска тестов конкретного модуля:

.. code-block:: bash

    pytest tests/data_loading/
    pytest tests/data_validation/
    pytest tests/feature_selection/
    pytest tests/preprocessing/

Для запуска тестов с покрытием:

.. code-block:: bash

    pytest --cov=core

Фикстуры
-------

В ``conftest.py`` определены общие фикстуры для тестов:

.. code-block:: python

    import pytest
    import pandas as pd
    import numpy as np
    
    @pytest.fixture
    def sample_dataframe():
        """Создание тестового DataFrame."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'score': [85.5, 90.0, 95.5]
        })
    
    @pytest.fixture
    def sample_schema():
        """Создание тестовой схемы данных."""
        return DataSchema([
            ColumnSchema('id', DataType.INTEGER, required=True),
            ColumnSchema('name', DataType.STRING, required=True),
            ColumnSchema('age', DataType.INTEGER, constraints={'min': 0, 'max': 150}),
            ColumnSchema('score', DataType.FLOAT, constraints={'min': 0.0, 'max': 100.0})
        ])

Примеры тестов
-----------

Тестирование загрузчика данных
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pytest
    from core.data_loading import FileDataLoader
    
    def test_file_loader_load_data(sample_dataframe, tmp_path):
        # Создание тестового файла
        file_path = tmp_path / "test.csv"
        sample_dataframe.to_csv(file_path, index=False)
        
        # Тестирование загрузки
        loader = FileDataLoader()
        df = loader.load_data(str(file_path))
        
        # Проверки
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)
        assert df.equals(sample_dataframe)

Тестирование валидатора
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pytest
    from core.data_validation import SchemaValidator
    
    def test_schema_validator_validate(sample_dataframe, sample_schema):
        # Тестирование валидации
        validator = SchemaValidator(sample_schema)
        validator.validate(sample_dataframe)  # Не должно вызывать исключений
    
    def test_schema_validator_validate_invalid(sample_dataframe, sample_schema):
        # Создание невалидных данных
        invalid_df = sample_dataframe.copy()
        invalid_df.loc[0, 'age'] = -1
        
        # Тестирование валидации
        validator = SchemaValidator(sample_schema)
        with pytest.raises(ValueError):
            validator.validate(invalid_df)

Тестирование селектора признаков
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pytest
    from core.feature_selection import StatisticalFeatureSelector
    from core.feature_selection.statistical import VarianceThreshold
    
    def test_feature_selector_select_features():
        # Создание тестовых данных
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1, 1, 1, 1, 1],  # Нулевая дисперсия
            'feature3': [1, 2, 1, 2, 1]
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        # Тестирование отбора
        selector = StatisticalFeatureSelector([
            VarianceThreshold(threshold=0.1)
        ])
        selected_features = selector.select_features(X, y)
        
        # Проверки
        assert len(selected_features) == 2
        assert 'feature2' not in selected_features

Тестирование препроцессора
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pytest
    from core.preprocessing import NumericPreprocessor
    from core.preprocessing.numeric import StandardScaler
    
    def test_preprocessor_fit_transform():
        # Создание тестовых данных
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        # Тестирование преобразования
        preprocessor = NumericPreprocessor([
            StandardScaler()
        ])
        X_transformed = preprocessor.fit_transform(X)
        
        # Проверки
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == X.shape
        assert np.allclose(X_transformed.mean(), 0, atol=1e-10)
        assert np.allclose(X_transformed.std(), 1, atol=1e-10)

Интеграционные тесты
-----------------

Для тестирования взаимодействия компонентов:

.. code-block:: python

    import pytest
    from core.data_loading import FileDataLoader
    from core.data_validation import SchemaValidator
    from core.feature_selection import FeatureSelector
    from core.preprocessing import Preprocessor
    
    def test_full_pipeline(tmp_path):
        # Создание тестовых данных
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [0, 1, 0],
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C']
        })
        
        # Сохранение данных
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        # Загрузка данных
        loader = FileDataLoader()
        loaded_df = loader.load_data(str(file_path))
        
        # Валидация
        schema = DataSchema([
            ColumnSchema('id', DataType.INTEGER, required=True),
            ColumnSchema('target', DataType.INTEGER, required=True),
            ColumnSchema('feature1', DataType.FLOAT),
            ColumnSchema('feature2', DataType.STRING)
        ])
        validator = SchemaValidator(schema)
        validator.validate(loaded_df)
        
        # Отбор признаков
        X = loaded_df.drop('target', axis=1)
        y = loaded_df['target']
        selector = FeatureSelector([...])
        X_selected = selector.select_features(X, y)
        
        # Предобработка
        preprocessor = Preprocessor([...])
        X_processed = preprocessor.fit_transform(X_selected)
        
        # Проверки
        assert isinstance(X_processed, pd.DataFrame)
        assert len(X_processed) == len(df)
        assert 'target' not in X_processed.columns 