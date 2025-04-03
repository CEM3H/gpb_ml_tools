Примеры использования
===================

Загрузка данных
-------------

Загрузка из CSV файла
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.data_loading import FileDataLoader
    
    # Создание загрузчика
    loader = FileDataLoader()
    
    # Загрузка данных
    df = loader.load_data(
        'data.csv',
        params={
            'sep': ',',
            'encoding': 'utf-8',
            'decimal': '.',
            'thousands': ','
        }
    )
    
    print(f"Загружено строк: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")

Загрузка из PostgreSQL
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.data.data_loaders import PostgresDataLoader
    
    # Создание загрузчика
    loader = PostgresDataLoader(
        host='localhost',
        port=5432,
        database='mydb',
        user='user',
        password='password'
    )
    
    # Загрузка данных
    data = loader.load_data('SELECT * FROM my_table')

Загрузка из Impala
~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.data.data_loaders import ImpalaDataLoader
    
    # Создание загрузчика
    loader = ImpalaDataLoader(
        host='localhost',
        port=21050,
        database='mydb'
    )
    
    # Загрузка данных
    data = loader.load_data('SELECT * FROM my_table')

Валидация данных
---------------

Базовая валидация
~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.validation.validators import NumericValidator, StringValidator
    
    # Создание валидаторов
    numeric_validator = NumericValidator(min_value=0, max_value=100)
    string_validator = StringValidator(min_length=1, max_length=50)
    
    # Валидация данных
    is_valid = numeric_validator.validate(42)
    is_valid = string_validator.validate('Hello')

Валидация дат и времени
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.validation.validators import DateTimeValidator
    
    # Создание валидатора
    date_validator = DateTimeValidator(
        format='%Y-%m-%d',
        min_date='2020-01-01',
        max_date='2023-12-31'
    )
    
    # Валидация даты
    is_valid = date_validator.validate('2023-01-01')

Отбор признаков
-------------

Статистический отбор
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.feature_selection.selectors import VarianceThreshold
    
    # Создание селектора
    selector = VarianceThreshold(threshold=0.1)
    
    # Отбор признаков
    selected_features = selector.select_features(X)

Отбор на основе моделей
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.feature_selection.selectors import MutualInfoSelector
    
    # Создание селектора
    selector = MutualInfoSelector(n_features=10)
    
    # Отбор признаков
    selected_features = selector.select_features(X, y)

Предобработка данных
------------------

Предобработка числовых признаков
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.preprocessing.preprocessors import StandardScaler
    
    # Создание препроцессора
    scaler = StandardScaler()
    
    # Предобработка данных
    X_scaled = scaler.fit_transform(X)

Предобработка категориальных признаков
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.preprocessing.preprocessors import OneHotEncoder
    
    # Создание препроцессора
    encoder = OneHotEncoder()
    
    # Предобработка данных
    X_encoded = encoder.fit_transform(X)

Полный пайплайн
-------------

.. code-block:: python

    from core.pipeline import ModelPipeline
    from core.data.data_loaders import FileDataLoader
    from core.validation.validators import SchemaValidator
    from core.feature_selection.selectors import VarianceThreshold
    from core.preprocessing.preprocessors import StandardScaler
    from core.models.model_types import ClassificationModel
    
    # Создание пайплайна
    pipeline = ModelPipeline(
        data_loader=FileDataLoader('data.csv'),
        validator=SchemaValidator(schema),
        feature_selector=VarianceThreshold(threshold=0.1),
        preprocessor=StandardScaler(),
        model=ClassificationModel()
    )
    
    # Запуск пайплайна
    pipeline.run_pipeline() 