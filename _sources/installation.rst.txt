Установка и использование
======================

Установка
--------

Библиотека GPB может быть установлена с помощью pip:

.. code-block:: bash

    pip install gpb-lib

Зависимости
----------

Библиотека требует следующие зависимости:

* Python >= 3.8
* pandas >= 1.3.0
* numpy >= 1.20.0
* scikit-learn >= 0.24.0
* psycopg2-binary >= 2.9.0 (для работы с PostgreSQL)
* impyla >= 0.17.0 (для работы с Impala)
* PyYAML >= 5.4.0

Быстрый старт
-----------

Загрузка данных
~~~~~~~~~~~~

.. code-block:: python

    from core.data_loading import FileDataLoader, PostgresDataLoader, ImpalaDataLoader
    
    # Загрузка из файла
    file_loader = FileDataLoader()
    df = file_loader.load_data('data.csv')
    
    # Загрузка из PostgreSQL
    pg_loader = PostgresDataLoader(
        host='localhost',
        port=5432,
        database='mydb',
        user='user',
        password='password'
    )
    df = pg_loader.load_data('SELECT * FROM mytable')
    
    # Загрузка из Impala
    impala_loader = ImpalaDataLoader(
        host='localhost',
        port=21050,
        database='mydb'
    )
    df = impala_loader.load_data('SELECT * FROM mytable')

Валидация данных
~~~~~~~~~~~~~

.. code-block:: python

    from core.data_validation import SchemaValidator
    from core.data.schema import DataSchema, ColumnSchema, DataType
    
    # Создание схемы
    schema = DataSchema([
        ColumnSchema('id', DataType.INTEGER, required=True),
        ColumnSchema('name', DataType.STRING, required=True),
        ColumnSchema('age', DataType.INTEGER, constraints={'min': 0, 'max': 150})
    ])
    
    # Валидация данных
    validator = SchemaValidator(schema)
    validator.validate(df)

Отбор признаков
~~~~~~~~~~~~

.. code-block:: python

    from core.feature_selection import FeatureSelector
    from core.feature_selection.statistical import VarianceThreshold
    from core.feature_selection.model_based import RFESelector
    from sklearn.ensemble import RandomForestClassifier
    
    # Создание селектора
    selector = FeatureSelector([
        VarianceThreshold(threshold=0.1),
        RFESelector(RandomForestClassifier(), n_features_to_select=10)
    ])
    
    # Отбор признаков
    selected_features = selector.select_features(X, y)

Предобработка данных
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.preprocessing import Preprocessor
    from core.preprocessing.numeric import StandardScaler
    from core.preprocessing.categorical import OneHotEncoder
    
    # Создание препроцессора
    preprocessor = Preprocessor([
        StandardScaler(),
        OneHotEncoder()
    ])
    
    # Предобработка данных
    X_transformed = preprocessor.fit_transform(X)

Пример полного пайплайна
---------------------

.. code-block:: python

    from core.data_loading import FileDataLoader
    from core.data_validation import SchemaValidator
    from core.data.schema import DataSchema, ColumnSchema, DataType
    from core.feature_selection import FeatureSelector
    from core.preprocessing import Preprocessor
    from sklearn.ensemble import RandomForestClassifier
    
    # Загрузка данных
    loader = FileDataLoader()
    df = loader.load_data('data.csv')
    
    # Валидация данных
    schema = DataSchema([
        ColumnSchema('id', DataType.INTEGER, required=True),
        ColumnSchema('target', DataType.INTEGER, required=True),
        ColumnSchema('feature1', DataType.FLOAT),
        ColumnSchema('feature2', DataType.STRING)
    ])
    validator = SchemaValidator(schema)
    validator.validate(df)
    
    # Разделение на признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Отбор признаков
    selector = FeatureSelector([
        VarianceThreshold(threshold=0.1),
        RFESelector(RandomForestClassifier(), n_features_to_select=10)
    ])
    X_selected = selector.select_features(X, y)
    
    # Предобработка данных
    preprocessor = Preprocessor([
        StandardScaler(),
        OneHotEncoder()
    ])
    X_processed = preprocessor.fit_transform(X_selected)
    
    # Обучение модели
    model = RandomForestClassifier()
    model.fit(X_processed, y) 