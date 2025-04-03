Архитектура библиотеки
====================

Обзор
-----

Библиотека GPB построена по модульному принципу и состоит из следующих основных компонентов:

* Модуль загрузки данных (``data_loading``)
* Модуль валидации данных (``data_validation``)
* Модуль отбора признаков (``feature_selection``)
* Модуль предобработки данных (``preprocessing``)

Каждый модуль представляет собой независимый компонент, который может использоваться как отдельно, так и в составе полного пайплайна обработки данных.

Модуль загрузки данных
--------------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── data_loading/
        ├── __init__.py
        ├── base.py
        ├── file_loader.py
        ├── postgres_loader.py
        └── impala_loader.py

Основные классы:

* ``DataLoader`` - базовый класс для загрузки данных
* ``FileDataLoader`` - загрузка данных из файлов (CSV, JSON, YAML)
* ``PostgresDataLoader`` - загрузка данных из PostgreSQL
* ``ImpalaDataLoader`` - загрузка данных из Impala

Модуль валидации данных
---------------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── data_validation/
        ├── __init__.py
        ├── base.py
        ├── schema.py
        └── validators/
            ├── __init__.py
            ├── base.py
            ├── numeric.py
            ├── string.py
            └── datetime.py

Основные классы:

* ``DataSchema`` - описание структуры данных
* ``ColumnSchema`` - описание колонки данных
* ``SchemaValidator`` - базовый валидатор
* Специализированные валидаторы для разных типов данных

Модуль отбора признаков
--------------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── feature_selection/
        ├── __init__.py
        ├── base.py
        ├── statistical/
        │   ├── __init__.py
        │   ├── variance.py
        │   ├── correlation.py
        │   └── mutual_info.py
        └── model_based/
            ├── __init__.py
            ├── rfe.py
            └── importance.py

Основные классы:

* ``FeatureSelector`` - базовый класс для отбора признаков
* ``StatisticalFeatureSelector`` - отбор на основе статистических методов
* ``ModelBasedFeatureSelector`` - отбор на основе моделей машинного обучения

Модуль предобработки данных
-------------------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── preprocessing/
        ├── __init__.py
        ├── base.py
        ├── numeric/
        │   ├── __init__.py
        │   ├── scalers.py
        │   └── transformers.py
        ├── categorical/
        │   ├── __init__.py
        │   ├── encoders.py
        │   └── transformers.py
        └── imputation/
            ├── __init__.py
            ├── simple.py
            └── knn.py

Основные классы:

* ``Preprocessor`` - базовый класс для предобработки данных
* ``NumericPreprocessor`` - предобработка числовых признаков
* ``CategoricalPreprocessor`` - предобработка категориальных признаков

Взаимодействие компонентов
------------------------

Компоненты библиотеки взаимодействуют следующим образом:

1. Данные загружаются с помощью одного из загрузчиков (``DataLoader``)
2. Загруженные данные проверяются на соответствие схеме (``SchemaValidator``)
3. Из данных отбираются наиболее значимые признаки (``FeatureSelector``)
4. Отобранные признаки проходят предобработку (``Preprocessor``)
5. Подготовленные данные могут быть использованы для обучения моделей

Пример взаимодействия:

.. code-block:: python

    # Загрузка данных
    loader = FileDataLoader()
    df = loader.load_data('data.csv')
    
    # Валидация
    validator = SchemaValidator(schema)
    validator.validate(df)
    
    # Отбор признаков
    selector = FeatureSelector([...])
    X_selected = selector.select_features(X, y)
    
    # Предобработка
    preprocessor = Preprocessor([...])
    X_processed = preprocessor.fit_transform(X_selected)

Расширяемость
-----------

Библиотека разработана с учетом возможности расширения. Вы можете:

1. Создавать собственные загрузчики данных, наследуясь от ``DataLoader``
2. Добавлять новые типы валидаторов, наследуясь от ``SchemaValidator``
3. Реализовывать новые методы отбора признаков, наследуясь от ``FeatureSelector``
4. Создавать собственные преобразователи данных, наследуясь от ``Preprocessor``

Пример создания собственного валидатора:

.. code-block:: python

    class CustomValidator(SchemaValidator):
        def _validate_custom(self, series: pd.Series, schema: ColumnSchema) -> None:
            # Реализация собственной логики валидации
            pass 