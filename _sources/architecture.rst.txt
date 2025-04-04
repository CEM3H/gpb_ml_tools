Архитектура библиотеки
====================

Обзор
-----

Библиотека GPB построена по модульному принципу и состоит из следующих основных компонентов:

* Модуль работы с данными (``data``)
* Модуль валидации данных (``validation``)
* Модуль отбора признаков (``feature_selection``)
* Модуль предобработки данных (``preprocessing``)
* Модуль моделей (``models``)
* Модуль отчетов (``reporting``)
* Модуль метрик (``metrics``)
* Модуль утилит (``utils``)
* Модуль пайплайна (``pipeline``)
* Модуль контейнера (``container``)

Каждый модуль представляет собой независимый компонент, который может использоваться как отдельно, так и в составе полного пайплайна обработки данных.

Модуль работы с данными
--------------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── data/
        ├── __init__.py
        ├── base.py
        ├── data_loaders.py
        ├── validation.py
        ├── schema.py
        └── data_types.py

Основные классы:

* ``DataLoader`` - базовый класс для загрузки данных
* ``FileDataLoader`` - загрузка данных из файлов (CSV, JSON, YAML)
* ``PostgresDataLoader`` - загрузка данных из PostgreSQL
* ``ImpalaDataLoader`` - загрузка данных из Impala
* ``SchemaValidator`` - валидатор данных на основе схемы
* ``DataSchema`` - описание структуры данных
* ``ColumnSchema`` - описание колонки данных

Модуль валидации данных
---------------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── validation/
        ├── __init__.py
        ├── base.py
        └── validators.py

Основные классы:

* ``BaseValidator`` - базовый валидатор
* ``NumericValidator`` - валидатор для числовых данных
* ``StringValidator`` - валидатор для строковых данных
* ``DateTimeValidator`` - валидатор для дат и времени

Модуль отбора признаков
--------------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── feature_selection/
        ├── __init__.py
        ├── base.py
        └── selectors.py

Основные классы:

* ``FeatureSelector`` - базовый класс для отбора признаков
* ``RandomForestFeatureSelector`` - отбор на основе случайного леса
* ``CorrelationFeatureSelector`` - отбор на основе корреляций

Модуль предобработки данных
-------------------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── preprocessing/
        ├── __init__.py
        ├── base.py
        └── preprocessors.py

Основные классы:

* ``DataPreprocessor`` - базовый класс для предобработки данных
* ``DefaultPreprocessor`` - стандартный препроцессор

Модуль моделей
------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── models/
        ├── __init__.py
        ├── base.py
        └── model_types.py

Основные классы:

* ``BaseModel`` - базовый класс для моделей
* ``RandomForestModel`` - модель на основе случайного леса
* ``CatBoostModel`` - модель на основе CatBoost

Модуль отчетов
------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── reporting/
        ├── __init__.py
        ├── utils.py
        └── word.py

Основные классы:

* ``WordReportGenerator`` - генератор отчетов в формате Word

Модуль метрик
-----------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── metrics/
        ├── __init__.py
        ├── classification.py
        └── regression.py

Основные функции:

* Метрики для задач классификации
* Метрики для задач регрессии

Модуль утилит
-----------

Структура
~~~~~~~~

.. code-block:: text

    core/
    └── utils/
        ├── __init__.py
        ├── serialization.py
        └── file_utils.py

Основные классы и функции:

* ``NumpyEncoder`` - JSON энкодер для NumPy типов
* Функции для сериализации и десериализации моделей
* Функции для безопасной работы с файлами

Модуль пайплайна
--------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    ├── __init__.py
    └── pipeline.py

Основные классы:

* ``ModelPipeline`` - класс для построения полного пайплайна обработки данных

Модуль контейнера
--------------

Структура
~~~~~~~~

.. code-block:: text

    core/
    ├── __init__.py
    └── container.py

Основные классы:

* ``ModelContainer`` - контейнер для хранения модели и связанных с ней данных
* ``ModelMetadata`` - метаданные модели

Взаимодействие компонентов
------------------------

Компоненты библиотеки взаимодействуют следующим образом:

1. Данные загружаются с помощью одного из загрузчиков (``DataLoader``)
2. Загруженные данные проверяются на соответствие схеме (``SchemaValidator``)
3. Из данных отбираются наиболее значимые признаки (``FeatureSelector``)
4. Отобранные признаки проходят предобработку (``DataPreprocessor``)
5. Подготовленные данные используются для обучения моделей (``BaseModel``)
6. Обученные модели упаковываются в контейнер (``ModelContainer``)
7. Для моделей могут быть сгенерированы отчеты (``WordReportGenerator``)

Пример взаимодействия:

.. code-block:: python

    # Загрузка данных
    loader = FileDataLoader()
    df = loader.load_data('data.csv')
    
    # Валидация
    schema = DataSchema(...)
    loader.set_schema(schema)
    df_validated = loader.validate_schema(df)
    
    # Отбор признаков
    selector = RandomForestFeatureSelector(...)
    X_selected = selector.fit_transform(X, y)
    
    # Предобработка
    preprocessor = DefaultPreprocessor(...)
    X_processed = preprocessor.fit_transform(X_selected)
    
    # Обучение модели
    model = RandomForestModel(...)
    model.fit(X_processed, y)
    
    # Создание контейнера
    metadata = ModelMetadata(...)
    container = ModelContainer(model, metadata)
    
    # Сохранение контейнера
    from core.utils.serialization import serialize_container
    with open('model_container.pkl', 'wb') as f:
        f.write(serialize_container(container))

Расширяемость
-----------

Библиотека разработана с учетом возможности расширения. Вы можете:

1. Создавать собственные загрузчики данных, наследуясь от ``DataLoader``
2. Добавлять новые типы валидаторов, наследуясь от ``BaseValidator``
3. Реализовывать новые методы отбора признаков, наследуясь от ``FeatureSelector``
4. Создавать собственные преобразователи данных, наследуясь от ``DataPreprocessor``
5. Создавать собственные модели, наследуясь от ``BaseModel``

Пример создания собственного валидатора:

.. code-block:: python

    from core.validation.base import BaseValidator
    
    class CustomValidator(BaseValidator):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2
        
        def validate(self, data):
            # Реализация собственной логики валидации
            return True 