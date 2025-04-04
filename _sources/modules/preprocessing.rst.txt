Модуль preprocessing
==================

Модуль для предобработки данных.

Основные классы
--------------

DataPreprocessor
~~~~~~~~~~~~~~~

Базовый класс для предобработки данных.

.. autoclass:: core.preprocessing.base.DataPreprocessor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Preprocessors
~~~~~~~~~~~~

Реализации препроцессоров.

.. automodule:: core.preprocessing.preprocessors
   :members:
   :undoc-members:
   :show-inheritance:

Примеры использования
-------------------

Стандартная предобработка данных
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.preprocessing.preprocessors import DefaultPreprocessor

    # Создание препроцессора
    preprocessor = DefaultPreprocessor(
        numeric_strategy='standard',
        categorical_strategy='onehot'
    )

    # Предобработка данных
    X_processed = preprocessor.fit_transform(X)
    print(f"Преобразованные признаки: {preprocessor.get_feature_names()}")

Стандартизация признаков
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.preprocessing.preprocessors import StandardScaler

    # Создание препроцессора
    scaler = StandardScaler()

    # Стандартизация данных
    X_scaled = scaler.fit_transform(X)
    print(f"Средние значения: {scaler.mean_}")
    print(f"Стандартные отклонения: {scaler.scale_}")

Масштабирование в заданный диапазон
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.preprocessing.preprocessors import MinMaxScaler

    # Создание препроцессора
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Масштабирование данных
    X_scaled = scaler.fit_transform(X)
    print(f"Минимальные значения: {scaler.min_}")
    print(f"Максимальные значения: {scaler.max_}")

Масштабирование с учетом выбросов
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.preprocessing.preprocessors import RobustScaler

    # Создание препроцессора
    scaler = RobustScaler()

    # Масштабирование данных
    X_scaled = scaler.fit_transform(X)
    print(f"Медианы: {scaler.center_}")
    print(f"Межквартильные размахи: {scaler.scale_}")

Предобработка категориальных признаков
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.preprocessing.preprocessors import OneHotEncoder

    # Создание препроцессора
    encoder = OneHotEncoder()

    # Кодирование категориальных признаков
    X_encoded = encoder.fit_transform(X)
    print(f"Категории: {encoder.categories_}")
    print(f"Преобразованные признаки: {encoder.get_feature_names()}")

Комбинированная предобработка
---------------------------

.. code-block:: python

    from core.preprocessing.preprocessors import DefaultPreprocessor
    from core.preprocessing.preprocessors import StandardScaler
    from core.preprocessing.preprocessors import OneHotEncoder

    # Создание препроцессоров
    scaler = StandardScaler()
    encoder = OneHotEncoder()

    # Предобработка числовых признаков
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))

    # Предобработка категориальных признаков
    X_categorical = encoder.fit_transform(X.select_dtypes(include=['object']))

    # Объединение результатов
    X_processed = pd.concat([X_numeric, X_categorical], axis=1)
    print(f"Преобразованные признаки: {X_processed.columns.tolist()}")

Обработка пропущенных значений
----------------------------

.. code-block:: python

    from core.preprocessing.preprocessors import SimpleImputer

    # Создание препроцессора
    imputer = SimpleImputer(strategy='mean')

    # Заполнение пропущенных значений
    X_imputed = imputer.fit_transform(X)
    print(f"Заполненные значения: {imputer.statistics_}") 