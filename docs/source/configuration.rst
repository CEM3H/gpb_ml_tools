Конфигурация и настройка
======================

Общие настройки
-------------

Библиотека GPB может быть настроена с помощью конфигурационного файла ``config.yaml``:

.. code-block:: yaml

    # Настройки логирования
    logging:
      level: INFO
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
      file: "gpb.log"
    
    # Настройки кэширования
    cache:
      enabled: true
      directory: ".cache"
      ttl: 3600
    
    # Настройки параллельной обработки
    parallel:
      enabled: true
      max_workers: 4
    
    # Настройки валидации
    validation:
      strict_mode: true
      max_errors: 100
    
    # Настройки отбора признаков
    feature_selection:
      min_features: 1
      max_features: 100
      correlation_threshold: 0.8
    
    # Настройки предобработки
    preprocessing:
      handle_missing: true
      handle_outliers: true
      outlier_threshold: 3.0

Настройки загрузчиков данных
--------------------------

FileDataLoader
~~~~~~~~~~~~

.. code-block:: yaml

    data_loading:
      file:
        default_encoding: utf-8
        default_separator: ","
        default_decimal: "."
        default_thousands: ","
        chunk_size: 10000
        max_rows: null

PostgresDataLoader
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    data_loading:
      postgres:
        connection_timeout: 30
        statement_timeout: 3600
        pool_size: 5
        max_overflow: 10
        pool_timeout: 30
        pool_recycle: 3600

ImpalaDataLoader
~~~~~~~~~~~~~

.. code-block:: yaml

    data_loading:
      impala:
        connection_timeout: 30
        socket_timeout: 3600
        use_ssl: false
        ca_cert: null
        auth_mechanism: PLAIN
        krb_service_name: impala

Настройки валидации
----------------

SchemaValidator
~~~~~~~~~~~~

.. code-block:: yaml

    validation:
      schema:
        strict_mode: true
        max_errors: 100
        skip_unknown_columns: false
        handle_missing_values: true
        handle_duplicates: true

DateTimeSchemaValidator
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    validation:
      datetime:
        default_format: "%Y-%m-%d %H:%M:%S"
        timezone: "UTC"
        handle_invalid_dates: true
        handle_missing_dates: true

Настройки отбора признаков
-----------------------

StatisticalFeatureSelector
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    feature_selection:
      statistical:
        variance_threshold: 0.1
        correlation_threshold: 0.8
        mutual_info_threshold: 0.0
        handle_missing_values: true

ModelBasedFeatureSelector
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    feature_selection:
      model_based:
        n_features_to_select: null
        step: 1
        importance_threshold: 0.01
        handle_missing_values: true

Настройки предобработки
--------------------

NumericPreprocessor
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    preprocessing:
      numeric:
        handle_missing_values: true
        handle_outliers: true
        outlier_threshold: 3.0
        scale_features: true
        robust_scaling: false

CategoricalPreprocessor
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    preprocessing:
      categorical:
        handle_missing_values: true
        handle_unknown: "error"
        min_frequency: 0.01
        max_categories: 100

Использование конфигурации
-----------------------

Загрузка конфигурации
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.config import load_config
    
    # Загрузка конфигурации из файла
    config = load_config('config.yaml')
    
    # Использование конфигурации
    loader = FileDataLoader(config=config['data_loading']['file'])
    validator = SchemaValidator(config=config['validation']['schema'])

Переопределение настроек
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.config import load_config
    
    # Загрузка базовой конфигурации
    config = load_config('config.yaml')
    
    # Переопределение настроек
    config['data_loading']['file']['default_encoding'] = 'cp1251'
    config['validation']['schema']['strict_mode'] = False
    
    # Использование обновленной конфигурации
    loader = FileDataLoader(config=config['data_loading']['file'])
    validator = SchemaValidator(config=config['validation']['schema'])

Программная настройка
------------------

Вы также можете настроить библиотеку программно:

.. code-block:: python

    from core.config import Config
    
    # Создание конфигурации
    config = Config()
    
    # Настройка логирования
    config.logging.level = 'DEBUG'
    config.logging.format = '%(asctime)s - %(message)s'
    
    # Настройка кэширования
    config.cache.enabled = True
    config.cache.directory = '.cache'
    config.cache.ttl = 3600
    
    # Настройка параллельной обработки
    config.parallel.enabled = True
    config.parallel.max_workers = 4
    
    # Применение конфигурации
    config.apply() 