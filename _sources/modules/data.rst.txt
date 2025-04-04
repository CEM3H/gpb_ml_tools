Модуль data
===========

Модуль для работы с данными, включая загрузку, валидацию и определение схемы данных.

Основные классы
-------------

DataLoader
~~~~~~~~~~

Базовый класс для загрузки данных.

.. autoclass:: core.data.base.DataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

FileDataLoader
~~~~~~~~~~~~~~

Загрузчик данных из файлов.

.. autoclass:: core.data.data_loaders.FileDataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

PostgresDataLoader
~~~~~~~~~~~~~~~~~

Загрузчик данных из PostgreSQL.

.. autoclass:: core.data.data_loaders.PostgresDataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ImpalaDataLoader
~~~~~~~~~~~~~~~

Загрузчик данных из Impala.

.. autoclass:: core.data.data_loaders.ImpalaDataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

SchemaValidator
~~~~~~~~~~~~~~

Валидатор схемы данных.

.. autoclass:: core.data.validation.SchemaValidator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

DataSchema
~~~~~~~~~~

Схема данных.

.. autoclass:: core.data.schema.DataSchema
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ColumnSchema
~~~~~~~~~~~

Схема колонки данных.

.. autoclass:: core.data.schema.ColumnSchema
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Функции работы с типами данных
-----------------------------

.. autofunction:: core.data.data_types.downcast_types
   :noindex:
   
.. autofunction:: core.data.data_types.apply_dtypes_from_schema
   :noindex:

.. autofunction:: core.data.data_types.infer_dtypes
   :noindex:

.. autofunction:: core.data.data_types.get_optimal_numeric_dtype
   :noindex:

.. autofunction:: core.data.data_types.validate_data_against_schema
   :noindex:

Примеры использования
-------------------

Загрузка данных из файла
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.data.data_loaders import FileDataLoader
    from core.data.schema import DataSchema, ColumnSchema

    # Создание схемы данных
    schema = DataSchema({
        'age': ColumnSchema('INTEGER'),
        'name': ColumnSchema('STRING'),
        'income': ColumnSchema('FLOAT')
    })

    # Создание загрузчика
    loader = FileDataLoader('data.csv', schema=schema)

    # Загрузка данных
    data = loader.load_data()

Загрузка данных из PostgreSQL
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.data.data_loaders import PostgresDataLoader
    from core.data.schema import DataSchema, ColumnSchema

    # Создание схемы данных
    schema = DataSchema({
        'age': ColumnSchema('INTEGER'),
        'name': ColumnSchema('STRING'),
        'income': ColumnSchema('FLOAT')
    })

    # Создание загрузчика
    loader = PostgresDataLoader(
        host='localhost',
        port=5432,
        database='mydb',
        user='user',
        password='password',
        table='mytable',
        schema=schema
    )

    # Загрузка данных
    data = loader.load_data()

Загрузка данных из Impala
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.data.data_loaders import ImpalaDataLoader
    from core.data.schema import DataSchema, ColumnSchema

    # Создание схемы данных
    schema = DataSchema({
        'age': ColumnSchema('INTEGER'),
        'name': ColumnSchema('STRING'),
        'income': ColumnSchema('FLOAT')
    })

    # Создание загрузчика
    loader = ImpalaDataLoader(
        host='localhost',
        port=21050,
        database='mydb',
        table='mytable',
        schema=schema
    )

    # Загрузка данных
    data = loader.load_data()

Валидация данных
~~~~~~~~~~~~~~~

.. code-block:: python

    from core.data.validation import SchemaValidator
    from core.data.schema import DataSchema, ColumnSchema

    # Создание схемы данных
    schema = DataSchema({
        'age': ColumnSchema('INTEGER'),
        'name': ColumnSchema('STRING'),
        'income': ColumnSchema('FLOAT')
    })

    # Создание валидатора
    validator = SchemaValidator(schema)

    # Валидация данных
    is_valid = validator.validate(data)
    if not is_valid:
        errors = validator.get_errors()
        print(f"Ошибки валидации: {errors}") 