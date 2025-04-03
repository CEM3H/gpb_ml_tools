Модуль validation
================

Модуль для валидации данных.

Основные классы
--------------

Validators
~~~~~~~~~~

Реализации валидаторов.

.. automodule:: core.validation.validators
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Примеры использования
-------------------

Валидация числовых данных
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.validation.validators import NumericValidator

    # Создание валидатора
    validator = NumericValidator(
        min_value=0,
        max_value=100,
        allow_nan=False
    )

    # Валидация данных
    is_valid = validator.validate(data['age'])
    if not is_valid:
        errors = validator.get_errors()
        print(f"Ошибки валидации: {errors}")

Валидация строковых данных
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.validation.validators import StringValidator

    # Создание валидатора
    validator = StringValidator(
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z]+$'
    )

    # Валидация данных
    is_valid = validator.validate(data['name'])
    if not is_valid:
        errors = validator.get_errors()
        print(f"Ошибки валидации: {errors}")

Валидация дат и времени
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.validation.validators import DateTimeValidator

    # Создание валидатора
    validator = DateTimeValidator(
        format='%Y-%m-%d',
        min_date='2000-01-01',
        max_date='2024-12-31'
    )

    # Валидация данных
    is_valid = validator.validate(data['date'])
    if not is_valid:
        errors = validator.get_errors()
        print(f"Ошибки валидации: {errors}") 