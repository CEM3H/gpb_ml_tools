"""
Модуль для валидации данных согласно схемам.

Этот модуль предоставляет классы для проверки соответствия данных
заданным схемам, включая проверку типов данных, ограничений на значения
и других требований к данным.

Примеры использования:
    ```python
    # Создание схемы данных
    schema = DataSchema([
        ColumnSchema(
            name='id',
            data_type=DataType.INTEGER,
            required=True,
            constraints={'unique': True}
        ),
        ColumnSchema(
            name='age',
            data_type=DataType.INTEGER,
            required=True,
            constraints={'min': 0, 'max': 150}
        ),
        ColumnSchema(
            name='email',
            data_type=DataType.STRING,
            required=True,
            constraints={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
        ),
        ColumnSchema(
            name='status',
            data_type=DataType.CATEGORICAL,
            required=True,
            categories=['active', 'inactive', 'blocked']
        )
    ])

    # Создание валидатора
    validator = SchemaValidator(schema)

    # Подготовка данных для проверки
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'age': [25, 30, 35],
        'email': ['user1@example.com', 'user2@example.com', 'user3@example.com'],
        'status': ['active', 'inactive', 'active']
    })

    # Проверка данных
    try:
        validator.validate(df)
        print("Данные соответствуют схеме")
    except ValueError as e:
        print(f"Ошибка валидации: {e}")
    ```
"""
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
import re
from .schema import DataSchema, ColumnSchema, DataType

class SchemaValidator:
    """
    Класс для валидации данных согласно схеме.
    
    Проверяет соответствие DataFrame заданной схеме данных, включая:
    - Наличие обязательных колонок
    - Типы данных колонок
    - Ограничения на значения
    - Уникальность значений
    - Отсутствие пропущенных значений
    
    Parameters
    ----------
    schema : DataSchema
        Схема данных для валидации
    
    Examples
    --------
    ```python
    # Создание схемы с различными типами данных и ограничениями
    schema = DataSchema([
        # Целочисленная колонка с ограничениями
        ColumnSchema(
            name='age',
            data_type=DataType.INTEGER,
            required=True,
            constraints={'min': 0, 'max': 150}
        ),
        
        # Строковая колонка с паттерном
        ColumnSchema(
            name='email',
            data_type=DataType.STRING,
            required=True,
            constraints={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
        ),
        
        # Категориальная колонка
        ColumnSchema(
            name='status',
            data_type=DataType.CATEGORICAL,
            required=True,
            categories=['active', 'inactive', 'blocked']
        ),
        
        # Дата и время с ограничениями
        ColumnSchema(
            name='created_at',
            data_type=DataType.DATETIME,
            required=True,
            constraints={
                'min': '2024-01-01',
                'max': '2024-12-31'
            }
        )
    ])

    # Создание валидатора
    validator = SchemaValidator(schema)

    # Примеры данных для проверки
    valid_data = pd.DataFrame({
        'age': [25, 30, 35],
        'email': ['user1@example.com', 'user2@example.com', 'user3@example.com'],
        'status': ['active', 'inactive', 'active'],
        'created_at': ['2024-03-20', '2024-03-21', '2024-03-22']
    })

    invalid_data = pd.DataFrame({
        'age': [25, -5, 35],  # Отрицательный возраст
        'email': ['invalid-email', 'user2@example.com', 'user3@example.com'],  # Неверный формат email
        'status': ['active', 'invalid_status', 'active'],  # Недопустимый статус
        'created_at': ['2024-03-20', '2023-12-31', '2024-03-22']  # Дата вне диапазона
    })

    # Проверка валидных данных
    try:
        validator.validate(valid_data)
        print("Данные соответствуют схеме")
    except ValueError as e:
        print(f"Ошибка валидации: {e}")

    # Проверка невалидных данных
    try:
        validator.validate(invalid_data)
    except ValueError as e:
        print(f"Ошибка валидации: {e}")
    ```
    """
    
    def __init__(self, schema: DataSchema):
        """
        Инициализация валидатора.
        
        Args:
            schema: Схема данных для валидации
        """
        self.schema = schema
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Проверяет DataFrame на соответствие схеме.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame для проверки
        
        Returns
        -------
        bool
            True, если данные соответствуют схеме
        
        Raises
        ------
        ValueError
            Если данные не соответствуют схеме
        """
        # Проверяем наличие всех обязательных колонок
        required_columns = [col.name for col in self.schema.columns if col.required]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {', '.join(missing_columns)}")
        
        # Проверяем каждую колонку
        for col_schema in self.schema.columns:
            if col_schema.name in df.columns:
                self._validate_column(df[col_schema.name], col_schema)
        
        return True
    
    def _validate_column(self, series: pd.Series, schema: ColumnSchema) -> None:
        """
        Проверяет колонку на соответствие схеме.
        
        Parameters
        ----------
        series : pd.Series
            Серия данных для проверки
        schema : ColumnSchema
            Схема колонки
        
        Raises
        ------
        ValueError
            Если данные не соответствуют схеме
        """
        # Проверяем null значения
        if schema.required and series.isna().any():
            raise ValueError(f"Колонка {schema.name} не должна содержать пропущенные значения")
        
        # Проверяем тип данных
        if schema.data_type == DataType.INTEGER:
            self._validate_integer(series, schema)
        elif schema.data_type == DataType.FLOAT:
            self._validate_float(series, schema)
        elif schema.data_type == DataType.STRING:
            self._validate_string(series, schema)
        elif schema.data_type == DataType.DATETIME:
            self._validate_datetime(series, schema)
        elif schema.data_type == DataType.BOOLEAN:
            self._validate_boolean(series, schema)
        elif schema.data_type == DataType.CATEGORICAL:
            self._validate_categorical(series, schema)
    
    def _validate_integer(self, series: pd.Series, schema: ColumnSchema) -> None:
        """
        Проверяет целочисленные данные.
        
        Parameters
        ----------
        series : pd.Series
            Серия данных для проверки
        schema : ColumnSchema
            Схема колонки
        
        Raises
        ------
        ValueError
            Если данные не соответствуют требованиям
        """
        if not pd.api.types.is_integer_dtype(series):
            raise ValueError(f"Колонка {schema.name} должна содержать целые числа")
        
        if schema.constraints:
            if 'min' in schema.constraints and series.min() < schema.constraints['min']:
                raise ValueError(f"Колонка {schema.name} содержит значения меньше {schema.constraints['min']}")
            if 'max' in schema.constraints and series.max() > schema.constraints['max']:
                raise ValueError(f"Колонка {schema.name} содержит значения больше {schema.constraints['max']}")
            if schema.constraints.get('unique', False) and series.duplicated().any():
                raise ValueError(f"Колонка {schema.name} должна содержать уникальные значения")
    
    def _validate_float(self, series: pd.Series, schema: ColumnSchema) -> None:
        """
        Проверяет числовые данные с плавающей точкой.
        
        Parameters
        ----------
        series : pd.Series
            Серия данных для проверки
        schema : ColumnSchema
            Схема колонки
        
        Raises
        ------
        ValueError
            Если данные не соответствуют требованиям
        """
        if not pd.api.types.is_float_dtype(series):
            raise ValueError(f"Колонка {schema.name} должна содержать числа с плавающей точкой")
        
        if schema.constraints:
            if 'min' in schema.constraints and series.min() < schema.constraints['min']:
                raise ValueError(f"Колонка {schema.name} содержит значения меньше {schema.constraints['min']}")
            if 'max' in schema.constraints and series.max() > schema.constraints['max']:
                raise ValueError(f"Колонка {schema.name} содержит значения больше {schema.constraints['max']}")
    
    def _validate_string(self, series: pd.Series, schema: ColumnSchema) -> None:
        """
        Проверяет строковые данные.
        
        Parameters
        ----------
        series : pd.Series
            Серия данных для проверки
        schema : ColumnSchema
            Схема колонки
        
        Raises
        ------
        ValueError
            Если данные не соответствуют требованиям
        """
        if not pd.api.types.is_string_dtype(series):
            raise ValueError(f"Колонка {schema.name} должна содержать строки")
        
        if schema.constraints and 'pattern' in schema.constraints:
            pattern = schema.constraints['pattern']
            if not series.str.match(pattern).all():
                raise ValueError(f"Колонка {schema.name} содержит значения, не соответствующие паттерну {pattern}")
    
    def _validate_datetime(self, series: pd.Series, schema: ColumnSchema) -> None:
        """
        Проверяет данные даты и времени.
        
        Parameters
        ----------
        series : pd.Series
            Серия данных для проверки
        schema : ColumnSchema
            Схема колонки
        
        Raises
        ------
        ValueError
            Если данные не соответствуют требованиям
        """
        if not pd.api.types.is_datetime64_any_dtype(series):
            raise ValueError(f"Колонка {schema.name} должна содержать даты и время")
        
        if schema.constraints:
            if 'min' in schema.constraints and series.min() < pd.Timestamp(schema.constraints['min']):
                raise ValueError(f"Колонка {schema.name} содержит даты раньше {schema.constraints['min']}")
            if 'max' in schema.constraints and series.max() > pd.Timestamp(schema.constraints['max']):
                raise ValueError(f"Колонка {schema.name} содержит даты позже {schema.constraints['max']}")
    
    def _validate_boolean(self, series: pd.Series, schema: ColumnSchema) -> None:
        """
        Проверяет булевы данные.
        
        Parameters
        ----------
        series : pd.Series
            Серия данных для проверки
        schema : ColumnSchema
            Схема колонки
        
        Raises
        ------
        ValueError
            Если данные не соответствуют требованиям
        """
        if not pd.api.types.is_bool_dtype(series):
            raise ValueError(f"Колонка {schema.name} должна содержать булевы значения")
    
    def _validate_categorical(self, series: pd.Series, schema: ColumnSchema) -> None:
        """
        Проверяет категориальные данные.
        
        Parameters
        ----------
        series : pd.Series
            Серия данных для проверки
        schema : ColumnSchema
            Схема колонки
        
        Raises
        ------
        ValueError
            Если данные не соответствуют требованиям
        """
        if not isinstance(series.dtype, pd.CategoricalDtype):
            raise ValueError(f"Колонка {schema.name} должна быть категориальной")
        
        if schema.categories:
            invalid_categories = set(series.cat.categories) - set(schema.categories)
            if invalid_categories:
                raise ValueError(f"Колонка {schema.name} содержит недопустимые категории: {', '.join(invalid_categories)}")
        
        if schema.constraints and schema.constraints.get('unique', False):
            if series.nunique() != len(series):
                raise ValueError(f"Колонка {schema.name} должна содержать уникальные значения") 