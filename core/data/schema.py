"""
Модуль для работы со схемами данных.

Этот модуль предоставляет классы для описания и валидации схем данных.
Схема данных определяет структуру DataFrame, включая типы данных колонок,
ограничения на значения и метаданные.

Примеры использования:
    ```python
    # Создание схемы с описанием колонок
    schema = DataSchema([
        ColumnSchema(
            name='id',
            data_type=DataType.INTEGER,
            required=True,
            description='Уникальный идентификатор записи',
            constraints={'unique': True}
        ),
        ColumnSchema(
            name='age',
            data_type=DataType.INTEGER,
            required=True,
            description='Возраст пользователя',
            constraints={'min': 0, 'max': 150}
        ),
        ColumnSchema(
            name='email',
            data_type=DataType.STRING,
            required=True,
            description='Email пользователя',
            constraints={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
        ),
        ColumnSchema(
            name='status',
            data_type=DataType.CATEGORICAL,
            required=True,
            description='Статус пользователя',
            categories=['active', 'inactive', 'blocked']
        )
    ])

    # Загрузка схемы из файла
    schema = DataSchema.from_file('schemas/users.json')

    # Создание схемы на основе DataFrame
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'age': [25, 30, 35],
        'email': ['user1@example.com', 'user2@example.com', 'user3@example.com'],
        'status': ['active', 'inactive', 'active']
    })
    schema = DataSchema.from_dataframe(df)

    # Сохранение схемы в файл
    schema.save('schemas/users.json')

    # Применение схемы к DataFrame
    df = schema.apply_to_dataframe(df)
    ```
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import json
import os
import yaml
from pathlib import Path
from enum import Enum

from .data_types import infer_dtypes, apply_dtypes_from_schema


class DataType(Enum):
    """
    Перечисление поддерживаемых типов данных для колонок.
    
    Attributes
    ----------
    INTEGER : str
        Целочисленный тип данных. Примеры: 1, 42, -100
    FLOAT : str
        Числа с плавающей точкой. Примеры: 1.5, 3.14, -0.001
    STRING : str
        Строковый тип данных. Примеры: "Hello", "123", "user@example.com"
    DATETIME : str
        Дата и время. Примеры: "2024-03-20", "2024-03-20 15:30:00"
    BOOLEAN : str
        Логический тип данных. Примеры: True, False
    CATEGORICAL : str
        Категориальный тип данных. Примеры: ["active", "inactive", "blocked"]
    """
    INTEGER = 'integer'
    FLOAT = 'float'
    STRING = 'string'
    DATETIME = 'datetime'
    BOOLEAN = 'boolean'
    CATEGORICAL = 'categorical'


class ColumnSchema:
    """
    Класс для описания схемы отдельной колонки данных.
    
    Parameters
    ----------
    name : str
        Имя колонки
    data_type : DataType
        Тип данных колонки
    required : bool, optional
        Флаг обязательности колонки, по умолчанию False
    description : str, optional
        Описание колонки, по умолчанию пустая строка
    categories : List[str], optional
        Список допустимых категорий для категориальных данных
    constraints : Dict[str, Any], optional
        Словарь с ограничениями на значения. Поддерживаемые ограничения:
        - unique: bool - значения должны быть уникальными
        - min: float/int - минимальное значение
        - max: float/int - максимальное значение
        - pattern: str - регулярное выражение для строковых значений
    
    Examples
    --------
    ```python
    # Создание схемы для числовой колонки
    age_schema = ColumnSchema(
        name='age',
        data_type=DataType.INTEGER,
        required=True,
        description='Возраст пользователя',
        constraints={'min': 0, 'max': 150}
    )

    # Создание схемы для строковой колонки с паттерном
    email_schema = ColumnSchema(
        name='email',
        data_type=DataType.STRING,
        required=True,
        description='Email пользователя',
        constraints={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
    )

    # Создание схемы для категориальной колонки
    status_schema = ColumnSchema(
        name='status',
        data_type=DataType.CATEGORICAL,
        required=True,
        description='Статус пользователя',
        categories=['active', 'inactive', 'blocked']
    )
    ```
    """
    
    def __init__(
        self,
        name: str,
        data_type: DataType,
        required: bool = False,
        description: str = '',
        categories: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.data_type = data_type
        self.required = required
        self.description = description
        self.categories = categories or []
        self.constraints = constraints or {}


class DataSchema:
    """
    Класс для работы со схемами данных.
    
    Схема данных описывает структуру DataFrame и содержит:
    - Типы данных колонок
    - Описания колонок
    - Флаги обязательности
    - Ограничения на значения
    - Метаданные
    
    Parameters
    ----------
    columns : List[ColumnSchema], optional
        Список схем колонок. Если None, создается пустая схема.
    
    Examples
    --------
    ```python
    # Создание схемы с описанием колонок
    schema = DataSchema([
        ColumnSchema(
            name='id',
            data_type=DataType.INTEGER,
            required=True,
            description='Уникальный идентификатор записи',
            constraints={'unique': True}
        ),
        ColumnSchema(
            name='age',
            data_type=DataType.INTEGER,
            required=True,
            description='Возраст пользователя',
            constraints={'min': 0, 'max': 150}
        ),
        ColumnSchema(
            name='email',
            data_type=DataType.STRING,
            required=True,
            description='Email пользователя',
            constraints={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
        ),
        ColumnSchema(
            name='status',
            data_type=DataType.CATEGORICAL,
            required=True,
            description='Статус пользователя',
            categories=['active', 'inactive', 'blocked']
        )
    ])

    # Загрузка схемы из JSON файла
    schema = DataSchema.from_file('schemas/users.json')

    # Создание схемы на основе DataFrame
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'age': [25, 30, 35],
        'email': ['user1@example.com', 'user2@example.com', 'user3@example.com'],
        'status': ['active', 'inactive', 'active']
    })
    schema = DataSchema.from_dataframe(df)

    # Сохранение схемы в YAML файл
    schema.save('schemas/users.yaml')

    # Применение схемы к DataFrame
    df = schema.apply_to_dataframe(df)

    # Получение информации о колонке
    age_info = schema.get_column_info('age')

    # Добавление новой колонки
    schema.add_column(
        name='created_at',
        data_type=DataType.DATETIME,
        required=True,
        description='Дата создания записи'
    )

    # Обновление информации о колонке
    schema.update_column('age', required=False)

    # Удаление колонки
    schema.remove_column('created_at')

    # Получение списка числовых колонок
    numeric_columns = schema.get_columns_by_type(DataType.INTEGER)

    # Получение сводной информации о схеме
    summary_df = schema.summary()
    ```
    """
    
    def __init__(self, columns: Optional[List[ColumnSchema]] = None):
        self.columns = columns or []
    
    @classmethod
    def from_file(cls, file_path: str) -> 'DataSchema':
        """
        Загружает схему из файла JSON или YAML.
        
        Parameters
        ----------
        file_path : str
            Путь к файлу схемы
        
        Returns
        -------
        DataSchema
            Загруженная схема данных
        
        Raises
        ------
        ValueError
            Если формат файла не поддерживается
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                data = json.load(f)
            elif ext in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {ext}")
        
        columns = []
        for col_data in data.get('columns', []):
            col_schema = ColumnSchema(
                name=col_data['name'],
                data_type=DataType(col_data['data_type']),
                required=col_data.get('required', False),
                description=col_data.get('description', ''),
                categories=col_data.get('categories'),
                constraints=col_data.get('constraints')
            )
            columns.append(col_schema)
        
        return cls(columns=columns)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, sample_size: int = 1000) -> 'DataSchema':
        """
        Создает схему на основе анализа DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame для анализа
        sample_size : int, optional
            Размер выборки для анализа, по умолчанию 1000
        
        Returns
        -------
        DataSchema
            Созданная схема данных
        """
        columns = []
        
        for col in df.columns:
            # Определяем тип данных
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                data_type = DataType.INTEGER
            elif pd.api.types.is_float_dtype(dtype):
                data_type = DataType.FLOAT
            elif pd.api.types.is_string_dtype(dtype):
                data_type = DataType.STRING
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                data_type = DataType.DATETIME
            elif pd.api.types.is_bool_dtype(dtype):
                data_type = DataType.BOOLEAN
            elif pd.api.types.is_categorical_dtype(dtype):
                data_type = DataType.CATEGORICAL
            else:
                data_type = DataType.STRING
            
            # Создаем схему колонки
            col_schema = ColumnSchema(
                name=col,
                data_type=data_type,
                required=df[col].isna().sum() == 0,
                description=f"Колонка {col}",
                categories=df[col].unique().tolist() if data_type == DataType.CATEGORICAL else None,
                constraints={
                    'min': float(df[col].min()) if pd.api.types.is_numeric_dtype(dtype) else None,
                    'max': float(df[col].max()) if pd.api.types.is_numeric_dtype(dtype) else None,
                    'unique': df[col].nunique() == len(df)
                }
            )
            columns.append(col_schema)
        
        return cls(columns=columns)
    
    def save(self, file_path: str) -> None:
        """
        Сохраняет схему в файл JSON или YAML.
        
        Parameters
        ----------
        file_path : str
            Путь к файлу для сохранения
        
        Raises
        ------
        ValueError
            Если формат файла не поддерживается
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        data = {
            'columns': [
                {
                    'name': col.name,
                    'data_type': col.data_type.value,
                    'required': col.required,
                    'description': col.description,
                    'categories': col.categories,
                    'constraints': col.constraints
                }
                for col in self.columns
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if ext == '.json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif ext in ['.yml', '.yaml']:
                yaml.dump(data, f, allow_unicode=True)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {ext}")
    
    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет схему к DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame для преобразования
        
        Returns
        -------
        pd.DataFrame
            DataFrame с примененной схемой
        """
        for col_schema in self.columns:
            if col_schema.name in df.columns:
                df[col_schema.name] = self._convert_column_type(
                    df[col_schema.name],
                    col_schema.data_type
                )
        return df
    
    def _convert_column_type(self, series: pd.Series, data_type: DataType) -> pd.Series:
        """
        Преобразует тип данных колонки.
        
        Parameters
        ----------
        series : pd.Series
            Серия данных для преобразования
        data_type : DataType
            Целевой тип данных
        
        Returns
        -------
        pd.Series
            Серия с преобразованным типом данных
        """
        if data_type == DataType.INTEGER:
            return pd.to_numeric(series, errors='coerce').astype('Int64')
        elif data_type == DataType.FLOAT:
            return pd.to_numeric(series, errors='coerce')
        elif data_type == DataType.STRING:
            return series.astype(str)
        elif data_type == DataType.DATETIME:
            return pd.to_datetime(series, errors='coerce')
        elif data_type == DataType.BOOLEAN:
            return series.astype(bool)
        elif data_type == DataType.CATEGORICAL:
            return series.astype('category')
        return series
    
    def get_column_info(self, column: str) -> Optional[ColumnSchema]:
        """
        Получает информацию о колонке.
        
        Parameters
        ----------
        column : str
            Имя колонки
        
        Returns
        -------
        Optional[ColumnSchema]
            Схема колонки или None, если колонка не найдена
        """
        for col in self.columns:
            if col.name == column:
                return col
        return None
    
    def add_column(self, column: str, data_type: DataType, required: bool = False,
                  description: str = '', categories: Optional[List[str]] = None,
                  constraints: Optional[Dict[str, Any]] = None) -> None:
        """
        Добавляет колонку в схему.
        
        Parameters
        ----------
        column : str
            Имя колонки
        data_type : DataType
            Тип данных
        required : bool, optional
            Флаг обязательности, по умолчанию False
        description : str, optional
            Описание колонки, по умолчанию пустая строка
        categories : List[str], optional
            Список категорий
        constraints : Dict[str, Any], optional
            Ограничения на значения
        """
        col_schema = ColumnSchema(
            name=column,
            data_type=data_type,
            required=required,
            description=description,
            categories=categories,
            constraints=constraints
        )
        self.columns.append(col_schema)
    
    def update_column(self, column: str, **kwargs) -> None:
        """
        Обновляет информацию о колонке.
        
        Parameters
        ----------
        column : str
            Имя колонки
        **kwargs
            Параметры для обновления
        """
        for col in self.columns:
            if col.name == column:
                for key, value in kwargs.items():
                    setattr(col, key, value)
                break
    
    def remove_column(self, column: str) -> None:
        """
        Удаляет колонку из схемы.
        
        Parameters
        ----------
        column : str
            Имя колонки
        """
        self.columns = [col for col in self.columns if col.name != column]
    
    def get_columns_by_type(self, data_type: DataType) -> List[str]:
        """
        Получает список колонок заданного типа.
        
        Parameters
        ----------
        data_type : DataType
            Тип данных
        
        Returns
        -------
        List[str]
            Список имен колонок
        """
        return [col.name for col in self.columns if col.data_type == data_type]
    
    def __str__(self) -> str:
        """
        Возвращает строковое представление схемы.
        
        Returns
        -------
        str
            Строковое представление схемы
        """
        return f"DataSchema(columns={[col.name for col in self.columns]})"
    
    def summary(self) -> pd.DataFrame:
        """
        Возвращает сводную информацию о схеме.
        
        Returns
        -------
        pd.DataFrame
            DataFrame с информацией о колонках
        """
        data = []
        for col in self.columns:
            data.append({
                'name': col.name,
                'data_type': col.data_type.value,
                'required': col.required,
                'description': col.description,
                'categories': col.categories,
                'constraints': col.constraints
            })
        return pd.DataFrame(data) 