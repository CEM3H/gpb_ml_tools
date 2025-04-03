from typing import Dict, List, Optional, Union, Any
import pandas as pd
import os
from pathlib import Path
import numpy as np
from datetime import datetime

from .schema import DataSchema, ColumnSchema, DataType
from .data_types import downcast_types
from .validation import SchemaValidator

class DataLoader:
    """
    Базовый класс для загрузки данных из различных источников с поддержкой схем данных.
    
    Этот класс предоставляет общий интерфейс для загрузки данных из разных источников,
    с возможностью валидации и оптимизации типов данных на основе схемы.
    
    Схема данных - это справочник, который содержит информацию о колонках:
    - Тип данных для каждой колонки
    - Описание колонки
    - Дополнительные атрибуты (обязательность, примеры значений и т.д.)
    
    Примеры:
        Создание загрузчика данных:
        ```
        # Создание загрузчика данных из файла
        file_loader = FileDataLoader({
            'file_path': 'data.csv',
            'file_type': 'csv'
        })
        
        # Создание загрузчика данных из PostgreSQL
        pg_loader = PostgresDataLoader({
            'host': 'localhost',
            'port': 5432,
            'database': 'my_db',
            'user': 'user',
            'password': 'password'
        })
        ```
        
        Загрузка данных с использованием схемы:
        ```
        # Установка схемы данных
        loader.set_schema('schemas/my_schema.json')
        
        # Загрузка данных (типы данных будут оптимизированы согласно схеме)
        df = loader.load_data("SELECT * FROM my_table")
        ```
        
        Создание схемы данных:
        ```
        # Загрузка данных без схемы
        df = loader.load_data("SELECT * FROM my_table")
        
        # Создание схемы на основе загруженных данных
        schema = loader.create_schema('schemas/my_schema.json')
        ```
    """
    
    def __init__(self, schema: Optional[DataSchema] = None):
        """
        Инициализация загрузчика данных.
        
        Args:
            schema: Схема данных (опционально)
        """
        self.schema = schema
        self._validator = SchemaValidator(schema) if schema else None
        self.optimize_types = True
        
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Загружает данные.
        
        Args:
            **kwargs: Параметры загрузки
            
        Returns:
            pd.DataFrame: Загруженные данные
        """
        raise NotImplementedError("Метод load_data должен быть реализован в дочернем классе")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Проверяет соответствие данных схеме.
        
        Args:
            df: DataFrame для валидации
            
        Returns:
            bool: True если данные соответствуют схеме
        """
        if self._validator:
            return self._validator.validate(df)
        return True
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует типы данных согласно схеме.
        
        Args:
            df: DataFrame для преобразования
            
        Returns:
            pd.DataFrame: DataFrame с преобразованными типами
        """
        if not self.schema:
            return df
            
        for col_schema in self.schema.columns:
            if col_schema.name in df.columns:
                df[col_schema.name] = self._convert_column_type(
                    df[col_schema.name], 
                    col_schema.data_type
                )
        
        return df
    
    def _convert_column_type(self, series: pd.Series, data_type: DataType) -> pd.Series:
        """
        Преобразует тип данных колонки.
        
        Args:
            series: Серия данных для преобразования
            data_type: Целевой тип данных
            
        Returns:
            pd.Series: Серия с преобразованным типом данных
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
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Проверка схемы загруженных данных.
        
        Проверяет соответствие DataFrame заданной схеме данных. В базовой реализации
        проверяет наличие всех обязательных колонок.
        
        Args:
            df: DataFrame для проверки
            
        Returns:
            True, если схема валидна, иначе вызывает исключение ValueError
            
        Raises:
            ValueError: Если отсутствуют обязательные колонки
            
        Примеры:
            ```
            # Проверка схемы данных
            try:
                loader.validate_schema(df)
                print("Схема валидна")
            except ValueError as e:
                print(f"Ошибка валидации: {str(e)}")
            ```
        """
        if self.schema is None:
            return True
        
        # Проверяем наличие всех обязательных колонок
        required_columns = [col for col, info in self.schema.schema.items() 
                            if info.get('required', False)]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        return True
    
    def set_schema(self, schema: Union[DataSchema, str]) -> None:
        """
        Установка схемы данных для загрузчика.
        
        Схема данных используется для валидации и оптимизации типов данных при загрузке.
        
        Args:
            schema: Объект схемы данных (DataSchema) или путь к файлу схемы (JSON или YAML)
            
        Примеры:
            ```
            # Установка схемы из файла
            loader.set_schema('schemas/my_schema.json')
            
            # Установка схемы из объекта
            schema = DataSchema.from_dataframe(df)
            loader.set_schema(schema)
            ```
        """
        if isinstance(schema, str):
            # Если передан путь к файлу, загружаем схему из файла
            self.schema = DataSchema.from_file(schema)
        else:
            # Иначе используем переданную схему
            self.schema = schema
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Оптимизация типов данных DataFrame.
        
        Оптимизирует типы данных в DataFrame для уменьшения использования памяти,
        используя схему данных или автоматическое определение типов.
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            DataFrame с оптимизированными типами данных
            
        Примеры:
            ```
            # Оптимизация типов данных
            df_optimized = loader._optimize_dataframe(df)
            ```
        """
        if not self.optimize_types:
            return df
            
        if self.schema is not None:
            # Если есть схема, применяем типы из неё
            return self.schema.apply_to_dataframe(df, self.optimize_types)
        else:
            # Иначе пытаемся автоматически оптимизировать типы
            return downcast_types(df, self.optimize_types)
    
    def get_schema_path(self, name: str) -> str:
        """
        Получение пути к файлу схемы по имени.
        
        Args:
            name: Имя схемы (обычно имя таблицы)
            
        Returns:
            Путь к файлу схемы в директории schemas
            
        Примеры:
            ```
            # Получение пути к файлу схемы
            schema_path = loader.get_schema_path('customers')
            print(f"Путь к файлу схемы: {schema_path}")
            ```
        """
        # Директория для хранения схем - schemas внутри текущей рабочей директории
        schema_dir = os.path.join(os.getcwd(), 'schemas')
        os.makedirs(schema_dir, exist_ok=True)
        
        # Путь к файлу схемы
        return os.path.join(schema_dir, f"{name}.json") 