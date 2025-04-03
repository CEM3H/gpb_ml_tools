"""
Тесты для проверки интеграции схем данных с реальными файлами схем.
"""

import pytest
import pandas as pd
import numpy as np
import os

from core.data.schema import DataSchema


def test_example_schema_file():
    """Тест работы с файлом example_schema.json."""
    # Проверяем наличие файла
    schema_path = 'schemas/example_schema.json'
    assert os.path.exists(schema_path), "Файл example_schema.json не найден"
    
    # Загружаем схему из файла
    schema = DataSchema.from_file(schema_path)
    
    # Проверяем, что схема не пустая
    assert len(schema.schema) > 0, "Схема пуста"
    
    # Проверяем некоторые ожидаемые колонки из реального файла example_schema.json
    expected_columns = ['customer_id', 'full_name', 'birth_date', 'salary', 'credit_score']
    for column in expected_columns:
        assert column in schema.schema, f"Ожидаемая колонка {column} отсутствует в схеме"
    
    # Проверяем, что колонка 'customer_id' имеет правильный тип
    assert schema.schema.get('customer_id', {}).get('dtype') in ['int', 'int64', 'int32'], \
        "Тип колонки customer_id должен быть целочисленным"


def test_generate_df_from_example_schema():
    """Тест генерации тестового DataFrame на основе example_schema.json."""
    # Загружаем схему
    schema_path = 'schemas/example_schema.json'
    schema = DataSchema.from_file(schema_path)
    
    # Создаем тестовый DataFrame с данными, соответствующими схеме
    data = {}
    for column, info in schema.schema.items():
        # Генерируем тестовые данные в соответствии с типом
        dtype = info.get('dtype')
        if dtype in ['int', 'int64', 'int32']:
            data[column] = [i for i in range(1, 6)]
        elif dtype in ['float', 'float64']:
            data[column] = [float(i) * 1.5 for i in range(1, 6)]
        elif dtype in ['bool', 'boolean']:
            data[column] = [i % 2 == 0 for i in range(5)]
        elif dtype == 'category':
            data[column] = ['A', 'B', 'C', 'A', 'B']
        else:  # str или object
            data[column] = [f"Value {i}" for i in range(1, 6)]
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Применяем схему к DataFrame
    df_typed = schema.apply_to_dataframe(df)
    
    # Проверяем, что все колонки имеют правильный тип
    for column, info in schema.schema.items():
        dtype = info.get('dtype')
        if dtype in ['int', 'int64', 'int32']:
            assert pd.api.types.is_integer_dtype(df_typed[column]), \
                f"Колонка {column} должна быть целочисленной"
        elif dtype in ['float', 'float64']:
            assert pd.api.types.is_float_dtype(df_typed[column]), \
                f"Колонка {column} должна быть float"
        elif dtype in ['bool', 'boolean']:
            assert pd.api.types.is_bool_dtype(df_typed[column]), \
                f"Колонка {column} должна быть boolean"
        elif dtype == 'category':
            assert isinstance(df_typed[column].dtype, pd.CategoricalDtype), \
                f"Колонка {column} должна быть категориальной"


def test_schema_summary():
    """Тест генерации сводки по схеме example_schema.json."""
    # Загружаем схему
    schema_path = 'schemas/example_schema.json'
    schema = DataSchema.from_file(schema_path)
    
    # Получаем сводку
    summary = schema.summary()
    
    # Проверяем структуру сводки
    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns).issuperset({'column', 'dtype', 'description', 'required'})
    
    # Проверяем, что все колонки схемы отражены в сводке
    assert len(summary) == len(schema.schema)
    for column in schema.schema:
        assert column in summary['column'].values 