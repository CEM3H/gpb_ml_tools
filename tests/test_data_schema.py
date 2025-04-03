"""
Тесты для модуля схем данных (DataSchema).
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
import tempfile
from pathlib import Path

from core.data.schema import DataSchema


@pytest.fixture
def sample_dataframe():
    """Создаёт тестовый DataFrame с разными типами данных для тестирования."""
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'age': [25, 30, 35, 40, 45],
        'is_active': [True, False, True, True, False],
        'score': [95.5, 80.0, 85.5, 90.0, 75.5],
        'category': ['A', 'B', 'A', 'C', 'B']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_schema_dict():
    """Создаёт словарь схемы данных для тестирования."""
    return {
        'id': {
            'dtype': 'int',
            'description': 'Уникальный идентификатор',
            'required': True
        },
        'name': {
            'dtype': 'category',
            'description': 'Имя пользователя',
            'required': True
        },
        'age': {
            'dtype': 'int',
            'description': 'Возраст пользователя',
            'required': False
        },
        'is_active': {
            'dtype': 'bool',
            'description': 'Активный пользователь или нет',
            'required': False
        },
        'score': {
            'dtype': 'float',
            'description': 'Оценка пользователя',
            'required': False
        },
        'category': {
            'dtype': 'category',
            'description': 'Категория пользователя',
            'required': False
        }
    }


def test_schema_init():
    """Тест инициализации схемы."""
    # Создание пустой схемы
    empty_schema = DataSchema()
    assert empty_schema.schema == {}
    
    # Создание схемы с заданными колонками
    schema_dict = {
        'id': {'dtype': 'int', 'description': 'ID'},
        'name': {'dtype': 'str', 'description': 'Name'}
    }
    schema = DataSchema(schema_dict)
    assert schema.schema == schema_dict


def test_from_dataframe(sample_dataframe):
    """Тест создания схемы из DataFrame."""
    schema = DataSchema.from_dataframe(sample_dataframe)
    
    # Проверяем, что схема создана для всех колонок
    for column in sample_dataframe.columns:
        assert column in schema.schema
    
    # Проверяем, что типы данных определены корректно
    assert schema.schema['id'].get('dtype') in ['int', 'int64', 'int32']
    assert schema.schema['name'].get('dtype') in ['category', 'string', 'object', 'str']
    assert schema.schema['age'].get('dtype') in ['int', 'int64', 'int32']
    assert schema.schema['is_active'].get('dtype') in ['bool', 'boolean']
    assert schema.schema['score'].get('dtype') in ['float', 'float64']
    assert schema.schema['category'].get('dtype') in ['category', 'string', 'object', 'str']


def test_save_and_load_schema(sample_schema_dict):
    """Тест сохранения и загрузки схемы из файла."""
    # Создаем схему
    schema = DataSchema(sample_schema_dict)
    
    # Создаем временный файл для тестирования
    with tempfile.TemporaryDirectory() as temp_dir:
        # Тест сохранения в JSON
        json_path = os.path.join(temp_dir, 'schema.json')
        schema.save(json_path)
        
        # Проверяем, что файл создан
        assert os.path.exists(json_path)
        
        # Загружаем схему из файла
        loaded_schema = DataSchema.from_file(json_path)
        
        # Проверяем, что загруженная схема соответствует исходной
        assert loaded_schema.schema == sample_schema_dict
        
        # Тест сохранения в YAML
        yaml_path = os.path.join(temp_dir, 'schema.yaml')
        schema.save(yaml_path)
        
        # Проверяем, что файл создан
        assert os.path.exists(yaml_path)
        
        # Загружаем схему из файла
        loaded_schema_yaml = DataSchema.from_file(yaml_path)
        
        # Проверяем, что загруженная схема соответствует исходной
        assert loaded_schema_yaml.schema == sample_schema_dict


def test_apply_to_dataframe(sample_dataframe, sample_schema_dict):
    """Тест применения схемы к DataFrame."""
    # Создаем схему
    schema = DataSchema(sample_schema_dict)
    
    # Применяем схему к DataFrame
    df_optimized = schema.apply_to_dataframe(sample_dataframe)
    
    # Проверяем, что все колонки сохранились
    assert set(df_optimized.columns) == set(sample_dataframe.columns)
    
    # Проверяем, что данные не изменились
    pd.testing.assert_frame_equal(
        df_optimized.astype('object'), 
        sample_dataframe.astype('object')
    )
    
    # Проверяем типы данных
    assert pd.api.types.is_integer_dtype(df_optimized['id'])
    assert isinstance(df_optimized['name'].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_integer_dtype(df_optimized['age'])
    assert pd.api.types.is_bool_dtype(df_optimized['is_active'])
    assert pd.api.types.is_float_dtype(df_optimized['score'])
    assert isinstance(df_optimized['category'].dtype, pd.CategoricalDtype)


def test_column_operations():
    """Тест операций с колонками схемы."""
    # Создаем пустую схему
    schema = DataSchema()
    
    # Добавляем колонку
    schema.add_column(
        column='id',
        dtype='int',
        description='Идентификатор',
        required=True
    )
    
    # Проверяем, что колонка добавлена
    assert 'id' in schema.schema
    assert schema.schema['id']['dtype'] == 'int'
    assert schema.schema['id']['description'] == 'Идентификатор'
    assert schema.schema['id']['required'] == True
    
    # Обновляем колонку
    schema.update_column(
        column='id',
        description='Уникальный идентификатор',
        min_value=1
    )
    
    # Проверяем обновление
    assert schema.schema['id']['description'] == 'Уникальный идентификатор'
    assert schema.schema['id']['min_value'] == 1
    
    # Добавляем ещё колонку
    schema.add_column(
        column='name',
        dtype='category',
        description='Имя пользователя'
    )
    
    # Получаем колонки по типу
    int_columns = schema.get_columns_by_type('int')
    cat_columns = schema.get_columns_by_type('category')
    
    assert int_columns == ['id']
    assert cat_columns == ['name']
    
    # Удаляем колонку
    schema.remove_column('id')
    
    # Проверяем, что колонка удалена
    assert 'id' not in schema.schema
    assert 'name' in schema.schema


def test_get_column_info(sample_schema_dict):
    """Тест получения информации о колонке."""
    schema = DataSchema(sample_schema_dict)
    
    # Получаем информацию о существующей колонке
    info = schema.get_column_info('id')
    assert info == sample_schema_dict['id']
    
    # Получаем информацию о несуществующей колонке
    info = schema.get_column_info('non_existent')
    assert info == {}


def test_summary(sample_schema_dict):
    """Тест генерации сводки по схеме."""
    schema = DataSchema(sample_schema_dict)
    
    # Получаем сводку
    summary = schema.summary()
    
    # Проверяем, что сводка имеет правильную структуру
    assert isinstance(summary, pd.DataFrame)
    assert 'column' in summary.columns
    assert 'dtype' in summary.columns
    assert 'description' in summary.columns
    assert 'required' in summary.columns
    
    # Проверяем содержимое
    assert len(summary) == len(sample_schema_dict)
    for column in sample_schema_dict:
        assert column in summary['column'].values 