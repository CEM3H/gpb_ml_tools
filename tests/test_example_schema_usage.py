"""
Тесты для примеров использования схем данных из examples/data_schema_example.py
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.data_schema_example import (
    generate_sample_data,
    example_schema_from_dataframe,
    example_load_with_schema,
    example_without_schema,
    example_downcast_types
)
from core.data.schema import DataSchema
from core.data.data_types import downcast_types
from core.data.data_loaders import FileDataLoader


@pytest.fixture
def sample_data():
    """Генерирует небольшой набор тестовых данных"""
    return generate_sample_data(rows=100)


def test_generate_sample_data():
    """Тест функции generate_sample_data"""
    # Генерируем тестовые данные с небольшим количеством строк
    df = generate_sample_data(rows=10)
    
    # Проверяем, что данные созданы правильно
    assert len(df) == 10
    assert list(df.columns) == [
        'customer_id', 'full_name', 'birth_date', 'salary', 'credit_score',
        'risk_category', 'is_active', 'registration_date', 'last_update', 'city',
        'transactions_count', 'avg_transaction_amount', 'segment',
        'has_credit_card', 'has_debit_card', 'has_deposit'
    ]
    
    # Проверяем типы данных
    assert pd.api.types.is_integer_dtype(df['customer_id'])
    assert pd.api.types.is_object_dtype(df['birth_date'])  # строка после strftime
    assert pd.api.types.is_numeric_dtype(df['salary'])


def test_downcast_types(sample_data):
    """Тест функции downcast_types"""
    df = sample_data
    
    # Применяем оптимизацию типов
    df_optimized = downcast_types(df)
    
    # Проверяем, что размер данных уменьшился
    original_memory = df.memory_usage(deep=True).sum()
    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    assert optimized_memory <= original_memory
    
    # Проверяем, что данные не изменились
    pd.testing.assert_frame_equal(
        df_optimized.astype('object'), 
        df.astype('object')
    )


def test_schema_from_dataframe(sample_data):
    """Тест создания схемы из DataFrame"""
    df = sample_data
    
    # Создаем временную директорию для тестирования
    with tempfile.TemporaryDirectory() as temp_dir:
        # Сохраняем оригинальное значение schemas_dir
        original_schemas_dir = os.getcwd()
        
        try:
            # Переходим во временную директорию
            os.chdir(temp_dir)
            
            # Создаем директорию для схем
            os.makedirs('schemas', exist_ok=True)
            
            # Создаем схему из DataFrame
            schema = DataSchema.from_dataframe(df)
            
            # Проверяем, что схема создана для всех колонок
            for column in df.columns:
                assert column in schema.schema
            
            # Проверяем, что сводка генерируется корректно
            summary = schema.summary()
            assert isinstance(summary, pd.DataFrame)
            assert 'column' in summary.columns
            
            # Проверяем сохранение схемы
            schema_path = 'schemas/test_schema.json'
            schema.save(schema_path)
            assert os.path.exists(schema_path)
            
            # Загружаем схему
            loaded_schema = DataSchema.from_file(schema_path)
            assert loaded_schema.schema == schema.schema
            
        finally:
            # Возвращаемся в оригинальную директорию
            os.chdir(original_schemas_dir)


def test_file_data_loader():
    """Тест FileDataLoader с использованием схемы"""
    # Генерируем тестовые данные
    df = generate_sample_data(rows=50)
    
    # Создаем временную директорию для тестирования
    with tempfile.TemporaryDirectory() as temp_dir:
        # Пути для временных файлов
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        schema_path = os.path.join(temp_dir, 'test_schema.json')
        
        # Сохраняем данные в CSV
        df.to_csv(csv_path, index=False)
        
        # Создаем схему из DataFrame и сохраняем
        schema = DataSchema.from_dataframe(df)
        schema.save(schema_path)
        
        # Создаем загрузчик данных с указанием схемы
        loader = FileDataLoader({
            'file_path': csv_path,
            'file_type': 'csv',
            'schema_path': schema_path,
            'optimize_types': True
        })
        
        # Загружаем данные
        df_loaded = loader.load_data()
        
        # Проверяем, что количество строк совпадает
        assert len(df_loaded) == len(df)
        
        # Проверяем, что все колонки загружены
        assert set(df_loaded.columns) == set(df.columns)
        
        # Проверяем создание схемы из загрузчика
        created_schema = loader.create_schema()
        assert isinstance(created_schema, DataSchema)
        assert len(created_schema.schema) > 0
        
        # Проверяем получение информации о колонках из схемы
        column_info = loader.schema.get_column_info('customer_id')
        assert 'dtype' in column_info
        
        # Проверяем получение колонок по типу
        categorical_columns = loader.schema.get_columns_by_type('category')
        assert isinstance(categorical_columns, list)


def test_integration_example_functions():
    """Интеграционный тест функций из примера, с мокированием операций ввода-вывода"""
    # Создаем временную директорию для тестирования
    with tempfile.TemporaryDirectory() as temp_dir:
        # Сохраняем оригинальное значение текущей директории
        original_dir = os.getcwd()
        
        try:
            # Переходим во временную директорию
            os.chdir(temp_dir)
            
            # Создаем директорию для схем
            os.makedirs('schemas', exist_ok=True)
            
            # Копируем example_schema.json из основной директории
            original_schema_path = os.path.join(original_dir, 'schemas', 'example_schema.json')
            new_schema_path = os.path.join(temp_dir, 'schemas', 'example_schema.json')
            
            if os.path.exists(original_schema_path):
                # Чтение и запись файла с помощью DataSchema
                schema = DataSchema.from_file(original_schema_path)
                schema.save(new_schema_path)
            
            try:
                # 1. Тестируем функцию downcast_types
                example_downcast_types()
                
                # 2. Тестируем функцию example_schema_from_dataframe с мок-объектом
                with patch('examples.data_schema_example.DataSchema.save') as mock_save:
                    example_schema_from_dataframe()
                    assert mock_save.called
                
                # 3. Тестируем функцию example_load_with_schema с мок-объектами
                # Создаем тестовый CSV файл
                test_df = generate_sample_data(rows=100)
                test_csv_path = os.path.join(temp_dir, 'example_data.csv')
                test_df.to_csv(test_csv_path, index=False)
                
                # Патчим функцию os.remove, чтобы предотвратить удаление файла
                with patch('os.remove') as mock_remove:
                    # Патчим FileDataLoader для использования нашего пути
                    with patch('examples.data_schema_example.FileDataLoader', return_value=FileDataLoader({
                        'file_path': test_csv_path,
                        'file_type': 'csv',
                        'schema_path': new_schema_path if os.path.exists(new_schema_path) else None,
                        'optimize_types': True
                    })):
                        # Выполняем функцию примера
                        if os.path.exists(new_schema_path):
                            example_load_with_schema()
                            assert mock_remove.called
                
                # 4. Тестируем функцию example_without_schema с мок-объектами
                test_df = generate_sample_data(rows=100)
                test_csv_path = os.path.join(temp_dir, 'example_data_no_schema.csv')
                test_df.to_csv(test_csv_path, index=False)
                
                # Патчим функцию os.remove, чтобы предотвратить удаление файла
                with patch('os.remove') as mock_remove:
                    # Патчим FileDataLoader для использования нашего пути
                    with patch('examples.data_schema_example.FileDataLoader', return_value=FileDataLoader({
                        'file_path': test_csv_path,
                        'file_type': 'csv',
                        'optimize_types': True
                    })):
                        # Выполняем функцию примера
                        example_without_schema()
                        assert mock_remove.called
                
                assert True  # Если дошли до этой точки, все функции выполнились без ошибок
                
            except Exception as e:
                pytest.fail(f"Выполнение примеров вызвало исключение: {str(e)}")
            
        finally:
            # Возвращаемся в оригинальную директорию
            os.chdir(original_dir) 