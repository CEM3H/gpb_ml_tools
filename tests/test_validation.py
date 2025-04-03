"""
Тесты для модуля валидации схем данных.
"""
import pytest
import pandas as pd
from datetime import datetime
from core.data.schema import DataSchema, ColumnSchema, DataType
from core.data.validation import SchemaValidator

@pytest.fixture
def schema():
    """Создает тестовую схему данных."""
    columns = [
        ColumnSchema(
            name='id',
            data_type=DataType.INTEGER,
            required=True,
            description='Уникальный идентификатор',
            constraints={'unique': True}
        ),
        ColumnSchema(
            name='value',
            data_type=DataType.FLOAT,
            required=True,
            description='Числовое значение',
            constraints={'min': 0, 'max': 100}
        ),
        ColumnSchema(
            name='category',
            data_type=DataType.CATEGORICAL,
            required=True,
            description='Категория',
            categories=['A', 'B', 'C']
        ),
        ColumnSchema(
            name='date',
            data_type=DataType.DATETIME,
            required=True,
            description='Дата',
            constraints={'min': '2023-01-01', 'max': '2023-12-31'}
        ),
        ColumnSchema(
            name='text',
            data_type=DataType.STRING,
            required=True,
            description='Текст',
            constraints={'pattern': r'^[A-Za-z]+$'}
        )
    ]
    return DataSchema(columns=columns)

@pytest.fixture
def validator(schema):
    """Создает валидатор для тестовой схемы."""
    return SchemaValidator(schema)

def test_validate_valid_data(validator):
    """Тест валидации корректных данных."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.0, 30.7],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'text': ['Hello', 'World', 'Test']
    })
    
    assert validator.validate(df) is True

def test_validate_missing_required_column(validator):
    """Тест валидации при отсутствии обязательной колонки."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.0, 30.7],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })
    
    with pytest.raises(ValueError, match="Отсутствуют обязательные колонки"):
        validator.validate(df)

def test_validate_invalid_integer(validator):
    """Тест валидации некорректных целых чисел."""
    df = pd.DataFrame({
        'id': [1, 2, 3.5],
        'value': [10.5, 20.0, 30.7],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'text': ['Hello', 'World', 'Test']
    })
    
    with pytest.raises(ValueError, match="должна содержать целые числа"):
        validator.validate(df)

def test_validate_invalid_float(validator):
    """Тест валидации некорректных чисел с плавающей точкой."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.0, 150.0],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'text': ['Hello', 'World', 'Test']
    })
    
    with pytest.raises(ValueError, match="содержит значения больше"):
        validator.validate(df)

def test_validate_invalid_categorical(validator):
    """Тест валидации некорректных категориальных значений."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.0, 30.7],
        'category': pd.Categorical(['A', 'B', 'D'], categories=['A', 'B', 'D']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'text': ['Hello', 'World', 'Test']
    })
    
    with pytest.raises(ValueError, match="содержит недопустимые категории"):
        validator.validate(df)

def test_validate_invalid_datetime(validator):
    """Тест валидации некорректных дат."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.0, 30.7],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': ['2023-01-01', '2023-01-02', 'invalid_date'],
        'text': ['Hello', 'World', 'Test']
    })
    
    with pytest.raises(ValueError, match="должна содержать даты и время"):
        validator.validate(df)

def test_validate_invalid_string_pattern(validator):
    """Тест валидации строк по паттерну."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.0, 30.7],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'text': ['Hello', 'World123', 'Test']
    })
    
    with pytest.raises(ValueError, match="содержит значения, не соответствующие паттерну"):
        validator.validate(df)

def test_validate_value_constraints(validator):
    """Тест валидации ограничений на значения."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.0, -5.0],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'text': ['Hello', 'World', 'Test']
    })
    
    with pytest.raises(ValueError, match="содержит значения меньше"):
        validator.validate(df)

def test_validate_unique_constraint(validator):
    """Тест валидации ограничения на уникальность."""
    df = pd.DataFrame({
        'id': [1, 2, 1],
        'value': [10.5, 20.0, 30.7],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'text': ['Hello', 'World', 'Test']
    })
    
    with pytest.raises(ValueError, match="должна содержать уникальные значения"):
        validator.validate(df)

def test_validate_not_null_constraint(validator):
    """Тест валидации ограничения на отсутствие null значений."""
    df = pd.DataFrame({
        'id': [1, 2, None],
        'value': [10.5, 20.0, 30.7],
        'category': pd.Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'text': ['Hello', 'World', 'Test']
    })
    
    with pytest.raises(ValueError, match="не должна содержать пропущенные значения"):
        validator.validate(df) 