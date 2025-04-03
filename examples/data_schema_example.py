import os
import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.schema import DataSchema
from core.data.data_types import downcast_types, apply_dtypes_from_schema
from core.data.data_loaders import FileDataLoader, PostgresDataLoader, ImpalaDataLoader


def generate_sample_data(rows=100000):
    """Генерирует тестовые данные для примера"""
    np.random.seed(42)
    
    # Создаем тестовый DataFrame
    data = {
        'customer_id': np.arange(1, rows + 1),
        'full_name': [f"Клиент {i}" for i in range(1, rows + 1)],
        'birth_date': pd.date_range(start='1950-01-01', periods=rows, freq='D'),
        'salary': np.random.normal(100000, 50000, rows),
        'credit_score': np.random.randint(300, 851, rows),
        'risk_category': np.random.choice(['Low', 'Medium', 'High', 'Very High'], rows),
        'is_active': np.random.choice([True, False], rows, p=[0.9, 0.1]),
        'registration_date': pd.date_range(start='2020-01-01', periods=rows, freq='h'),
        'last_update': pd.date_range(start='2023-01-01', periods=rows, freq='h'),
        'city': np.random.choice(['Москва', 'Санкт-Петербург', 'Казань', 'Новосибирск', 'Екатеринбург'], rows),
        'transactions_count': np.random.randint(0, 1000, rows),
        'avg_transaction_amount': np.random.normal(5000, 2000, rows),
        'segment': np.random.choice(['Mass', 'Upper Mass', 'Affluent', 'Private'], rows),
        'has_credit_card': np.random.choice([True, False], rows, p=[0.7, 0.3]),
        'has_debit_card': np.random.choice([True, False], rows, p=[0.95, 0.05]),
        'has_deposit': np.random.choice([True, False], rows, p=[0.4, 0.6])
    }
    
    df = pd.DataFrame(data)
    
    # Преобразуем datetime колонки в строковый формат для наглядности примера
    df['birth_date'] = df['birth_date'].dt.strftime('%Y-%m-%d')
    df['registration_date'] = df['registration_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['last_update'] = df['last_update'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df


def example_downcast_types():
    """Пример использования функции downcast_types"""
    print("Пример оптимизации типов данных с помощью downcast_types")
    print("-" * 80)
    
    # Генерируем тестовые данные
    df = generate_sample_data()
    
    # Выводим информацию о типах и памяти до оптимизации
    print("До оптимизации:")
    print(df.dtypes)
    memory_before = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Память: {memory_before:.2f} МБ")
    
    # Засекаем время оптимизации
    start_time = time.time()
    
    # Оптимизируем типы данных
    df_optimized = downcast_types(df)
    
    end_time = time.time()
    
    # Выводим информацию о типах и памяти после оптимизации
    print("\nПосле оптимизации:")
    print(df_optimized.dtypes)
    memory_after = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Память: {memory_after:.2f} МБ")
    print(f"Экономия памяти: {(1 - memory_after / memory_before) * 100:.2f}%")
    print(f"Время оптимизации: {end_time - start_time:.2f} сек")
    
    print("-" * 80)


def example_schema_from_dataframe():
    """Пример создания схемы данных из DataFrame"""
    print("Пример создания схемы данных из DataFrame")
    print("-" * 80)
    
    # Генерируем тестовые данные
    df = generate_sample_data(rows=1000)  # Используем меньше строк для примера
    
    # Создаем схему данных из DataFrame
    schema = DataSchema.from_dataframe(df)
    
    # Выводим информацию о схеме
    print("Созданная схема данных:")
    
    # Выводим сводку по схеме
    print(schema.summary().head())
    
    # Сохраняем схему в файл
    schema_path = 'schemas/auto_generated_schema.json'
    schema.save(schema_path)
    print(f"Схема сохранена в {schema_path}")
    
    print("-" * 80)


def example_load_with_schema():
    """Пример загрузки данных с использованием схемы"""
    print("Пример загрузки данных с использованием схемы")
    print("-" * 80)
    
    # Генерируем тестовые данные и сохраняем в CSV
    df = generate_sample_data()
    csv_path = 'example_data.csv'
    df.to_csv(csv_path, index=False)
    
    # Путь к примеру схемы
    schema_path = 'schemas/example_schema.json'
    
    # Создаем загрузчик данных с указанием схемы
    loader = FileDataLoader({
        'file_path': csv_path,
        'file_type': 'csv',
        'schema_path': schema_path,
        'optimize_types': True
    })
    
    # Загружаем данные
    start_time = time.time()
    df_loaded = loader.load_data()
    end_time = time.time()
    
    # Выводим информацию о загруженных данных
    print(f"Загружено строк: {len(df_loaded)}")
    print(f"Время загрузки: {end_time - start_time:.2f} сек")
    print("\nТипы данных после загрузки:")
    print(df_loaded.dtypes)
    
    memory_usage = df_loaded.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Память: {memory_usage:.2f} МБ")
    
    # Удаляем временный файл
    os.remove(csv_path)
    
    # Получаем информацию о колонках
    print("\nИнформация о колонках из схемы:")
    print(f"Колонка 'customer_id': {loader.schema.get_column_info('customer_id')}")
    print(f"Категориальные колонки: {loader.schema.get_columns_by_type('category')}")
    
    print("-" * 80)


def example_without_schema():
    """Пример загрузки данных без схемы"""
    print("Пример загрузки данных без схемы (автоматическая оптимизация)")
    print("-" * 80)
    
    # Генерируем тестовые данные и сохраняем в CSV
    df = generate_sample_data()
    csv_path = 'example_data_no_schema.csv'
    df.to_csv(csv_path, index=False)
    
    # Создаем загрузчик данных без указания схемы
    loader = FileDataLoader({
        'file_path': csv_path,
        'file_type': 'csv',
        'optimize_types': True
    })
    
    # Загружаем данные
    start_time = time.time()
    df_loaded = loader.load_data()
    end_time = time.time()
    
    # Выводим информацию о загруженных данных
    print(f"Загружено строк: {len(df_loaded)}")
    print(f"Время загрузки: {end_time - start_time:.2f} сек")
    print("\nТипы данных после загрузки:")
    print(df_loaded.dtypes)
    
    memory_usage = df_loaded.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Память: {memory_usage:.2f} МБ")
    
    # Удаляем временный файл
    os.remove(csv_path)
    
    # Создаем схему на основе загруженных данных
    schema = loader.create_schema()
    print("\nАвтоматически созданная схема:")
    print(schema.summary().head())
    
    print("-" * 80)


def main():
    """Основная функция"""
    # Создаем директорию для схем, если она не существует
    os.makedirs('schemas', exist_ok=True)
    
    # Запускаем примеры
    print("\n=== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ СПРАВОЧНИКОВ ДАННЫХ И ОПТИМИЗАЦИИ ТИПОВ ===\n")
    
    example_downcast_types()
    example_schema_from_dataframe()
    example_load_with_schema()
    example_without_schema()
    
    print("\n=== ЗАВЕРШЕНИЕ ПРИМЕРОВ ===\n")


if __name__ == "__main__":
    main() 