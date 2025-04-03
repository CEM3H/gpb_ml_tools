from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import json

def downcast_types(df: pd.DataFrame, optimize_types: bool = True) -> pd.DataFrame:
    """
    Приведение типов данных DataFrame к оптимальным для экономии памяти.
    
    Args:
        df: Исходный DataFrame
        optimize_types: Флаг, указывающий нужно ли оптимизировать типы данных
        
    Returns:
        DataFrame с оптимизированными типами данных
    """
    if not optimize_types:
        return df
    
    result = df.copy()
    
    # Оптимизация целочисленных колонок
    int_cols = result.select_dtypes(include=['int']).columns
    for col in int_cols:
        # Проверяем наличие отрицательных значений
        if result[col].min() >= 0:
            # Если нет отрицательных значений, используем беззнаковые целые
            if result[col].max() <= 255:
                result[col] = result[col].astype(np.uint8)
            elif result[col].max() <= 65535:
                result[col] = result[col].astype(np.uint16)
            elif result[col].max() <= 4294967295:
                result[col] = result[col].astype(np.uint32)
            else:
                result[col] = result[col].astype(np.uint64)
        else:
            # Если есть отрицательные значения, используем знаковые целые
            if result[col].min() >= -128 and result[col].max() <= 127:
                result[col] = result[col].astype(np.int8)
            elif result[col].min() >= -32768 and result[col].max() <= 32767:
                result[col] = result[col].astype(np.int16)
            elif result[col].min() >= -2147483648 and result[col].max() <= 2147483647:
                result[col] = result[col].astype(np.int32)
            else:
                result[col] = result[col].astype(np.int64)
    
    # Оптимизация числовых колонок с плавающей точкой
    float_cols = result.select_dtypes(include=['float']).columns
    for col in float_cols:
        # Проверяем, можно ли преобразовать в float32 без потери точности
        # Для простоты используем float32 для всех float-колонок 
        # (в реальных проектах может потребоваться более тонкая логика)
        result[col] = result[col].astype(np.float32)
    
    # Оптимизация категориальных колонок
    cat_cols = result.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Если уникальных значений меньше половины от общего числа значений
        if result[col].nunique() < len(result[col]) / 2:
            result[col] = result[col].astype('category')
    
    return result

def apply_dtypes_from_schema(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]], 
                             optimize_types: bool = True) -> pd.DataFrame:
    """
    Применяет типы данных из схемы к DataFrame с возможностью оптимизации.
    
    Функция преобразует типы данных каждой колонки DataFrame в соответствии со схемой.
    Если включена оптимизация типов, то для числовых колонок будут использованы
    наиболее экономичные типы данных на основе анализа значений.
    
    Args:
        df: Исходный DataFrame
        schema: Словарь схемы данных, где ключи - имена колонок,
                значения - словари с информацией о типах данных
        optimize_types: Флаг оптимизации типов данных (по умолчанию True).
                        Если True, будут использованы наиболее экономичные типы.
                        
    Returns:
        DataFrame с оптимизированными типами данных
        
    Примеры:
        ```python
        # Создание простого DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Anna', 'Bob', 'Charlie', 'David', 'Elena'],
            'age': [25, 32, 41, 29, 35],
            'registered': [True, True, False, True, False]
        })
        
        # Создание схемы
        schema = {
            'id': {'dtype': 'int'},
            'name': {'dtype': 'category'},
            'age': {'dtype': 'int'},
            'registered': {'dtype': 'bool'}
        }
        
        # Применение схемы с оптимизацией
        df_optimized = apply_dtypes_from_schema(df, schema)
        
        # Вывод информации о типах данных и использовании памяти
        print(df_optimized.dtypes)
        print(f"Память до оптимизации: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print(f"Память после оптимизации: {df_optimized.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Применение схемы без оптимизации (стандартные типы pandas)
        df_standard = apply_dtypes_from_schema(df, schema, optimize_types=False)
        print(df_standard.dtypes)
        ```
    """
    result_df = df.copy()
    
    # Применяем типы данных к каждой колонке
    for column, info in schema.items():
        if column not in df.columns:
            continue
        
        dtype = info.get('dtype')
        if not dtype:
            continue
        
        try:
            # Обработка категориальных данных
            if dtype == 'category':
                result_df[column] = result_df[column].astype('category')
            
            # Обработка логических данных
            elif dtype == 'bool':
                result_df[column] = result_df[column].astype('bool')
            
            # Обработка данных даты и времени
            elif dtype == 'datetime':
                result_df[column] = pd.to_datetime(result_df[column])
            
            # Обработка числовых данных с оптимизацией
            elif dtype in ['int', 'float'] and optimize_types:
                optimal_dtype = get_optimal_numeric_dtype(result_df[column])
                result_df[column] = result_df[column].astype(optimal_dtype)
            
            # Обработка числовых данных без оптимизации
            elif dtype == 'int':
                result_df[column] = result_df[column].astype('int64')
            elif dtype == 'float':
                result_df[column] = result_df[column].astype('float64')
            
            # Обработка строковых данных
            elif dtype == 'str':
                result_df[column] = result_df[column].astype('str')
                
        except Exception as e:
            print(f"Ошибка при преобразовании колонки {column}: {str(e)}")
    
    return result_df

def infer_dtypes(df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Dict[str, Any]]:
    """
    Автоматически определяет типы данных и метаданные для колонок DataFrame.
    
    Функция анализирует каждую колонку DataFrame и создает словарь метаданных, содержащий:
    - dtype: рекомендуемый тип данных для хранения
    - description: автоматически сгенерированное описание
    - min/max: минимальное и максимальное значения для числовых колонок
    - unique_count: количество уникальных значений для категориальных колонок
    - example: пример значения из колонки
    
    Args:
        df: DataFrame для анализа
        sample_size: размер выборки для анализа.
                     Если размер DataFrame больше sample_size, будет использована выборка
                     для ускорения анализа.
                     
    Returns:
        Словарь схемы данных, где ключи - имена колонок, а значения - словари с метаданными
        
    Примеры:
        ```python
        # Анализ небольшого DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Anna', 'Bob', 'Charlie', 'David', 'Elena'],
            'age': [25, 32, 41, 29, 35],
            'registered': [True, True, False, True, False]
        })
        
        schema = infer_dtypes(df)
        print(json.dumps(schema, indent=2))
        
        # Анализ большого DataFrame с выборкой
        df_large = pd.read_csv('large_data.csv')
        schema = infer_dtypes(df_large, sample_size=10000)
        
        # Сохранение результата в файл
        with open('data_schema.json', 'w') as f:
            json.dump(schema, f, indent=2)
        ```
    """
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df.copy()
        
    schema = {}
    
    for column in df.columns:
        try:
            col_data = sample_df[column]
            col_dtype = str(df[column].dtype)
            
            # Основная информация
            column_info = {
                'dtype': col_dtype,
                'description': f'Column {column}',
                'example': str(col_data.iloc[0]) if not col_data.empty else None
            }
            
            # Числовые данные
            if np.issubdtype(df[column].dtype, np.number):
                column_info['min'] = float(col_data.min()) if not pd.isna(col_data.min()) else None
                column_info['max'] = float(col_data.max()) if not pd.isna(col_data.max()) else None
                
                # Подбираем оптимальный тип для хранения
                if np.issubdtype(df[column].dtype, np.integer):
                    column_info['dtype'] = 'int'
                else:
                    column_info['dtype'] = 'float'
            
            # Категориальные данные
            elif df[column].dtype == 'object' or df[column].dtype.name == 'category':
                unique_count = col_data.nunique()
                ratio = unique_count / len(col_data) if len(col_data) > 0 else 0
                
                column_info['unique_count'] = int(unique_count)
                
                # Если мало уникальных значений, предлагаем категориальный тип
                if ratio < 0.5 and unique_count < 100:
                    column_info['dtype'] = 'category'
                else:
                    column_info['dtype'] = 'str'
            
            # Логические данные
            elif df[column].dtype == 'bool':
                column_info['dtype'] = 'bool'
            
            # Даты и время
            elif np.issubdtype(df[column].dtype, np.datetime64):
                column_info['dtype'] = 'datetime'
                
            schema[column] = column_info
        except Exception as e:
            # Если произошла ошибка при обработке колонки, заполняем базовой информацией
            schema[column] = {
                'dtype': str(df[column].dtype),
                'description': f'Column {column} (error during inference: {str(e)})'
            }
    
    return schema

def get_optimal_numeric_dtype(series: pd.Series) -> str:
    """
    Определяет оптимальный числовой тип данных для серии значений.
    
    Функция анализирует значения в серии и определяет наиболее экономичный
    тип данных, который может хранить все возможные значения:
    - Для целых чисел: int8, int16, int32, int64
    - Для чисел с плавающей точкой: float32, float64
    
    Args:
        series: Серия pandas с числовыми значениями
        
    Returns:
        Строка с оптимальным типом данных pandas
        
    Примеры:
        ```python
        # Определение оптимального типа для целых чисел в небольшом диапазоне
        s1 = pd.Series([1, 2, 3, 4, 5, -5, -10])
        optimal_type = get_optimal_numeric_dtype(s1)  # Вернет 'int8'
        
        # Определение оптимального типа для больших целых чисел
        s2 = pd.Series([100000, 200000, 300000])
        optimal_type = get_optimal_numeric_dtype(s2)  # Вернет 'int32'
        
        # Определение оптимального типа для чисел с плавающей точкой
        s3 = pd.Series([1.5, 2.7, 3.14159])
        optimal_type = get_optimal_numeric_dtype(s3)  # Вернет 'float32'
        
        # Применение оптимального типа к колонке DataFrame
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        optimal_type = get_optimal_numeric_dtype(df['value'])
        df['value'] = df['value'].astype(optimal_type)
        print(f"Новый тип колонки: {df['value'].dtype}")
        ```
    """
    # Проверяем, что серия не пустая
    if len(series) == 0:
        return 'object'
    
    # Если есть NaN значения, убираем их для анализа
    s = series.dropna()
    if len(s) == 0:
        return 'float32'  # Если все значения NaN, используем float
    
    # Проверяем, что это числовые данные
    if not np.issubdtype(s.dtype, np.number):
        return 'object'
    
    # Проверяем, целые ли числа
    if np.issubdtype(s.dtype, np.integer) or s.equals(s.astype(int)):
        min_val, max_val = s.min(), s.max()
        
        # Определяем оптимальный тип для целых чисел
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            return 'int8'
        elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            return 'int16'
        elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            return 'int32'
        else:
            return 'int64'
    else:
        # Проверяем, можно ли использовать float32 без потери точности
        # (это приблизительная проверка)
        if s.min() >= np.finfo(np.float32).min and s.max() <= np.finfo(np.float32).max:
            # Дополнительная проверка на точность
            s32 = s.astype('float32')
            if (s - s32).abs().max() < 1e-6:
                return 'float32'
        
        return 'float64'

def validate_data_against_schema(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]], 
                                strict: bool = False) -> List[str]:
    """
    Проверяет соответствие DataFrame схеме данных.
    
    Функция проверяет наличие обязательных колонок, типы данных и другие
    ограничения, указанные в схеме. По умолчанию проверка нестрогая (проверяются
    только обязательные колонки). В строгом режиме проверяются все колонки и типы.
    
    Args:
        df: DataFrame для проверки
        schema: Словарь схемы данных
        strict: Флаг строгой проверки (по умолчанию False).
                Если True, проверяются все колонки из схемы и их типы.
                Если False, проверяются только обязательные колонки.
                
    Returns:
        Список найденных ошибок. Если список пустой, ошибок нет.
        
    Примеры:
        ```python
        # Создание тестового DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'optional_col': [10, 20, 30]
        })
        
        # Создание схемы с обязательными колонками
        schema = {
            'id': {'dtype': 'int', 'required': True},
            'name': {'dtype': 'category', 'required': True},
            'missing_col': {'dtype': 'float', 'required': True},
            'optional_col': {'dtype': 'int', 'required': False}
        }
        
        # Нестрогая проверка (выявит отсутствие обязательной колонки missing_col)
        errors = validate_data_against_schema(df, schema)
        print("Ошибки нестрогой проверки:", errors)
        
        # Строгая проверка (выявит также несоответствие типа name)
        errors = validate_data_against_schema(df, schema, strict=True)
        print("Ошибки строгой проверки:", errors)
        
        # Проверка фильтрации по размеру значений
        schema_with_constraints = {
            'id': {'dtype': 'int', 'required': True, 'min': 2},
            'name': {'dtype': 'str', 'required': True}
        }
        errors = validate_data_against_schema(df, schema_with_constraints)
        print("Ошибки с ограничениями:", errors)
        ```
    """
    errors = []
    
    # Проверка наличия обязательных колонок
    for column, info in schema.items():
        if info.get('required', False) and column not in df.columns:
            errors.append(f"Отсутствует обязательная колонка '{column}'")
    
    # Если нет обязательных колонок или не нужна строгая проверка, возвращаем результат
    if not strict or not errors:
        # Проверка типов данных для имеющихся колонок
        for column, info in schema.items():
            if column not in df.columns:
                continue
            
            dtype = info.get('dtype')
            if not dtype:
                continue
            
            # Проверка типа данных
            if dtype == 'int' and not np.issubdtype(df[column].dtype, np.integer):
                errors.append(f"Колонка '{column}' должна быть целочисленной, текущий тип: {df[column].dtype}")
            
            elif dtype == 'float' and not np.issubdtype(df[column].dtype, np.number):
                errors.append(f"Колонка '{column}' должна быть числом с плавающей точкой, текущий тип: {df[column].dtype}")
            
            elif dtype == 'bool' and df[column].dtype != 'bool':
                errors.append(f"Колонка '{column}' должна быть логического типа, текущий тип: {df[column].dtype}")
            
            elif dtype == 'datetime' and not np.issubdtype(df[column].dtype, np.datetime64):
                errors.append(f"Колонка '{column}' должна быть типа datetime, текущий тип: {df[column].dtype}")
            
            # Проверка ограничений
            if 'min' in info and df[column].min() < info['min']:
                errors.append(f"Значения в колонке '{column}' не могут быть меньше {info['min']}")
            
            if 'max' in info and df[column].max() > info['max']:
                errors.append(f"Значения в колонке '{column}' не могут быть больше {info['max']}")
            
            if 'allowed_values' in info:
                invalid_values = set(df[column].unique()) - set(info['allowed_values'])
                if invalid_values:
                    errors.append(f"Колонка '{column}' содержит недопустимые значения: {invalid_values}")
    
    return errors

def extract_schema_subset(schema: Dict[str, Dict[str, Any]], 
                         columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Извлекает подмножество схемы для указанных колонок.
    
    Функция создает новую схему, содержащую только указанные колонки из исходной схемы.
    Если список колонок не указан, возвращается полная копия схемы.
    
    Args:
        schema: Исходная схема данных
        columns: Список колонок для включения в подмножество (если None, используются все колонки)
        
    Returns:
        Новая схема, содержащая только указанные колонки
        
    Примеры:
        ```python
        # Создание тестовой схемы
        schema = {
            'id': {'dtype': 'int', 'description': 'ID клиента'},
            'name': {'dtype': 'str', 'description': 'Имя клиента'},
            'age': {'dtype': 'int', 'description': 'Возраст клиента'},
            'balance': {'dtype': 'float', 'description': 'Баланс счета'}
        }
        
        # Извлечение подмножества колонок
        subset = extract_schema_subset(schema, ['id', 'name'])
        print(subset)  # Выведет схему только для колонок 'id' и 'name'
        
        # Извлечение всей схемы (создание копии)
        full_copy = extract_schema_subset(schema)
        
        # Использование с DataFrame для создания подмножества данных
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'age': [25, 30, 35],
            'balance': [100.0, 200.0, 300.0]
        })
        
        # Получение схемы только для числовых колонок
        numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
        numeric_schema = extract_schema_subset(schema, numeric_cols)
        ```
    """
    if columns is None:
        return {col: info.copy() for col, info in schema.items()}
    
    return {col: info.copy() for col, info in schema.items() if col in columns} 