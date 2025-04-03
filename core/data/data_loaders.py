"""
Модуль для загрузки данных из различных источников.

Этот модуль предоставляет классы для загрузки данных из файлов,
баз данных и других источников с поддержкой валидации схемы данных.

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

    # Загрузка данных из CSV файла
    file_loader = FileDataLoader(schema)
    df = file_loader.load_data('data/users.csv')

    # Загрузка данных из PostgreSQL
    pg_loader = PostgresDataLoader(
        connection_params={
            'host': 'localhost',
            'port': 5432,
            'database': 'my_db',
            'user': 'user',
            'password': 'password',
            'use_ssl': True
        },
        schema=schema
    )
    df = pg_loader.load_data("SELECT * FROM users")

    # Загрузка данных из Impala
    impala_loader = ImpalaDataLoader(
        connection_params={
            'host': 'impala.example.com',
            'port': 21050,
            'database': 'default',
            'auth_mechanism': 'GSSAPI',
            'use_ssl': True
        },
        schema=schema
    )
    df = impala_loader.load_data("SELECT * FROM users")
    ```
"""

from typing import Dict, Optional, Union, List, Any
import pandas as pd
import sqlalchemy as sa
import os
from pathlib import Path
import json
import yaml
from .schema import DataSchema
from .validation import SchemaValidator

class DataLoader:
    """
    Базовый класс для загрузки данных.
    
    Предоставляет общий интерфейс для загрузки данных из различных источников
    с поддержкой валидации схемы.
    
    Parameters
    ----------
    schema : DataSchema, optional
        Схема данных для валидации, по умолчанию None
    
    Examples
    --------
    ```python
    # Создание базового загрузчика с валидацией
    schema = DataSchema([
        ColumnSchema(
            name='id',
            data_type=DataType.INTEGER,
            required=True
        )
    ])
    loader = DataLoader(schema)

    # Проверка данных на соответствие схеме
    df = pd.DataFrame({'id': [1, 2, 3]})
    try:
        loader.validate_data(df)
        print("Данные соответствуют схеме")
    except ValueError as e:
        print(f"Ошибка валидации: {e}")
    ```
    """
    
    def __init__(self, schema: Optional[DataSchema] = None):
        self.schema = schema
        self.validator = SchemaValidator(schema) if schema else None
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Загружает данные из источника.
        
        Parameters
        ----------
        **kwargs : dict
            Дополнительные параметры для загрузки данных
        
        Returns
        -------
        pd.DataFrame
            Загруженные данные
        
        Raises
        ------
        NotImplementedError
            Если метод не реализован в дочернем классе
        """
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Проверяет данные на соответствие схеме.
        
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
            Если схема не задана или данные не соответствуют схеме
        """
        if not self.validator:
            raise ValueError("Схема данных не задана")
        return self.validator.validate(df)

class FileDataLoader(DataLoader):
    """
    Класс для загрузки данных из файлов.
    
    Поддерживает загрузку данных из CSV, JSON, YAML и других форматов файлов.
    
    Parameters
    ----------
    schema : DataSchema, optional
        Схема данных для валидации, по умолчанию None
    
    Examples
    --------
    ```python
    # Создание схемы данных
    schema = DataSchema([
        ColumnSchema(
            name='id',
            data_type=DataType.INTEGER,
            required=True
        ),
        ColumnSchema(
            name='name',
            data_type=DataType.STRING,
            required=True
        )
    ])

    # Создание загрузчика файлов
    loader = FileDataLoader(schema)

    # Загрузка данных из CSV
    df = loader.load_data('data/users.csv')

    # Загрузка данных из JSON
    df = loader.load_data('data/users.json')

    # Загрузка данных из YAML
    df = loader.load_data('data/users.yaml')

    # Загрузка с дополнительными параметрами
    df = loader.load_data(
        'data/users.csv',
        sep=';',
        encoding='utf-8',
        na_values=['NA', 'missing']
    )
    ```
    """
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Загружает данные из файла.
        
        Parameters
        ----------
        file_path : str
            Путь к файлу
        **kwargs : dict
            Дополнительные параметры для pd.read_csv/read_json/etc.
        
        Returns
        -------
        pd.DataFrame
            Загруженные данные
        
        Raises
        ------
        FileNotFoundError
            Если файл не найден
        ValueError
            Если формат файла не поддерживается
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif ext == '.json':
            df = pd.read_json(file_path, **kwargs)
        elif ext == '.yaml' or ext == '.yml':
            with open(file_path) as f:
                data = yaml.safe_load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")
        
        if self.schema:
            self.validate_data(df)
        
        return df

class PostgresDataLoader(DataLoader):
    """
    Класс для загрузки данных из PostgreSQL.
    
    Parameters
    ----------
    connection_params : dict
        Параметры подключения к базе данных:
        - host : str
            Хост базы данных
        - port : int
            Порт базы данных
        - database : str
            Имя базы данных
        - user : str
            Имя пользователя
        - password : str
            Пароль
        - use_ssl : bool, optional
            Использовать SSL подключение
    schema : DataSchema, optional
        Схема данных для валидации, по умолчанию None
    
    Examples
    --------
    ```python
    # Создание схемы данных
    schema = DataSchema([
        ColumnSchema(
            name='id',
            data_type=DataType.INTEGER,
            required=True
        ),
        ColumnSchema(
            name='name',
            data_type=DataType.STRING,
            required=True
        )
    ])

    # Создание загрузчика PostgreSQL
    loader = PostgresDataLoader(
        connection_params={
            'host': 'localhost',
            'port': 5432,
            'database': 'my_db',
            'user': 'user',
            'password': 'password',
            'use_ssl': True
        },
        schema=schema
    )

    # Загрузка данных из таблицы
    df = loader.load_data("SELECT * FROM users")

    # Загрузка с дополнительными параметрами
    df = loader.load_data(
        "SELECT * FROM users WHERE age > %(min_age)s",
        params={'min_age': 18},
        index_col='id'
    )

    # Загрузка с использованием JOIN
    query = (
        "SELECT u.id, u.name, o.order_date, o.amount "
        "FROM users u "
        "JOIN orders o ON u.id = o.user_id "
        "WHERE o.order_date >= %(start_date)s"
    )
    df = loader.load_data(
        query,
        params={'start_date': '2024-01-01'}
    )
    ```
    """
    
    def __init__(self, connection_params: Dict[str, Any], schema: Optional[DataSchema] = None):
        super().__init__(schema)
        self._validate_connection_params(connection_params)
        self.connection_params = connection_params
        self.engine = self._create_engine()
    
    def _validate_connection_params(self, params: Dict[str, Any]) -> None:
        """
        Проверяет параметры подключения.
        
        Parameters
        ----------
        params : dict
            Параметры подключения
        
        Raises
        ------
        ValueError
            Если отсутствуют обязательные параметры
        """
        required = ['host', 'port', 'database', 'user', 'password']
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Отсутствуют обязательные параметры: {', '.join(missing)}")
    
    def _create_engine(self) -> sa.engine.Engine:
        """
        Создает движок SQLAlchemy.
        
        Returns
        -------
        sqlalchemy.engine.Engine
            Движок для подключения к базе данных
        
        Raises
        ------
        ImportError
            Если не установлен sqlalchemy
        """
        try:
            import sqlalchemy as sa
        except ImportError:
            raise ImportError("Для работы с PostgreSQL требуется установить sqlalchemy")
        
        connect_args = {}
        if self.connection_params.get('use_ssl'):
            connect_args['sslmode'] = 'require'
        
        url = sa.URL.create(
            drivername='postgresql',
            host=self.connection_params['host'],
            port=self.connection_params['port'],
            database=self.connection_params['database'],
            username=self.connection_params['user'],
            password=self.connection_params['password']
        )
        
        return sa.create_engine(url, connect_args=connect_args)
    
    def load_data(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Загружает данные из PostgreSQL.
        
        Parameters
        ----------
        query : str
            SQL запрос
        **kwargs : dict
            Дополнительные параметры для pd.read_sql
        
        Returns
        -------
        pd.DataFrame
            Загруженные данные
        
        Raises
        ------
        ValueError
            Если данные не соответствуют схеме
        """
        df = pd.read_sql(query, self.engine, **kwargs)
        
        if self.schema:
            self.validate_data(df)
            
            # Преобразуем типы данных согласно схеме
            for col in self.schema.columns:
                if col.name in df.columns:
                    df[col.name] = df[col.name].astype(col.data_type.value)
        
        return df

class ImpalaDataLoader(DataLoader):
    """
    Класс для загрузки данных из Impala.
    
    Parameters
    ----------
    connection_params : dict
        Параметры подключения к базе данных:
        - host : str
            Хост базы данных
        - port : int
            Порт базы данных
        - database : str
            Имя базы данных
        - auth_mechanism : str, optional
            Механизм аутентификации ('GSSAPI' или 'PLAIN')
        - user : str, optional
            Имя пользователя (для PLAIN)
        - password : str, optional
            Пароль (для PLAIN)
        - use_ssl : bool, optional
            Использовать SSL подключение
    schema : DataSchema, optional
        Схема данных для валидации, по умолчанию None
    
    Examples
    --------
    ```python
    # Создание схемы данных
    schema = DataSchema([
        ColumnSchema(
            name='id',
            data_type=DataType.INTEGER,
            required=True
        ),
        ColumnSchema(
            name='name',
            data_type=DataType.STRING,
            required=True
        )
    ])

    # Создание загрузчика Impala с GSSAPI аутентификацией
    loader = ImpalaDataLoader(
        connection_params={
            'host': 'impala.example.com',
            'port': 21050,
            'database': 'default',
            'auth_mechanism': 'GSSAPI',
            'use_ssl': True
        },
        schema=schema
    )

    # Создание загрузчика Impala с PLAIN аутентификацией
    loader = ImpalaDataLoader(
        connection_params={
            'host': 'impala.example.com',
            'port': 21050,
            'database': 'default',
            'auth_mechanism': 'PLAIN',
            'user': 'user',
            'password': 'password',
            'use_ssl': True
        },
        schema=schema
    )

    # Загрузка данных из таблицы
    df = loader.load_data("SELECT * FROM users")

    # Загрузка с дополнительными параметрами
    df = loader.load_data(
        "SELECT * FROM users WHERE age > %(min_age)s",
        params={'min_age': 18},
        index_col='id'
    )

    # Загрузка с использованием JOIN
    query = (
        "SELECT u.id, u.name, o.order_date, o.amount "
        "FROM users u "
        "JOIN orders o ON u.id = o.user_id "
        "WHERE o.order_date >= %(start_date)s"
    )
    df = loader.load_data(
        query,
        params={'start_date': '2024-01-01'}
    )
    ```
    """
    
    def __init__(self, connection_params: Dict[str, Any], schema: Optional[DataSchema] = None):
        super().__init__(schema)
        self._validate_connection_params(connection_params)
        self.connection_params = connection_params
        self.engine = self._create_engine()
    
    def _validate_connection_params(self, params: Dict[str, Any]) -> None:
        """
        Проверяет параметры подключения.
        
        Parameters
        ----------
        params : dict
            Параметры подключения
        
        Raises
        ------
        ValueError
            Если отсутствуют обязательные параметры
        """
        required = ['host', 'port', 'database']
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Отсутствуют обязательные параметры: {', '.join(missing)}")
    
    def _create_engine(self) -> sa.engine.Engine:
        """
        Создает движок SQLAlchemy.
        
        Returns
        -------
        sqlalchemy.engine.Engine
            Движок для подключения к базе данных
        
        Raises
        ------
        ImportError
            Если не установлен sqlalchemy
        """
        try:
            import sqlalchemy as sa
        except ImportError:
            raise ImportError("Для работы с Impala требуется установить sqlalchemy")
        
        connect_args = {}
        auth_mechanism = self.connection_params.get('auth_mechanism', 'GSSAPI')
        connect_args['auth_mechanism'] = auth_mechanism
        
        if auth_mechanism == 'PLAIN':
            if 'user' not in self.connection_params or 'password' not in self.connection_params:
                raise ValueError("Для PLAIN аутентификации требуются user и password")
            connect_args['user'] = self.connection_params['user']
            connect_args['password'] = self.connection_params['password']
        
        if self.connection_params.get('use_ssl'):
            connect_args['use_ssl'] = 'true'
        
        url = sa.URL.create(
            drivername='impala',
            host=self.connection_params['host'],
            port=self.connection_params['port'],
            database=self.connection_params['database']
        )
        
        return sa.create_engine(url, connect_args=connect_args)
    
    def load_data(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Загружает данные из Impala.
        
        Parameters
        ----------
        query : str
            SQL запрос
        **kwargs : dict
            Дополнительные параметры для pd.read_sql
        
        Returns
        -------
        pd.DataFrame
            Загруженные данные
        
        Raises
        ------
        ValueError
            Если данные не соответствуют схеме
        """
        df = pd.read_sql(query, self.engine, **kwargs)
        
        if self.schema:
            self.validate_data(df)
            
            # Преобразуем типы данных согласно схеме
            for col in self.schema.columns:
                if col.name in df.columns:
                    df[col.name] = df[col.name].astype(col.data_type.value)
        
        return df 