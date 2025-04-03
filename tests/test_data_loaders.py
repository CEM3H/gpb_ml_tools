import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sqlalchemy as sa
import numpy as np

from core.data.data_loaders import PostgresDataLoader, ImpalaDataLoader


class TestPostgresDataLoader:
    """Тесты для PostgresDataLoader"""
    
    @patch('core.data.data_loaders.sa.create_engine')
    def test_postgres_init(self, mock_create_engine):
        """Тест инициализации PostgresDataLoader"""
        # Настройка мока
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Создание загрузчика без SSL
        connection_params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        }
        
        loader = PostgresDataLoader(connection_params)
        
        # Проверка создания engine без SSL
        mock_create_engine.assert_called_once()
        args, kwargs = mock_create_engine.call_args
        assert args[0] == "postgresql://test_user:test_password@localhost:5432/test_db"
        assert 'connect_args' in kwargs
        assert kwargs['connect_args'] == {}
    
    @patch('core.data.data_loaders.sa.create_engine')
    def test_postgres_init_with_ssl(self, mock_create_engine):
        """Тест инициализации PostgresDataLoader с SSL"""
        # Настройка мока
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Создание загрузчика с SSL
        connection_params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password',
            'use_ssl': 'true'
        }
        
        loader = PostgresDataLoader(connection_params)
        
        # Проверка создания engine с SSL
        mock_create_engine.assert_called_once()
        args, kwargs = mock_create_engine.call_args
        assert args[0] == "postgresql://test_user:test_password@localhost:5432/test_db"
        assert 'connect_args' in kwargs
        assert kwargs['connect_args'] == {'sslmode': 'require'}
    
    @patch('core.data.data_loaders.sa.create_engine')
    @patch('core.data.data_loaders.pd.read_sql')
    def test_postgres_load_data(self, mock_read_sql, mock_create_engine):
        """Тест загрузки данных из PostgreSQL"""
        # Настройка моков
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Мок для DataFrame
        df = pd.DataFrame({
            'col1': np.array([1, 2, 3], dtype=np.uint8),
            'col2': ['a', 'b', 'c']
        })
        mock_read_sql.return_value = df
        
        # Отключаем оптимизацию типов
        loader = PostgresDataLoader({
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        })
        
        # При вызове load_data отключаем оптимизацию типов
        result = loader.load_data("SELECT * FROM test_table", optimize_types=False)
        
        # Проверка вызова read_sql
        mock_read_sql.assert_called_once_with("SELECT * FROM test_table", mock_engine)
        
        # Проверка результата
        pd.testing.assert_frame_equal(result, df)
    
    @patch('core.data.data_loaders.sa.create_engine')
    @patch('core.data.data_loaders.pd.read_sql')
    @patch('core.data.data_loaders.os.path.exists')
    @patch('core.data.data_loaders.DataSchema')
    @patch('core.data.data_loaders.DataLoader.get_schema_path')
    def test_postgres_load_data_with_schema(self, mock_get_schema_path, mock_schema_class, mock_exists, mock_read_sql, mock_create_engine):
        """Тест загрузки данных из PostgreSQL с схемой данных"""
        # Настройка моков
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Мок для DataFrame
        df = pd.DataFrame({
            'col1': np.array([1, 2, 3], dtype=np.uint8),
            'col2': ['a', 'b', 'c']
        })
        mock_read_sql.return_value = df
        
        # Настройка мока для пути к схеме
        mock_schema_path = 'mock_schema_path.json'
        mock_get_schema_path.return_value = mock_schema_path
        
        # Мок для схемы данных
        mock_exists.return_value = True
        mock_schema = MagicMock()
        mock_schema_class.load.return_value = mock_schema
        
        # Заменяем загрузку схемы из файла на возврат мока
        with patch('core.data.base.DataSchema.from_file', return_value=mock_schema):
            # Создание загрузчика и загрузка данных с схемой
            loader = PostgresDataLoader({
                'host': 'localhost',
                'port': 5432,
                'database': 'test_db',
                'user': 'test_user',
                'password': 'test_password'
            })
            
            result = loader.load_data("SELECT * FROM test_table", schema_name='test_schema', optimize_types=False)
            
            # Проверка вызова методов для схемы данных
            assert mock_exists.called
            mock_get_schema_path.assert_called_once_with('test_schema')
            
            # Проверка вызова read_sql
            mock_read_sql.assert_called_once_with("SELECT * FROM test_table", mock_engine)
            
            # Проверка результата
            pd.testing.assert_frame_equal(result, df)


class TestImpalaDataLoader:
    """Тесты для ImpalaDataLoader"""
    
    @patch('core.data.data_loaders.sa.create_engine')
    def test_impala_init_with_kerberos(self, mock_create_engine):
        """Тест инициализации ImpalaDataLoader с Kerberos"""
        # Настройка мока
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Создание загрузчика с Kerberos
        connection_params = {
            'host': 'impala-host',
            'port': 21050,
            'database': 'test_db',
            'use_ssl': 'true',
            'auth_mechanism': 'GSSAPI'
        }
        
        loader = ImpalaDataLoader(connection_params)
        
        # Проверка создания engine с Kerberos
        mock_create_engine.assert_called_once()
        args, kwargs = mock_create_engine.call_args
        assert args[0] == "impala://impala-host:21050/test_db"
        assert 'connect_args' in kwargs
        assert kwargs['connect_args']['auth_mechanism'] == 'GSSAPI'
        assert kwargs['connect_args']['use_ssl'] == 'true'
    
    @patch('core.data.data_loaders.sa.create_engine')
    def test_impala_init_with_plain_auth(self, mock_create_engine):
        """Тест инициализации ImpalaDataLoader с PLAIN"""
        # Настройка мока
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Создание загрузчика с PLAIN
        connection_params = {
            'host': 'impala-host',
            'port': 21050,
            'database': 'test_db',
            'auth_mechanism': 'PLAIN',
            'user': 'test_user',
            'password': 'test_password'
        }
        
        loader = ImpalaDataLoader(connection_params)
        
        # Проверка создания engine с PLAIN
        mock_create_engine.assert_called_once()
        args, kwargs = mock_create_engine.call_args
        assert args[0] == "impala://impala-host:21050/test_db"
        assert 'connect_args' in kwargs
        assert kwargs['connect_args']['auth_mechanism'] == 'PLAIN'
        assert kwargs['connect_args']['user'] == 'test_user'
        assert kwargs['connect_args']['password'] == 'test_password'
    
    @patch('core.data.data_loaders.sa.create_engine')
    @patch('core.data.data_loaders.pd.read_sql')
    def test_impala_load_data(self, mock_read_sql, mock_create_engine):
        """Тест загрузки данных из Impala"""
        # Настройка моков
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Мок для DataFrame
        df = pd.DataFrame({
            'col1': np.array([1, 2, 3], dtype=np.uint8),
            'col2': ['a', 'b', 'c']
        })
        mock_read_sql.return_value = df
        
        # Создание загрузчика и загрузка данных
        loader = ImpalaDataLoader({
            'host': 'impala-host',
            'port': 21050,
            'database': 'test_db',
            'auth_mechanism': 'GSSAPI'
        })
        
        # При вызове load_data отключаем оптимизацию типов
        result = loader.load_data("SELECT * FROM test_table", optimize_types=False)
        
        # Проверка вызова read_sql
        mock_read_sql.assert_called_once_with("SELECT * FROM test_table", mock_engine)
        
        # Проверка результата
        pd.testing.assert_frame_equal(result, df)
    
    @patch('core.data.data_loaders.sa.create_engine')
    @patch('core.data.data_loaders.pd.read_sql')
    @patch('core.data.data_loaders.os.path.exists')
    @patch('core.data.data_loaders.DataSchema')
    @patch('core.data.data_loaders.DataLoader.get_schema_path')
    def test_impala_load_data_with_schema(self, mock_get_schema_path, mock_schema_class, mock_exists, mock_read_sql, mock_create_engine):
        """Тест загрузки данных из Impala с схемой данных"""
        # Настройка моков
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Мок для DataFrame
        df = pd.DataFrame({
            'col1': np.array([1, 2, 3], dtype=np.uint8),
            'col2': ['a', 'b', 'c']
        })
        mock_read_sql.return_value = df
        
        # Настройка мока для пути к схеме
        mock_schema_path = 'mock_schema_path.json'
        mock_get_schema_path.return_value = mock_schema_path
        
        # Мок для схемы данных
        mock_exists.return_value = True
        mock_schema = MagicMock()
        mock_schema_class.load.return_value = mock_schema
        
        # Заменяем загрузку схемы из файла на возврат мока
        with patch('core.data.base.DataSchema.from_file', return_value=mock_schema):
            # Создание загрузчика и загрузка данных с схемой
            loader = ImpalaDataLoader({
                'host': 'impala-host',
                'port': 21050,
                'database': 'test_db',
                'auth_mechanism': 'GSSAPI'
            })
            
            result = loader.load_data("SELECT * FROM test_table", schema_name='test_schema', optimize_types=False)
            
            # Проверка вызова методов для схемы данных
            assert mock_exists.called
            mock_get_schema_path.assert_called_once_with('test_schema')
            
            # Проверка вызова read_sql
            mock_read_sql.assert_called_once_with("SELECT * FROM test_table", mock_engine)
            
            # Проверка результата
            pd.testing.assert_frame_equal(result, df) 