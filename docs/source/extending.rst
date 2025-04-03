Расширение библиотеки
===================

Создание собственного загрузчика данных
------------------------------------

Для создания собственного загрузчика данных необходимо наследоваться от базового класса ``DataLoader``:

.. code-block:: python

    from core.data_loading.base import DataLoader
    
    class CustomDataLoader(DataLoader):
        """
        Пользовательский загрузчик данных.
        
        Parameters
        ----------
        param1 : str
            Описание параметра 1
        param2 : int, optional
            Описание параметра 2, по умолчанию 42
        """
        
        def __init__(self, param1: str, param2: int = 42):
            super().__init__()
            self.param1 = param1
            self.param2 = param2
        
        def load_data(self, source: str, **kwargs) -> pd.DataFrame:
            """
            Загрузка данных из источника.
            
            Parameters
            ----------
            source : str
                Путь к источнику данных
            **kwargs
                Дополнительные параметры
            
            Returns
            -------
            pd.DataFrame
                Загруженные данные
            
            Raises
            ------
            ValueError
                Если источник недоступен
            """
            # Реализация загрузки данных
            pass

Создание собственного валидатора
-----------------------------

Для создания собственного валидатора необходимо наследоваться от базового класса ``SchemaValidator``:

.. code-block:: python

    from core.data_validation.base import SchemaValidator
    from core.data.schema import ColumnSchema
    
    class CustomValidator(SchemaValidator):
        """
        Пользовательский валидатор данных.
        
        Parameters
        ----------
        param1 : str
            Описание параметра 1
        param2 : int, optional
            Описание параметра 2, по умолчанию 42
        """
        
        def __init__(self, param1: str, param2: int = 42):
            super().__init__()
            self.param1 = param1
            self.param2 = param2
        
        def _validate_custom(self, series: pd.Series, schema: ColumnSchema) -> None:
            """
            Валидация пользовательского типа данных.
            
            Parameters
            ----------
            series : pd.Series
                Серия данных для валидации
            schema : ColumnSchema
                Схема колонки
            
            Raises
            ------
            ValueError
                Если данные не соответствуют схеме
            """
            # Реализация валидации
            pass

Создание собственного селектора признаков
--------------------------------------

Для создания собственного селектора признаков необходимо наследоваться от базового класса ``FeatureSelector``:

.. code-block:: python

    from core.feature_selection.base import FeatureSelector
    
    class CustomFeatureSelector(FeatureSelector):
        """
        Пользовательский селектор признаков.
        
        Parameters
        ----------
        param1 : str
            Описание параметра 1
        param2 : int, optional
            Описание параметра 2, по умолчанию 42
        """
        
        def __init__(self, param1: str, param2: int = 42):
            super().__init__()
            self.param1 = param1
            self.param2 = param2
        
        def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
            """
            Отбор признаков.
            
            Parameters
            ----------
            X : pd.DataFrame
                Матрица признаков
            y : pd.Series
                Вектор целевой переменной
            
            Returns
            -------
            List[str]
                Список отобранных признаков
            """
            # Реализация отбора признаков
            pass

Создание собственного препроцессора
--------------------------------

Для создания собственного препроцессора необходимо наследоваться от базового класса ``Preprocessor``:

.. code-block:: python

    from core.preprocessing.base import Preprocessor
    
    class CustomPreprocessor(Preprocessor):
        """
        Пользовательский препроцессор данных.
        
        Parameters
        ----------
        param1 : str
            Описание параметра 1
        param2 : int, optional
            Описание параметра 2, по умолчанию 42
        """
        
        def __init__(self, param1: str, param2: int = 42):
            super().__init__()
            self.param1 = param1
            self.param2 = param2
        
        def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CustomPreprocessor':
            """
            Обучение препроцессора.
            
            Parameters
            ----------
            X : pd.DataFrame
                Матрица признаков
            y : pd.Series, optional
                Вектор целевой переменной
            
            Returns
            -------
            CustomPreprocessor
                Обученный препроцессор
            """
            # Реализация обучения
            return self
        
        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            """
            Применение преобразований.
            
            Parameters
            ----------
            X : pd.DataFrame
                Матрица признаков
            
            Returns
            -------
            pd.DataFrame
                Преобразованная матрица признаков
            """
            # Реализация преобразований
            pass

Регистрация пользовательских компонентов
-------------------------------------

Для регистрации пользовательских компонентов необходимо добавить их в соответствующие модули:

.. code-block:: python

    # В файле core/data_loading/__init__.py
    from .custom_loader import CustomDataLoader
    
    __all__ = [
        'DataLoader',
        'FileDataLoader',
        'PostgresDataLoader',
        'ImpalaDataLoader',
        'CustomDataLoader'  # Добавляем пользовательский загрузчик
    ]
    
    # В файле core/data_validation/__init__.py
    from .custom_validator import CustomValidator
    
    __all__ = [
        'SchemaValidator',
        'CustomValidator'  # Добавляем пользовательский валидатор
    ]
    
    # В файле core/feature_selection/__init__.py
    from .custom_selector import CustomFeatureSelector
    
    __all__ = [
        'FeatureSelector',
        'CustomFeatureSelector'  # Добавляем пользовательский селектор
    ]
    
    # В файле core/preprocessing/__init__.py
    from .custom_preprocessor import CustomPreprocessor
    
    __all__ = [
        'Preprocessor',
        'CustomPreprocessor'  # Добавляем пользовательский препроцессор
    ]

Пример использования пользовательских компонентов
---------------------------------------------

.. code-block:: python

    from core.data_loading import CustomDataLoader
    from core.data_validation import CustomValidator
    from core.feature_selection import CustomFeatureSelector
    from core.preprocessing import CustomPreprocessor
    
    # Создание пользовательских компонентов
    loader = CustomDataLoader(param1='value1')
    validator = CustomValidator(param1='value2')
    selector = CustomFeatureSelector(param1='value3')
    preprocessor = CustomPreprocessor(param1='value4')
    
    # Использование в пайплайне
    df = loader.load_data('data.csv')
    validator.validate(df)
    X_selected = selector.select_features(X, y)
    X_processed = preprocessor.fit_transform(X_selected) 