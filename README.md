# GPB Library

Библиотека для разработки и деплоя ML-моделей, включающая функционал для:
- Загрузки данных из различных источников (файлы, PostgreSQL, Impala)
- Валидации данных по схеме
- Управления схемами данных
- Предобработки данных
- Отбора признаков
- Обучения и валидации моделей
- Генерации отчетов

## Структура проекта

```
gpb_lib/
├── core/                  # Основной код библиотеки
│   ├── data/             # Загрузка данных и схемы данных
│   │   ├── schema.py     # Работа со схемами данных
│   │   ├── data_types.py # Типы данных и их обработка
│   │   ├── data_loaders.py # Загрузчики данных
│   │   └── base.py       # Базовые классы
│   ├── models/           # Реализации моделей
│   │   ├── model_types.py # Типы моделей (CatBoost, LightGBM и др.)
│   │   └── base.py       # Базовый класс модели
│   ├── preprocessing/    # Предобработка данных
│   │   ├── preprocessors.py # Реализации препроцессоров
│   │   └── base.py       # Базовый класс препроцессора
│   ├── feature_selection/ # Отбор признаков
│   │   ├── selectors.py  # Реализации селекторов
│   │   └── base.py       # Базовый класс селектора
│   ├── validation/       # Валидация моделей
│   │   ├── validators.py # Реализации валидаторов
│   │   └── base.py       # Базовый класс валидатора
│   ├── metrics/          # Метрики и оценка моделей
│   │   ├── metrics.py    # Реализации метрик
│   │   └── base.py       # Базовый класс метрики
│   ├── reporting/        # Генерация отчетов
│   │   ├── reporters.py  # Реализации генераторов отчетов
│   │   └── base.py       # Базовый класс репортера
│   ├── utils/            # Утилиты и вспомогательные функции
│   │   ├── config.py     # Работа с конфигурациями
│   │   ├── logging.py    # Настройка логирования
│   │   └── exceptions.py # Пользовательские исключения
│   └── pipeline.py       # Основной пайплайн
├── notebooks/            # Jupyter-ноутбуки с примерами
│   ├── 01_data_loading/  # Загрузка данных
│   │   ├── 01_file_loading.ipynb
│   │   ├── 02_postgres_loading.ipynb
│   │   └── 03_impala_loading.ipynb
│   ├── 02_data_validation/ # Валидация данных
│   │   ├── 01_schema_validation.ipynb
│   │   └── 02_custom_constraints.ipynb
│   ├── 03_preprocessing/ # Предобработка данных
│   │   ├── 01_basic_preprocessing.ipynb
│   │   └── 02_advanced_preprocessing.ipynb
│   ├── 04_feature_selection/ # Отбор признаков
│   │   ├── 01_random_forest_selector.ipynb
│   │   └── 02_correlation_selector.ipynb
│   ├── 05_model_training/ # Обучение моделей
│   │   ├── 01_catboost_training.ipynb
│   │   └── 02_lightgbm_training.ipynb
│   ├── 06_model_validation/ # Валидация моделей
│   │   ├── 01_cross_validation.ipynb
│   │   └── 02_hyperparameter_tuning.ipynb
│   └── 07_reporting/     # Генерация отчетов
│       ├── 01_basic_reporting.ipynb
│       └── 02_advanced_reporting.ipynb
├── tests/                # Тесты проекта
│   ├── test_data/        # Тесты для работы с данными
│   ├── test_models/      # Тесты для моделей
│   ├── test_preprocessing/ # Тесты для предобработки
│   ├── test_validation/  # Тесты для валидации
│   └── test_pipeline/    # Тесты для пайплайна
├── schemas/              # Предопределенные схемы данных
│   ├── user_schema.yaml  # Схема пользовательских данных
│   └── model_schema.yaml # Схема данных модели
├── examples/             # Примеры использования
│   ├── basic_usage.py    # Базовые примеры
│   └── advanced_usage.py # Продвинутые примеры
├── docs/                 # Документация
│   ├── api/             # API документация
│   └── guides/          # Руководства по использованию
├── requirements.txt      # Зависимости проекта
├── requirements-dev.txt  # Зависимости для разработки
├── setup.py             # Настройка пакета
├── .gitignore          # Игнорируемые git файлы
└── README.md            # Документация
```

## Установка

```bash
pip install -r requirements.txt
```

Для работы с Jupyter-ноутбуками дополнительно установите:
```bash
pip install jupyter notebook
```

## Использование

### Jupyter-ноутбуки

В папке `notebooks/` находятся Jupyter-ноутбуки с примерами использования всех основных компонентов библиотеки:

1. Загрузка данных:
   - Загрузка из файлов (CSV, JSON, YAML)
   - Загрузка из PostgreSQL
   - Загрузка из Impala

2. Валидация данных:
   - Валидация по схеме
   - Создание пользовательских ограничений

3. Предобработка данных:
   - Базовая предобработка
   - Продвинутые техники предобработки

4. Отбор признаков:
   - Отбор с помощью Random Forest
   - Отбор на основе корреляций

5. Обучение моделей:
   - Обучение CatBoost
   - Обучение LightGBM

6. Валидация моделей:
   - Кросс-валидация
   - Подбор гиперпараметров

7. Генерация отчетов:
   - Базовые отчеты
   - Продвинутые отчеты

Для запуска ноутбуков:
```bash
jupyter notebook notebooks/
```

### Основной пайплайн

```python
from core.pipeline import ModelPipeline
from core.models.model_types import CatBoostModel
from core.preprocessing.preprocessors import DefaultPreprocessor
from core.feature_selection.selectors import RandomForestFeatureSelector
from core.validation.validators import DefaultModelValidator
from core.data.data_loaders import PostgresDataLoader

# Инициализация компонентов
data_loader = PostgresDataLoader(
    connection_params={
        'host': 'localhost',
        'port': 5432,
        'database': 'my_db',
        'user': 'user',
        'password': 'password'
    }
)
preprocessor = DefaultPreprocessor(
    numeric_features=['age', 'income'],
    categorical_features=['gender', 'education']
)
feature_selector = RandomForestFeatureSelector(threshold=0.01)
model = CatBoostModel(hyperparameters={
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6
})
validator = DefaultModelValidator(cv=5)

# Создание и запуск пайплайна
pipeline = ModelPipeline(
    data_loader=data_loader,
    preprocessor=preprocessor,
    feature_selector=feature_selector,
    model=model,
    validator=validator
)

# Запуск пайплайна и получение контейнера модели
model_container = pipeline.run()
```

### Загрузка данных из файлов

```python
from core.data.schema import DataSchema, ColumnSchema, DataType
from core.data.data_loaders import FileDataLoader

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

# Загрузка данных
df = loader.load_data('data/users.csv')
```

### Загрузка данных из PostgreSQL

```python
from core.data.data_loaders import PostgresDataLoader

# Создание загрузчика PostgreSQL
loader = PostgresDataLoader(
    connection_params={
        'host': 'localhost',
        'port': 5432,
        'database': 'my_db',
        'user': 'user',
        'password': 'password'
    },
    schema=schema
)

# Загрузка данных
df = loader.load_data("SELECT * FROM users")
```

### Загрузка данных из Impala

```python
from core.data.data_loaders import ImpalaDataLoader

# Создание загрузчика Impala
loader = ImpalaDataLoader(
    connection_params={
        'host': 'impala.example.com',
        'port': 21050,
        'database': 'default',
        'auth_mechanism': 'GSSAPI'
    },
    schema=schema
)

# Загрузка данных
df = loader.load_data("SELECT * FROM users")
```

### Предобработка данных

```python
from core.preprocessing.preprocessors import DefaultPreprocessor

# Создание препроцессора
preprocessor = DefaultPreprocessor(
    numeric_features=['age', 'income'],
    categorical_features=['gender', 'education'],
    target_column='churn'
)

# Применение предобработки
X_train, y_train = preprocessor.fit_transform(train_data)
X_test, y_test = preprocessor.transform(test_data)
```

### Отбор признаков

```python
from core.feature_selection.selectors import RandomForestFeatureSelector

# Создание селектора признаков
selector = RandomForestFeatureSelector(
    threshold=0.01,
    n_estimators=100
)

# Отбор признаков
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

### Валидация модели

```python
from core.validation.validators import DefaultModelValidator

# Создание валидатора
validator = DefaultModelValidator(
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

# Валидация модели
results = validator.validate(model, X_train, y_train)
```

## Разработка

### Установка зависимостей для разработки

```bash
pip install -r requirements-dev.txt
```

### Запуск тестов

```bash
pytest tests/
```

## Лицензия

MIT 