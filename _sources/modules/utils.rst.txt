Модуль utils
===========

Модуль с утилитами и вспомогательными функциями.

Основные классы
--------------

NumpyEncoder
~~~~~~~~~~

JSON энкодер для работы с NumPy объектами.

.. automodule:: core.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Функции сериализации
-----------------

Функции для сохранения и загрузки моделей и связанных с ними данных.

Функции для работы с файлами
--------------------------

Функции для безопасной работы с файлами.

.. code-block:: python

    from core.utils.file_utils import safe_path_join, validate_file_extension, get_safe_filename

    # Безопасное объединение путей
    base_dir = '/path/to/app/data'
    file_path = safe_path_join(base_dir, 'reports', 'report.csv')
    
    # Проверка расширения файла
    is_allowed = validate_file_extension('data.csv', ['.csv', '.xlsx'])
    
    # Получение безопасного имени файла
    safe_name = get_safe_filename('user-input../file!name.csv')
    # Результат: 'user-input__file_name.csv'

Примеры использования
-------------------

Сериализация модели и пайплайна
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.utils.serialization import serialize_model, deserialize_model
    from core.models import RandomForestModel
    from core.preprocessing import StandardScaler
    from core.feature_selection import CorrelationFeatureSelector
    from sklearn.pipeline import Pipeline

    # Создаем модель
    model = RandomForestModel(hyperparameters={'n_estimators': 100, 'random_state': 42})
    
    # Создаем пайплайн
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', CorrelationFeatureSelector(threshold=0.8)),
        ('model', model)
    ])
    
    # Обучаем пайплайн
    pipeline.fit(X_train, y_train)
    
    # Сохраняем модель и пайплайн
    with open('random_forest_model.pkl', 'wb') as f:
        f.write(serialize_model(model))
    
    with open('preprocessing_pipeline.pkl', 'wb') as f:
        f.write(serialize_pipeline(pipeline))
    
    # Загружаем модель и пайплайн
    with open('random_forest_model.pkl', 'rb') as f:
        loaded_model = deserialize_model(f.read())
    
    with open('preprocessing_pipeline.pkl', 'rb') as f:
        loaded_pipeline = deserialize_pipeline(f.read())
    
    # Делаем предсказания
    predictions = loaded_pipeline.predict(X_test)

Сериализация метаданных
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.utils.serialization import serialize_metadata, deserialize_metadata
    import pandas as pd
    
    # Создаем метаданные
    metadata = {
        'features': X_train.columns.tolist(),
        'target': 'target_column',
        'model_params': model.get_params(),
        'feature_importance': model.feature_importances_.tolist(),
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    # Сохраняем метаданные
    with open('model_metadata.json', 'w') as f:
        f.write(serialize_metadata(metadata))
    
    # Загружаем метаданные
    with open('model_metadata.json', 'r') as f:
        loaded_metadata = deserialize_metadata(f.read())
    
    # Используем метаданные
    features = loaded_metadata['features']
    training_date = pd.Timestamp(loaded_metadata['training_date']) 