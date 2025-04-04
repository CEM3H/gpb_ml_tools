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

Примеры использования
-------------------

Сериализация модели и пайплайна
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.utils import save_model, load_model
    from core.models import RandomForestModel
    from core.preprocessing import StandardScaler
    from core.feature_selection import CorrelationFeatureSelector
    from sklearn.pipeline import Pipeline

    # Создаем модель
    model = RandomForestModel(n_estimators=100, random_state=42)
    
    # Создаем пайплайн
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', CorrelationFeatureSelector(threshold=0.8)),
        ('model', model)
    ])
    
    # Обучаем пайплайн
    pipeline.fit(X_train, y_train)
    
    # Сохраняем модель и пайплайн
    save_model(model, 'random_forest_model.pkl')
    save_model(pipeline, 'preprocessing_pipeline.pkl')
    
    # Загружаем модель и пайплайн
    loaded_model = load_model('random_forest_model.pkl')
    loaded_pipeline = load_model('preprocessing_pipeline.pkl')
    
    # Делаем предсказания
    predictions = loaded_pipeline.predict(X_test)

Сериализация метаданных
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.utils import save_metadata, load_metadata
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
    save_metadata(metadata, 'model_metadata.json')
    
    # Загружаем метаданные
    loaded_metadata = load_metadata('model_metadata.json')
    
    # Используем метаданные
    features = loaded_metadata['features']
    training_date = pd.Timestamp(loaded_metadata['training_date']) 