Модуль container
==============

Модуль контейнера предоставляет классы для хранения моделей машинного обучения
и связанных с ними артефактов. Включает метаданные модели, пайплайн предобработки,
важность признаков и метрики качества.

Основные классы
-------------

ModelMetadata
~~~~~~~~~~~

.. autoclass:: core.container.ModelMetadata
   :members:
   :undoc-members:
   :show-inheritance:

ModelContainer
~~~~~~~~~~~

.. autoclass:: core.container.ModelContainer
   :members:
   :undoc-members:
   :show-inheritance:

Примеры использования
------------------

Создание и сохранение контейнера
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.container import ModelContainer, ModelMetadata
    from datetime import datetime
    
    # Создаем метаданные
    metadata = ModelMetadata(
        model_id='model_001',
        model_name='Customer Churn Model',
        author='Data Science Team',
        target_description='Вероятность оттока клиента',
        features_description={
            'age': 'Возраст клиента',
            'income': 'Доход клиента',
            'gender': 'Пол клиента',
            'education': 'Уровень образования'
        },
        train_tables=['customers.csv'],
        created_at=datetime.now().strftime('%Y-%m-%d')
    )
    
    # Создаем контейнер
    container = ModelContainer(metadata)
    
    # Добавляем модель и артефакты
    container.model = trained_model
    container.preprocessing_pipeline = preprocessing_pipeline
    container.feature_importance = feature_importance
    container.metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.87,
        'f1': 0.84
    }
    
    # Сохраняем контейнер
    container.save('model_container.pkl')

Загрузка контейнера
~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.container import ModelContainer
    
    # Загружаем контейнер
    container = ModelContainer.load('model_container.pkl')
    
    # Используем модель для предсказаний
    predictions = container.model.predict(X_test)
    
    # Получаем метаданные
    print(f"Модель: {container.metadata.model_name}")
    print(f"Автор: {container.metadata.author}")
    print(f"Метрики: {container.metrics}") 