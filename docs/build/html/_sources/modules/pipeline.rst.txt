Модуль pipeline
==============

Модуль пайплайна предоставляет инструменты для управления процессом разработки
моделей машинного обучения. Включает загрузку данных, предобработку, отбор признаков,
обучение, валидацию и создание отчетов.

Основные классы
-------------

ModelPipeline
~~~~~~~~~~~

.. autoclass:: core.pipeline.ModelPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Примеры использования
------------------

Создание и запуск пайплайна
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.pipeline import ModelPipeline
    from core.container import ModelMetadata
    from core.data.base import DataLoader
    from core.preprocessing.preprocessors import DefaultPreprocessor
    from core.feature_selection.selectors import RandomForestFeatureSelector
    from core.models.model_types import CatBoostModel
    from core.validation.base import ModelValidator
    
    # Создаем компоненты пайплайна
    data_loader = DataLoader('path/to/data.csv')
    preprocessor = DefaultPreprocessor(
        numeric_features=['age', 'income'],
        categorical_features=['gender', 'education']
    )
    feature_selector = RandomForestFeatureSelector(threshold=0.1)
    model = CatBoostModel({
        'iterations': 1000,
        'learning_rate': 0.1
    })
    validator = ModelValidator()
    
    # Создаем пайплайн
    pipeline = ModelPipeline(
        data_loader=data_loader,
        preprocessor=preprocessor,
        feature_selector=feature_selector,
        model=model,
        validator=validator
    )
    
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
        created_at='2024-04-01'
    )
    
    # Запускаем пайплайн
    container = pipeline.run_pipeline(metadata)
    
    # Генерируем отчет
    pipeline.generate_report('model_report.docx') 