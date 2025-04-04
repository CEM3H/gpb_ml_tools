Модуль models
============

Модуль для работы с моделями машинного обучения предоставляет унифицированный интерфейс
для различных типов моделей, включая Random Forest и CatBoost. Поддерживает оптимизацию
гиперпараметров и оценку важности признаков.

Основные классы
--------------

BaseModel
~~~~~~~~

Базовый класс для всех моделей.

.. automodule:: core.models
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

RandomForestModel
~~~~~~~~~~~~~~

Модель на основе случайного леса.

CatBoostModel
~~~~~~~~~~

Модель на основе CatBoost.

Примеры использования
-------------------

Обучение Random Forest модели
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.models import RandomForestModel

    # Создание модели
    model = RandomForestModel(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # Обучение
    model.fit(X_train, y_train)

    # Предсказание
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Оценка качества
    from core.metrics import classification_report
    report = classification_report(y_test, y_pred)
    print(report)

Обучение CatBoost модели
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.models import CatBoostModel

    # Создание модели
    model = CatBoostModel(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        random_seed=42
    )

    # Обучение
    model.fit(X_train, y_train, 
              cat_features=['category1', 'category2'])

    # Предсказание
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Оценка качества
    from core.metrics import classification_report
    report = classification_report(y_test, y_pred)
    print(report) 