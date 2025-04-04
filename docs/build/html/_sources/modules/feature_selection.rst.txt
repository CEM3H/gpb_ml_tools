Модуль feature_selection
======================

Модуль для отбора признаков.

Основные классы
--------------

FeatureSelector
~~~~~~~~~~~~~~

Базовый класс для отбора признаков.

.. autoclass:: core.feature_selection.base.FeatureSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Selectors
~~~~~~~~~

Реализации селекторов признаков.

.. automodule:: core.feature_selection.selectors
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Примеры использования
-------------------

Отбор признаков на основе случайного леса
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.feature_selection.selectors import RandomForestFeatureSelector

    # Создание селектора
    selector = RandomForestFeatureSelector(
        n_estimators=100,
        max_depth=5,
        n_features=10
    )

    # Отбор признаков
    selected_features = selector.fit_transform(X, y)
    print(f"Отобранные признаки: {selector.get_feature_names()}")

Отбор признаков на основе корреляций
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.feature_selection.selectors import CorrelationFeatureSelector

    # Создание селектора
    selector = CorrelationFeatureSelector(
        threshold=0.8,
        method='pearson'
    )

    # Отбор признаков
    selected_features = selector.fit_transform(X)
    print(f"Отобранные признаки: {selector.get_feature_names()}") 