Модуль metrics
=============

Модуль для расчета метрик качества моделей.

Функции для классификации
----------------------

Функции и классы для оценки качества классификационных моделей.

.. automodule:: core.metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Функции для регрессии
------------------

Функции и классы для оценки качества регрессионных моделей.

Примеры использования
-------------------

Оценка качества классификации
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.metrics import classification_report, roc_auc_score

    # Создаем предсказания модели
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0, 0, 1]
    y_prob = [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], 
              [0.7, 0.3], [0.4, 0.6], [0.9, 0.1], [0.1, 0.9]]

    # Вычисление метрик качества
    report = classification_report(y_true, y_pred)
    print(report)

    # Вычисление ROC AUC
    auc = roc_auc_score(y_true, [prob[1] for prob in y_prob])
    print(f"ROC AUC: {auc:.4f}")

    # Построение ROC-кривой
    from core.metrics import plot_roc_curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plot_roc_curve(y_true, [prob[1] for prob in y_prob])
    plt.title('ROC Curve')
    plt.show()

Оценка качества регрессии
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.metrics import mean_squared_error, r2_score

    # Создаем предсказания модели
    y_true = [3.1, 2.7, 5.6, 7.2, 4.5, 8.1, 9.3, 2.2]
    y_pred = [2.9, 2.8, 5.2, 7.5, 4.8, 7.9, 9.1, 2.5]

    # Вычисление метрик качества
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"R^2: {r2:.4f}")

    # Визуализация результатов
    from core.metrics import plot_regression_results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plot_regression_results(y_true, y_pred)
    plt.title('Regression Results')
    plt.show() 