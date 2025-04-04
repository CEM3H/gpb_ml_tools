Модуль reporting
==============

Модуль для создания отчетов и визуализации результатов моделей.

Основные классы
--------------

WordReportGenerator
~~~~~~~~~~~~~~~~

Генератор отчетов в формате Word.

.. automodule:: core.reporting
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Вспомогательные функции
--------------------

Функции для создания диаграмм, графиков и форматирования данных.

Примеры использования
-------------------

Создание отчета о модели
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.reporting import WordReportGenerator
    from core.models import RandomForestModel
    import pandas as pd

    # Обучаем модель
    model = RandomForestModel(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Получаем предсказания
    y_pred = model.predict(X_test)
    
    # Создаем отчет
    report_generator = WordReportGenerator(
        title="Отчет о модели классификации",
        author="Аналитик"
    )
    
    # Добавляем разделы отчета
    report_generator.add_heading("Описание модели", level=1)
    report_generator.add_paragraph("Модель случайного леса с 100 деревьями.")
    
    report_generator.add_heading("Метрики качества", level=1)
    
    # Добавляем таблицу метрик
    from core.metrics import classification_report
    metrics = classification_report(y_test, y_pred, output_dict=True)
    
    # Преобразуем метрики в DataFrame
    metrics_df = pd.DataFrame(metrics).transpose()
    report_generator.add_table(metrics_df.reset_index().rename(columns={'index': 'Метрика'}))
    
    # Добавляем графики
    report_generator.add_heading("Графики", level=1)
    
    # Построение ROC-кривой
    from core.metrics import plot_roc_curve
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.title('ROC Curve')
    
    # Добавляем график в отчет
    report_generator.add_figure(plt)
    
    # Сохраняем отчет
    report_generator.save("model_report.docx")

Создание графиков для отчета
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from core.reporting import plot_feature_importance, plot_confusion_matrix
    import matplotlib.pyplot as plt
    
    # Получаем важности признаков из модели
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Строим график важности признаков
    plt.figure(figsize=(10, 6))
    plot_feature_importance(feature_importance, feature_names, top_n=10)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    # Строим матрицу ошибок
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(y_test, y_pred, labels=['Класс 0', 'Класс 1'])
    plt.tight_layout()
    plt.savefig("confusion_matrix.png") 