"""
Тесты для модуля генерации отчетов.
"""

import pytest
import numpy as np
import pandas as pd
import os
import io
from datetime import datetime
from PIL import Image

from core.reporting.word import WordReportGenerator
from core.reporting.utils import (
    plot_roc_curve, plot_precision_recall_curve, 
    plot_confusion_matrix_heatmap, plot_feature_importance,
    plot_residuals
)
from core.container import ModelContainer, ModelMetadata
from core.models.model_types import CatBoostModel
from sklearn.datasets import make_classification


@pytest.fixture
def test_data():
    """Генерация тестовых данных."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, 
        n_redundant=1, n_classes=2, random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def test_model(test_data):
    """Создание тестовой модели."""
    X, y = test_data
    model = CatBoostModel({'iterations': 10, 'verbose': False})
    model.fit(X, y)
    return model


@pytest.fixture
def test_container(test_model, test_data):
    """Создание тестового контейнера модели."""
    X, y = test_data
    
    # Создаем метаданные
    metadata = ModelMetadata(
        model_id="test_model_id",
        model_name="Test Model",
        author="Test Author",
        target_description="Binary classification",
        features_description={f'feature_{i}': f'Feature {i}' for i in range(5)},
        train_tables=["test_table"],
        created_at=datetime.now().strftime("%Y-%m-%d")
    )
    
    # Создаем контейнер
    container = ModelContainer(metadata)
    container.model = test_model
    
    # Добавляем метрики
    container.metrics = {
        'accuracy': 0.85,
        'f1': 0.84,
        'precision': 0.83,
        'recall': 0.82,
        'roc_auc': 0.9,
        'cv_roc_auc_mean': 0.88,
        'cv_roc_auc_std': 0.02
    }
    
    # Добавляем важность признаков
    container.feature_importance = {
        f'feature_{i}': np.random.rand() for i in range(5)
    }
    
    return container


def test_plot_roc_curve(test_data):
    """Тест функции построения ROC-кривой."""
    X, y = test_data
    y_pred_proba = np.random.rand(len(y))
    
    # Генерируем ROC-кривую
    img_stream = plot_roc_curve(y, y_pred_proba)
    
    # Проверяем, что результат - это поток с изображением
    assert isinstance(img_stream, io.BytesIO)
    
    # Пробуем открыть изображение
    img = Image.open(img_stream)
    assert img.width > 0
    assert img.height > 0


def test_plot_precision_recall_curve(test_data):
    """Тест функции построения precision-recall кривой."""
    X, y = test_data
    y_pred_proba = np.random.rand(len(y))
    
    # Генерируем precision-recall кривую
    img_stream = plot_precision_recall_curve(y, y_pred_proba)
    
    # Проверяем, что результат - это поток с изображением
    assert isinstance(img_stream, io.BytesIO)
    
    # Пробуем открыть изображение
    img = Image.open(img_stream)
    assert img.width > 0
    assert img.height > 0


def test_plot_confusion_matrix_heatmap(test_data):
    """Тест функции построения матрицы ошибок."""
    X, y = test_data
    y_pred = np.random.randint(0, 2, size=len(y))
    
    # Генерируем матрицу ошибок
    img_stream = plot_confusion_matrix_heatmap(y, y_pred)
    
    # Проверяем, что результат - это поток с изображением
    assert isinstance(img_stream, io.BytesIO)
    
    # Пробуем открыть изображение
    img = Image.open(img_stream)
    assert img.width > 0
    assert img.height > 0


def test_plot_feature_importance(test_container):
    """Тест функции построения графика важности признаков."""
    # Генерируем график важности признаков
    img_stream = plot_feature_importance(test_container.feature_importance)
    
    # Проверяем, что результат - это поток с изображением
    assert isinstance(img_stream, io.BytesIO)
    
    # Пробуем открыть изображение
    img = Image.open(img_stream)
    assert img.width > 0
    assert img.height > 0


def test_word_report_generator_init(test_container):
    """Тест инициализации генератора отчетов."""
    # Создаем генератор отчетов
    generator = WordReportGenerator(test_container)
    
    # Проверяем, что генератор создан корректно
    assert generator.container == test_container
    assert generator.document is not None


def test_word_report_generator_add_title(test_container):
    """Тест добавления заголовка в отчет."""
    generator = WordReportGenerator(test_container)
    generator.add_title("Test Title")
    
    # Проверяем, что в документе есть заголовок
    assert len(generator.document.paragraphs) > 1
    assert "Test Title" in generator.document.paragraphs[0].text


def test_word_report_generator_add_section(test_container):
    """Тест добавления раздела в отчет."""
    generator = WordReportGenerator(test_container)
    generator.add_section("Test Section", level=1)
    
    # Проверяем, что в документе есть раздел
    assert "Test Section" in generator.document.paragraphs[0].text


def test_word_report_generator_add_table(test_container):
    """Тест добавления таблицы в отчет."""
    generator = WordReportGenerator(test_container)
    
    # Добавляем таблицу
    headers = ["Column 1", "Column 2"]
    data = [["Value 1", "Value 2"], ["Value 3", "Value 4"]]
    generator.add_table(data, headers, title="Test Table")
    
    # Проверяем, что в документе есть таблица
    assert len(generator.document.tables) == 1
    assert generator.document.tables[0].rows[0].cells[0].text == "Column 1"
    assert generator.document.tables[0].rows[0].cells[1].text == "Column 2"


def test_word_report_generator_generate_report(test_container, test_data, tmp_path):
    """Тест генерации полного отчета."""
    X, y = test_data
    generator = WordReportGenerator(test_container)
    
    # Генерируем отчет
    output_path = os.path.join(tmp_path, "test_report.docx")
    report_path = generator.generate_report(X, y, output_path)
    
    # Проверяем, что отчет создан
    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 0 