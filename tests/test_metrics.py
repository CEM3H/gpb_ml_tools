"""
Тесты для модуля расчета метрик.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from core.metrics.classification import (
    calculate_basic_metrics, calculate_cross_val_metrics,
    generate_confusion_matrix, generate_classification_report
)
from core.metrics.regression import (
    calculate_basic_metrics as calculate_regression_metrics,
    calculate_cross_val_metrics as calculate_regression_cv_metrics,
    calculate_residuals
)
from core.models.model_types import CatBoostModel


@pytest.fixture
def classification_data():
    """Генерация данных для задачи классификации."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, 
        n_redundant=2, random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def classification_predictions():
    """Предсказания для задачи классификации."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)
    y_pred_proba = np.random.rand(100)
    return y_true, y_pred, y_pred_proba


@pytest.fixture
def regression_predictions():
    """Предсказания для задачи регрессии."""
    np.random.seed(42)
    y_true = np.random.normal(0, 1, size=100)
    y_pred = y_true + np.random.normal(0, 0.2, size=100)  # добавляем некоторый шум
    return y_true, y_pred


def test_classification_basic_metrics(classification_predictions):
    """Тест расчета базовых метрик классификации."""
    y_true, y_pred, y_pred_proba = classification_predictions
    
    # Расчет метрик без вероятностей
    metrics = calculate_basic_metrics(y_true, y_pred)
    
    # Проверяем, что в результате есть ожидаемые метрики
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    
    # Проверяем, что метрики имеют разумные значения
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    
    # Проверяем, что метрики, требующие вероятностей, отсутствуют
    assert 'roc_auc' not in metrics
    assert 'average_precision' not in metrics
    
    # Расчет метрик с вероятностями
    metrics_with_proba = calculate_basic_metrics(y_true, y_pred, y_pred_proba)
    
    # Проверяем наличие дополнительных метрик
    assert 'roc_auc' in metrics_with_proba
    assert 'average_precision' in metrics_with_proba
    
    # Проверяем, что значения разумные
    assert 0 <= metrics_with_proba['roc_auc'] <= 1
    assert 0 <= metrics_with_proba['average_precision'] <= 1


def test_regression_basic_metrics(regression_predictions):
    """Тест расчета базовых метрик регрессии."""
    y_true, y_pred = regression_predictions
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    # Проверяем, что в результате есть ожидаемые метрики
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'median_ae' in metrics
    assert 'explained_variance' in metrics
    
    # Проверяем, что метрики имеют разумные значения для регрессии
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['median_ae'] >= 0
    assert metrics['r2'] <= 1  # R2 может быть отрицательным для плохих моделей


def test_cross_val_metrics(classification_data):
    """Тест расчета метрик с использованием кросс-валидации."""
    X, y = classification_data
    
    # Создаем модель с использованием sklearn, который правильно определяется как классификатор
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Рассчитываем метрики с кросс-валидацией
    cv_metrics = calculate_cross_val_metrics(model, X, y, cv=3, metric='roc_auc')
    
    # Проверяем, что в результате есть ожидаемые метрики
    assert 'cv_roc_auc_mean' in cv_metrics
    assert 'cv_roc_auc_std' in cv_metrics
    
    # Проверяем, что значения разумные
    assert 0 <= cv_metrics['cv_roc_auc_mean'] <= 1
    assert cv_metrics['cv_roc_auc_std'] >= 0


def test_generate_confusion_matrix(classification_predictions):
    """Тест генерации матрицы ошибок."""
    y_true, y_pred, _ = classification_predictions
    
    # Генерируем матрицу ошибок
    cm, labels = generate_confusion_matrix(y_true, y_pred)
    
    # Проверяем, что матрица имеет правильный размер
    assert cm.shape == (len(labels), len(labels))
    
    # Проверяем, что сумма элементов матрицы равна количеству предсказаний
    assert cm.sum() == len(y_true)
    
    # Проверяем, что все значения неотрицательны
    assert np.all(cm >= 0)


def test_classification_report(classification_predictions):
    """Тест генерации отчета о классификации."""
    y_true, y_pred, _ = classification_predictions
    
    # Генерируем отчет
    report = generate_classification_report(y_true, y_pred)
    
    # Проверяем, что отчет содержит ожидаемые ключи
    assert '0' in report  # метрики для класса 0
    assert '1' in report  # метрики для класса 1
    assert 'accuracy' in report
    assert 'macro avg' in report
    assert 'weighted avg' in report
    
    # Проверяем, что метрики для классов содержат ожидаемые поля
    assert 'precision' in report['0']
    assert 'recall' in report['0']
    assert 'f1-score' in report['0']
    assert 'support' in report['0']
    
    # Проверяем, что значения разумные
    assert 0 <= report['0']['precision'] <= 1
    assert 0 <= report['0']['recall'] <= 1
    assert 0 <= report['0']['f1-score'] <= 1
    assert report['0']['support'] > 0


def test_calculate_residuals(regression_predictions):
    """Тест расчета остатков для регрессии."""
    y_true, y_pred = regression_predictions
    
    # Рассчитываем остатки
    residuals = calculate_residuals(y_true, y_pred)
    
    # Проверяем, что размерность остатков совпадает с размерностью предсказаний
    assert len(residuals) == len(y_true)
    
    # Проверяем, что остатки рассчитаны правильно
    np.testing.assert_array_almost_equal(residuals, y_true - y_pred) 