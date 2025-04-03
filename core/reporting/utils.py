"""
Вспомогательные функции для генерации графиков для отчетов.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> io.BytesIO:
    """
    Строит ROC-кривую и возвращает график в виде изображения.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred_proba: Предсказанные вероятности
        
    Returns:
        io.BytesIO: Поток с изображением
    """
    # Вычисляем ROC-кривую
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Создаем график
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC кривая')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Сохраняем график в поток
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
    img_stream.seek(0)
    plt.close()
    
    return img_stream

def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> io.BytesIO:
    """
    Строит кривую точности-полноты и возвращает график в виде изображения.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred_proba: Предсказанные вероятности
        
    Returns:
        io.BytesIO: Поток с изображением
    """
    # Вычисляем precision-recall кривую
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Создаем график
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall кривая')
    plt.grid(True)
    
    # Сохраняем график в поток
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
    img_stream.seek(0)
    plt.close()
    
    return img_stream

def plot_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray) -> io.BytesIO:
    """
    Строит матрицу ошибок в виде тепловой карты и возвращает график в виде изображения.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred: Предсказанные значения
        
    Returns:
        io.BytesIO: Поток с изображением
    """
    # Вычисляем матрицу ошибок
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Создаем тепловую карту
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.title('Матрица ошибок')
    
    # Сохраняем график в поток
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
    img_stream.seek(0)
    plt.close()
    
    return img_stream

def plot_feature_importance(feature_importance: Dict[str, float], 
                           top_n: int = 10) -> io.BytesIO:
    """
    Строит график важности признаков и возвращает его в виде изображения.
    
    Args:
        feature_importance: Словарь с важностью признаков
        top_n: Количество наиболее важных признаков для отображения
        
    Returns:
        io.BytesIO: Поток с изображением
    """
    # Сортируем признаки по важности
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_features = sorted_features[:top_n]
    
    # Создаем график
    plt.figure(figsize=(10, 6))
    feature_names = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    # Строим горизонтальный столбчатый график
    plt.barh(range(len(feature_names)), importances, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Важность')
    plt.title('Важность признаков')
    plt.tight_layout()
    
    # Сохраняем график в поток
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
    img_stream.seek(0)
    plt.close()
    
    return img_stream

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> io.BytesIO:
    """
    Строит график остатков для регрессионной модели.
    
    Args:
        y_true: Истинные значения целевой переменной
        y_pred: Предсказанные значения
        
    Returns:
        io.BytesIO: Поток с изображением
    """
    residuals = y_true - y_pred
    
    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title('График остатков')
    plt.grid(True)
    
    # Сохраняем график в поток
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
    img_stream.seek(0)
    plt.close()
    
    return img_stream 