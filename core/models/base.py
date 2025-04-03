from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import numpy as np
import optuna

class BaseModel(ABC):
    """Абстрактный базовый класс для моделей"""
    def __init__(self, hyperparameters: Optional[Dict] = None):
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.best_params = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Обучение модели"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Получение предсказаний"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Получение важности признаков"""
        pass
        
    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict,
        n_trials: int = 100,
        metric: str = 'roc_auc'
    ) -> Dict:
        """Оптимизация гиперпараметров с помощью Optuna"""
        def objective(trial):
            # Создаем словарь параметров для текущего испытания
            params = {}
            for param_name, param_space in param_space.items():
                if isinstance(param_space, tuple):
                    if isinstance(param_space[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_space[0],
                            param_space[1]
                        )
                    elif isinstance(param_space[0], float):
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_space[0],
                            param_space[1]
                        )
                elif isinstance(param_space, list):
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_space
                    )
            
            # Обновляем параметры модели
            self.hyperparameters.update(params)
            
            # Обучаем модель
            self.fit(X, y)
            
            # Оцениваем качество
            if metric == 'roc_auc':
                return self._evaluate_roc_auc(X, y)
            elif metric == 'accuracy':
                return self._evaluate_accuracy(X, y)
            else:
                raise ValueError(f"Неподдерживаемая метрика: {metric}")
        
        # Создаем и запускаем оптимизацию
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Сохраняем лучшие параметры
        self.best_params = study.best_params
        self.hyperparameters.update(self.best_params)
        
        return self.best_params
        
    def _evaluate_roc_auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Оценка ROC-AUC"""
        from sklearn.metrics import roc_auc_score
        y_pred = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_pred)
        
    def _evaluate_accuracy(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Оценка точности"""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred) 