from typing import Dict
import pandas as pd
from ..models.base import BaseModel

class ModelValidator:
    """Класс для валидации модели"""
    def __init__(self, validation_params: Dict):
        self.validation_params = validation_params
        
    def validate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Валидация модели и расчет метрик"""
        pass 