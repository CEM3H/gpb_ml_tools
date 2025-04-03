"""
GPB ML Library - библиотека для разработки и деплоя ML-моделей.
"""

from .version import __version__

# Импортируем основные компоненты для удобства использования
from .pipeline import ModelPipeline
from .container import ModelContainer, ModelMetadata

__all__ = [
    'ModelPipeline',
    'ModelContainer', 
    'ModelMetadata',
    '__version__'
] 