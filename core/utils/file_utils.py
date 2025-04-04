import os
import pathlib
from typing import Union, List, Optional

class SecurityError(Exception):
    """
    Исключение, возникающее при проблемах безопасности при работе с файлами.
    """
    pass

def safe_path_join(base_dir: str, *paths: str) -> str:
    """
    Безопасно объединяет базовую директорию с дополнительными путями, 
    предотвращая path traversal атаки.
    
    Parameters
    ----------
    base_dir : str
        Базовая директория
    *paths : str
        Дополнительные компоненты пути
    
    Returns
    -------
    str
        Безопасный абсолютный путь
    
    Raises
    ------
    SecurityError
        Если обнаружена попытка path traversal
    """
    base_path = os.path.abspath(base_dir)
    joined_path = os.path.abspath(os.path.join(base_path, *paths))
    
    # Проверяем, что путь находится внутри базовой директории
    if not joined_path.startswith(base_path + os.sep) and joined_path != base_path:
        raise SecurityError(f"Попытка path traversal: {joined_path} за пределами {base_path}")
    
    return joined_path

def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Проверяет, что расширение файла находится в списке разрешенных.
    
    Parameters
    ----------
    filename : str
        Имя файла для проверки
    allowed_extensions : List[str]
        Список разрешенных расширений (с точкой, например ['.csv', '.json'])
    
    Returns
    -------
    bool
        True если расширение разрешено, иначе False
    """
    ext = pathlib.Path(filename).suffix.lower()
    return ext in [ext.lower() for ext in allowed_extensions]

def get_safe_filename(filename: str) -> str:
    """
    Создает безопасное имя файла, удаляя потенциально опасные символы.
    
    Parameters
    ----------
    filename : str
        Исходное имя файла
    
    Returns
    -------
    str
        Безопасное имя файла
    """
    # Удаляем все символы, кроме букв, цифр, точек, дефисов и подчеркиваний
    import re
    safe_name = re.sub(r'[^\w\.\-]', '_', filename)
    
    # Предотвращаем использование относительных путей
    safe_name = os.path.basename(safe_name)
    
    return safe_name 