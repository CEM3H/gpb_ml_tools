#!/bin/bash

# Проверяем, активировано ли виртуальное окружение
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Виртуальное окружение не активировано!"
    echo "Активируйте виртуальное окружение перед установкой."
    echo "Пример: source ~/.virtualenvs/gpb_env/bin/activate"
    exit 1
fi

# Устанавливаем необходимые пакеты для сборки
pip install -U setuptools wheel build pip

# Устанавливаем пакет в режиме разработки
pip install -e .

echo "Пакет gpb_lib успешно установлен в режиме разработки."
echo "Теперь можно импортировать библиотеку как обычный пакет Python:"
echo ">>> from core.pipeline import ModelPipeline"
echo ">>> from core.data.schema import DataSchema" 