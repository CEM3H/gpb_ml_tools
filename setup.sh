#!/bin/bash

# Создаем директорию для виртуальных окружений, если она не существует
mkdir -p ~/.virtualenvs

# Создаем виртуальное окружение
python3 -m venv ~/.virtualenvs/gpb_env

# Активируем окружение
source ~/.virtualenvs/gpb_env/bin/activate

# Обновляем pip
pip install --upgrade pip

# Устанавливаем зависимости
pip install -r requirements.txt

echo "Виртуальное окружение gpb_env успешно создано и настроено"
echo "Для активации окружения выполните: source ~/.virtualenvs/gpb_env/bin/activate" 