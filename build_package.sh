#!/bin/bash

# Очистка предыдущих сборок
rm -rf dist/
rm -rf build/
rm -rf *.egg-info

# Установка необходимых инструментов
pip install -U setuptools wheel build pip

# Сборка пакета
python -m build

echo "Сборка завершена. Файлы находятся в каталоге dist/"
echo "Wheel-файл можно установить командой:"
echo "pip install dist/*.whl" 