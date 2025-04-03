from setuptools import setup, find_packages
import os

# Читаем версию из файла version.py
with open(os.path.join('core', 'version.py'), 'r') as f:
    exec(f.read())

# Читаем README.md для описания пакета
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Читаем requirements.txt для зависимостей
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="gpb_lib",
    version=__version__,  # Версия из version.py
    author="GPB Team",
    author_email="example@example.com",
    description="Библиотека для разработки и деплоя ML-моделей",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/gpb_lib",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'core': ['*.yaml', '*.json'],
    }
) 