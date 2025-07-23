# 1_prepare_data.py
from sklearn.datasets import fetch_openml
from clearml import Dataset, Task
import numpy as np



task = Task.init(
    project_name='examples-mnist',
    task_name='data preparation script',
    task_type='data_processing'  # Указываем тип задачи
)

# --- Шаг 1: Создаем проект для датасета в ClearML ---
# Это как `Task.init`, но для данных.
dataset_project = 'examples-mnist'
dataset_name = 'mnist_dataset'

# --- Шаг 2: Скачиваем данные ---
print('Скачиваем данные MNIST...')
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist['data'], mnist['target']

# --- Шаг 3: Сохраняем данные локально ---
# ClearML будет загружать файлы из этой папки
np.save('mnist_features.npy', X)
np.save('mnist_labels.npy', y)

# --- Шаг 4: Создаем и загружаем датасет в ClearML ---
print('Создаем датасет в ClearML...')
dataset = Dataset.create(
    dataset_project=dataset_project,
    dataset_name=dataset_name
)

# Добавляем файлы в датасет
print('Добавляем файлы...')
dataset.add_files(path='.') # Добавляем все файлы из текущей папки

# Загружаем данные на сервер и делаем их доступными для других задач
print('Загружаем данные на сервер...')
dataset.upload()
dataset.finalize()

print('Датасет готов!')