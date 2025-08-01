# 2_train_model.py
from clearml import Task, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # для сохранения модели

# --- Шаг 1: Инициализируем задачу в ClearML ---
task = Task.init(
    project_name='examples-mnist',
    task_name='train_logistic_regression',
    output_uri=True # Очень важно! Говорит ClearML сохранять артефакты (модели)
)

# --- Шаг 2: Получаем данные из ClearML Data ---
print('Получаем последнюю версию датасета...')
# Это скачает данные, если их нет локально
dataset = Dataset.get(dataset_project='examples-mnist', dataset_name='mnist_dataset')
local_dataset_path = dataset.get_local_copy()

print(f'Данные находятся в: {local_dataset_path}')
X = np.load(f'{local_dataset_path}/mnist_features.npy', allow_pickle=True)
y = np.load(f'{local_dataset_path}/mnist_labels.npy', allow_pickle=True)

# --- Шаг 3: Определяем параметры ---
# ClearML автоматически их подхватит
params = {
    'test_size': 0.2,
    'random_state': 42,
    'C': 0.1
}
# Явно подключаем параметры к задаче для лучшей организации
task.connect(params)

# --- Шаг 4: Обучение модели ---
print('Разделяем данные и обучаем модель...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['test_size'], random_state=params['random_state']
)

model = LogisticRegression(C=params['C'], max_iter=1000) # Простая модель
model.fit(X_train, y_train)

# --- Шаг 5: Оценка и логирование метрик ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy = {acc}')

# Логируем метрику в ClearML, чтобы видеть красивые графики
task.get_logger().report_scalar(
    title='performance', series='accuracy', value=acc, iteration=1
)

# --- Шаг 6: Сохраняем модель ---
# ClearML автоматически найдет этот файл и загрузит его
joblib.dump(model, 'model.pkl')

print('Задача выполнена! Модель сохранена и загружена в ClearML.')