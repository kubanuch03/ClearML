# 3a_pipeline_comparison.py
from clearml import PipelineController

# Создаем новый пайплайн с новым именем
pipe = PipelineController(
    name='MNIST Comparison Pipeline', # Новое имя
    project='examples-mnist',
    version='0.0.1'
)

# Очередь выполнения остается та же
pipe.set_default_execution_queue('default')

# Шаг 1: Подготовка данных (остается без изменений)
pipe.add_step(
    name='prepare_data',
    base_task_project='examples-mnist',
    base_task_name='data preparation script'
)

# Шаг 2a: Обучение базовой модели
# Он зависит от шага 'prepare_data'
pipe.add_step(
    name='train_base_model', # Новое имя для шага
    parents=['prepare_data'],
    base_task_project='examples-mnist',
    base_task_name='train_logistic_regression' # Используем СТАРЫЙ шаблон
)

# Шаг 2b: Обучение более сильной модели
# Он ТОЖЕ зависит от шага 'prepare_data', поэтому будет запущен параллельно с шагом 2a
pipe.add_step(
    name='train_stronger_model', # Новое имя для шага
    parents=['prepare_data'],
    base_task_project='examples-mnist',
    base_task_name='train_logistic_regression_stronger' # Используем НОВЫЙ шаблон
)

# Запускаем определение пайплайна
print('Отправляем определение пайплайна для сравнения в ClearML...')
pipe.start(queue='default')

print('Пайплайн для сравнения создан! Проверьте веб-интерфейс.')