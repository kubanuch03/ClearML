# 3_pipeline.py
from clearml import PipelineController

pipe = PipelineController(
    name='MNIST End-to-End',
    project='examples-mnist',
    version='0.0.1',
)

# Указываем, что пайплайн будет выполняться в очереди 'default'
pipe.set_default_execution_queue('default')

# Добавляем шаг подготовки данных
# Он запустит наш скрипт 1_prepare_data.py
pipe.add_step(
    name='prepare_data',
    base_task_project='examples-mnist',
    base_task_name='data preparation script' # Можно создать пустую задачу-шаблон
)

# Добавляем шаг обучения
# Он запустит скрипт 2_train_model.py
# parents=['prepare_data'] означает, что этот шаг начнется только ПОСЛЕ успешного завершения шага 'prepare_data'
pipe.add_step(
    name='train_model',
    parents=['prepare_data'],
    base_task_project='examples-mnist',
    base_task_name='train_logistic_regression' # Используем нашу задачу как шаблон
)

print('Отправляем определение пайплайна в ClearML и ставим его в очередь default...')
pipe.start(queue='default')

print('Пайплайн запущен! Проверьте веб-интерфейс.')