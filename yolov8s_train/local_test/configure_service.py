# configure_service.py (ВЕРСИЯ 2 - ИСПРАВЛЕННАЯ)
from clearml import Task
import os

# --- НАСТРОЙКИ ---
SERVING_SERVICE_ID = "195cfa816f3743ed9b71c3286cb18fe0"
MODEL_PROJECT = "Car Detection Project"
MODEL_NAME = "yolov8s_cars_finetuned3"
ENDPOINT_NAME = "car-detector"
PREPROCESS_FILE = "preprocess.py"
PREPROCESS_CLASS = "Preprocess"

# --- ОСНОВНОЙ КОД ---
print("--- [КОНФИГУРАТОР, ВЕРСИЯ 2] Запускаем скрипт ---")

# Проверка наличия файла
if not os.path.exists(PREPROCESS_FILE):
    print(f"!!! ОШИБКА: Файл '{PREPROCESS_FILE}' не найден. Выход.")
    exit()

try:
    # 1. Получаем объект задачи сервиса
    print(f"Подключаемся к сервису с ID: {SERVING_SERVICE_ID}")
    serving_task = Task.get_task(task_id=SERVING_SERVICE_ID)

    # 2. Читаем содержимое скрипта
    print(f"Читаем содержимое файла: {PREPROCESS_FILE}")
    with open(PREPROCESS_FILE, 'r') as f:
        script_content = f.read()
    print("Содержимое скрипта прочитано.")

    # 3. Формируем конфигурацию для эндпоинта
    endpoint_config = {
        "project": MODEL_PROJECT,
        "name": MODEL_NAME,
        "engine": "custom",
        "serving_script": {
            "source": script_content,
            "entry_point": f"{PREPROCESS_FILE}:{PREPROCESS_CLASS}"
        },
        # "extra_python_packages": ["ultralytics", "opencv-python-headless"]
    }

    # 4. Обновляем конфигурацию через параметры задачи (ПРАВИЛЬНЫЙ СПОСОБ)
    print(f"Обновляем конфигурацию для эндпоинта: {ENDPOINT_NAME}")
    
    # Получаем текущие параметры эндпоинтов
    endpoints = serving_task.get_parameters_as_dict().get("Models", {})
    
    # Добавляем/обновляем наш эндпоинт
    endpoints[ENDPOINT_NAME] = endpoint_config
    
    # Сохраняем обновленный словарь эндпоинтов обратно в задачу
    serving_task.set_parameters(Models=endpoints)
    
    print("\n--- УСПЕХ! (ВЕРСИЯ 2) ---")
    print("Конфигурация сервиса развертывания успешно обновлена.")
    print("Теперь можно запускать 'docker run'.")

except Exception as e:
    print(f"\n--- ПРОИЗОШЛА НЕПРЕДВИДЕННАЯ ОШИБКА (ВЕРСИЯ 2) ---")
    print(e)