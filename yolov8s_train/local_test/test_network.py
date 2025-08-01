# test_network.py
from clearml import Task

# --- НАСТРОЙКИ ДЛЯ ТЕСТА ---
SERVING_SERVICE_ID = "195cfa816f3743ed9b71c3286cb18fe0"
ENDPOINT_NAME = "test-endpoint" # НОВОЕ ИМЯ ЭНДПОИНТА
# Используем ту же модель, он ее все равно не будет загружать, если пакеты не установятся
MODEL_PROJECT = "Car Detection Project" 
MODEL_NAME = "yolov8s_cars_finetuned3"
# Минимальный скрипт-пустышка
DUMMY_SCRIPT_CONTENT = "class Preprocess:\n    def load(self, model_path):\n        print('Hello from dummy script')\n    def predict(self, data):\n        return {'status': 'ok'}"
# САМОЕ ВАЖНОЕ: простой пакет для проверки сети
PACKAGES_TO_INSTALL = ["requests"]

# --- КОД ---
print("--- [СЕТЕВОЙ ТЕСТ] Настраиваем минимальный эндпоинт ---")
try:
    serving_task = Task.get_task(task_id=SERVING_SERVICE_ID)
    endpoint_config = {
        "project": MODEL_PROJECT,
        "name": MODEL_NAME,
        "engine": "custom",
        "serving_script": {
            "source": DUMMY_SCRIPT_CONTENT,
            "entry_point": "dummy.py:Preprocess"
        },
        "extra_python_packages": PACKAGES_TO_INSTALL
    }
    endpoints = serving_task.get_parameters_as_dict().get("Models", {})
    endpoints[ENDPOINT_NAME] = endpoint_config
    serving_task.set_parameters(Models=endpoints)
    print("\n--- [СЕТЕВОЙ ТЕСТ] УСПЕХ! Конфигурация обновлена.")
    print("Теперь остановите старый Docker и запустите новый.")
except Exception as e:
    print(f"\n--- [СЕТЕВОЙ ТЕСТ] ОШИБКА ---")
    print(e)