# test.py
print("--- ТЕСТОВЫЙ СКРИПТ УСПЕШНО ЗАГРУЖЕН ---")

class Preprocess:
    def __init__(self):
        print("--- ТЕСТОВЫЙ КЛАСС ИНИЦИАЛИЗИРОВАН ---")

    def load(self, model_path: str):
        print(f"--- МОДЕЛЬ ДОЛЖНА БЫТЬ ЗАГРУЖЕНА ИЗ {model_path} ---")

    def predict(self, data):
        return {"hello": "world"}