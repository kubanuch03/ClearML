# serving_script_final.py
import json
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
import base64

# САМОЕ ВАЖНОЕ СООБЩЕНИЕ. ЕСЛИ МЫ ЕГО НЕ УВИДИМ, ЭТОТ КОД НЕ ВЫПОЛНЯЕТСЯ.
print("--- [FINAL VERSION] PYTHON INTERPRETER HAS LOADED THIS SCRIPT! ---")

# serving_script_bulletproof.py

# ... (все импорты остаются) ...
print("--- [BULLETPROOF VERSION] PYTHON INTERPRETER HAS LOADED THIS SCRIPT! ---")

class Preprocess:
    def __init__(self):
        print("--- [BULLETPROOF] Preprocess __init__ ---")
        self.model = None
        self.labels = None

    def load(self, model_path: str):
        print("--- [BULLETPROOF] Preprocess load ---")
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"[BULLETPROOF] Файл модели не найден: {model_path}")

        # <<< ИЗМЕНЕНИЕ №1: ЯВНО ЗАГРУЖАЕМ МОДЕЛЬ НА CPU >>>
        # Это гарантирует, что и локально, и на сервере будет одинаковая среда.
        print("--- [BULLETPROOF] Загрузка модели на CPU ---")
        self.model = YOLO(model_file)
        self.model.to('cpu') # Принудительно переключаем модель на CPU

        self.labels = self.model.names
        print("--- [BULLETPROOF] Модель успешно загружена на CPU ---")

    def preprocess(self, request: dict, model_endpoint: str, version: str) -> np.ndarray:
        # Эту функцию можно не менять, она уже идеальна
        print("--- [BULLETPROOF] Preprocess preprocess ---")
        image_base64 = request.get("image_base64")
        if not image_base64: raise ValueError("[BULLETPROOF] Ожидается 'image_base64'")
        try:
            image_bytes = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            bgr_frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if bgr_frame is None: raise ValueError("[BULLETPROOF] Не удалось декодировать base64")
            target_size = 640
            h, w, _ = bgr_frame.shape
            if h > w:
                new_h, new_w = target_size, int(w * (target_size / h))
            else:
                new_w, new_h = target_size, int(h * (target_size / w))
            resized_bgr_frame = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(resized_bgr_frame, cv2.COLOR_BGR2RGB)
            return rgb_frame
        except Exception as e:
            raise ValueError(f"[BULLETPROOF] Ошибка обработки: {e}")

    def predict(self, data: np.ndarray) -> dict:
        # ЭТОТ ТЕСТ НУЖЕН, ЧТОБЫ ПРОВЕРИТЬ, ВЫЗЫВАЕТСЯ ЛИ ФУНКЦИЯ ВООБЩЕ
        print("--- [DUMMY PREDICT] Функция predict была успешно вызвана! ---")
        
        # Мы НЕ вызываем модель. Вместо этого возвращаем фальшивые данные.
        # results = self.model(data, conf=0.01) # Проблемная строка закомментирована

        dummy_predictions = [
            {
                'label': 'dummy_car',
                'confidence': 0.99,
                'bbox': [10, 10, 100, 100]
            }
        ]
        
        print(f"[DUMMY PREDICT] Возвращаем {len(dummy_predictions)} фальшивых объектов")
        return {
                "object_count": 178,
                "predictions": []
            }
    
    # postprocess можно не менять
    def postprocess(self, data: dict, model_endpoint: str, version: str) -> dict:
        print("--- [BULLETPROOF] Preprocess postprocess ---")
        if not data: data = {"predictions": []}
        predictions_list = data.get("predictions", [])
        return {"object_count": len(predictions_list), "predictions": predictions_list}