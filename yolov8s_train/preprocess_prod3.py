import base64
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import sys
import logging
import functools # <--- ОБЯЗАТЕЛЬНО ДОБАВЬТЕ ЭТОТ ИМПОРТ

# Настраиваем логирование
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Preprocess:
    def __init__(self):
        logging.info("--- Preprocess __init__ ---")
        self.model = None
        self.labels = None

    def load(self, model_path: str):
        logging.info(f"--- НАЧАЛО ЗАГРУЗКИ МОДЕЛИ, путь: {model_path} ---")
        
        # --- НАЧАЛО БЛОКА MONKEY-PATCHING ---
        # Это временное решение для обхода конфликта ClearML и PyTorch 2.6+
        # Мы принудительно выключаем безопасный режим `weights_only=True`
        # только на время вызова YOLO(model_path)
        
        original_torch_load = torch.load
        try:
            logging.warning("ВРЕМЕННОЕ РЕШЕНИЕ: Подменяем torch.load, чтобы установить weights_only=False.")
            # Создаем новую версию torch.load с нужным нам параметром по умолчанию
            torch.load = functools.partial(torch.load, weights_only=False)
            
            model_file = Path(model_path)
            if not model_file.is_file():
                logging.error(f"Файл модели НЕ НАЙДЕН по пути: {model_path}")
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            
            logging.info(f"Файл модели найден. Версия PyTorch: {torch.__version__}. Пытаюсь загрузить через подмененный YOLO...")
            
            # Теперь этот вызов внутри будет использовать нашу версию torch.load
            self.model = YOLO(model_path)
            self.model.to('cpu')

            if hasattr(self.model, 'names'):
                self.labels = self.model.names
                logging.info(f"Метки классов успешно загружены: {self.labels}")
            else:
                logging.warning("Не удалось получить self.model.names.")
                self.labels = {0: 'car'}

            logging.info("--- МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА ВНУТРИ Preprocess.load ---")
            return True
        
        except Exception as e:
            logging.error(f"КРИТИЧЕСКАЯ ОШИБКА в методе load: {e}", exc_info=True)
            raise e
        finally:
            # ВОЗВРАЩАЕМ ВСЕ КАК БЫЛО! Это критически важно.
            torch.load = original_torch_load
            logging.info("Восстановили оригинальный torch.load.")
        # --- КОНЕЦ БЛОКА MONKEY-PATCHING ---

    
    def preprocess(self, request: dict, model_endpoint: str, version: str) -> np.ndarray:
        image_base64 = request.get("image_base64")
        if not image_base64: raise ValueError("Ожидается 'image_base64'")
        image_bytes = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    def process(self, data: np.ndarray, state: dict, collect_custom_statistics_fn=None) -> dict:
        if self.model is None: raise RuntimeError("Модель не была загружена.")
        results = self.model.predict(data, conf=0.25)
        predictions = []
        for res in results:
            boxes = res.boxes.cpu().numpy()
            for box in boxes:
                predictions.append({
                    'label': self.labels.get(int(box.cls), 'unknown_class'),
                    'confidence': float(box.conf),
                    'bbox_xyxy': box.xyxy.tolist()[0]
                })
        return {"predictions": predictions}
    
    def postprocess(self, data: dict, state: dict, collect_custom_statistics_fn=None) -> dict:
        return data