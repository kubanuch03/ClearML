# serving_script.py
import json
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

class Preprocess:
    def __init__(self):
        self.model = None
        self.labels = None

    def load(self, model_path: str):
        """
        Эта функция вызывается для загрузки модели.
        model_path будет содержать путь к папке с моделью.
        """
        model_file = list(Path(model_path).glob('*.pt'))[0]
        self.model = YOLO(model_file)
        self.labels = self.model.names
        print(f"Модель {model_file} успешно загружена.")

    def preprocess(self, request: dict) -> np.ndarray:
        """
        Обработка входящего запроса. Мы ожидаем URL картинки.
        """
        image_url = request.get("image_url")
        if not image_url:
            raise ValueError("Необходимо передать 'image_url' в запросе")

        cap = cv2.VideoCapture(image_url)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Не удалось скачать или прочитать изображение по URL: {image_url}")
        
        return frame

    def predict(self, data: np.ndarray) -> dict:
        """
        Основная функция предсказания.
        """
        results = self.model(data)
        result = results[0]  
        
        predictions = []
        for box in result.boxes:
            label = self.labels[int(box.cls)]
            conf = float(box.conf)
            xyxy = box.xyxy.cpu().numpy()[0].tolist()
            
            predictions.append({
                'label': label,
                'confidence': conf,
                'bbox': xyxy
            })
        
        return {"predictions": predictions}

    def postprocess(self, data: dict) -> dict:
        """
        Финальная обработка ответа перед отправкой клиенту.
        """
        return data