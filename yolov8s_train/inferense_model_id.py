# predict.py
from ultralytics import YOLO
from clearml import Task, Model
import cv2
import os

task = Task.init(
    project_name='Car Detection Project',
    task_name='Inference with best YOLOv8s model'
)

TEST_IMAGE_PATH = 'test_img/car_test.jpeg'  

print("Получаем лучшую модель из реестра ClearML...")

try:
    model_object = Model(model_id='9b6d9a3dff7442e3880d15404ebcc70e')
    
    local_model_path = model_object.get_local_copy()
    print(f"Модель успешно скачана: {local_model_path}")

except Exception as e:
    print(f"Ошибка при получении модели из ClearML: {e}")
    print("Убедитесь, что имя проекта и имя задачи указаны верно.")
    exit()  


print("Загружаем дообученную модель в YOLO...")
model = YOLO(local_model_path)

print(f"Делаем предсказание для изображения: {TEST_IMAGE_PATH}")
results = model(TEST_IMAGE_PATH)

print("Обрабатываем и загружаем результат...")

annotated_image = results[0].plot() 

image_filename = os.path.basename(TEST_IMAGE_PATH)

task.get_logger().report_image(
    title="Inference Result",
    series=f"Prediction on {image_filename}",
    iteration=0,
    image=cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  
)

print("\nИнференс завершен!")
print(f"Результат можно посмотреть на странице эксперимента '{task.name}' в разделе PLOTS.")

# Чтобы показать картинку на экране локально (опционально)
cv2.imshow("YOLOv8 Inference Result", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()