from ultralytics import YOLO
import cv2

# Загружаем НАШУ, дообученную модель
model = YOLO('../runs/detect/yolov8s_cars_finetuned3/weights/best.pt') 

# Путь к картинке для теста
image_path = './test_img/car_test.jpeg'

# Делаем предсказание
results = model(image_path)

# Рисуем результаты и показываем их
annotated_frame = results[0].plot()
# print(annotated_frame)
cv2.imshow("YOLOv8 Inference", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()