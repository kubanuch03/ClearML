from ultralytics import YOLO
from clearml import Task

task = Task.init(
    project_name='YOLOv8 Project',
    task_name='Finetune YOLOv8s on my custom data',
    output_uri=True  # Важно, чтобы ClearML сохранял итоговую модель
)

model = YOLO('yolov8s.pt')

results = model.train(
    data='data.yaml',   
    epochs=50,
    imgsz=640,
    batch=8,
    name='yolov8s_finetuned' # Название папки с результатами
)


print("Дообучение завершено! Все результаты в ClearML.")