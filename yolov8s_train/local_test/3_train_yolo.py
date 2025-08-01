# 3_train_yolo.py
from ultralytics import YOLO
from clearml import Task, Dataset
import os
import yaml  

task = Task.init(
    project_name='Car Detection Project',
    task_name='Finetune YOLOv8s on my cars',
    output_uri=True
)

print("Получаем датасет из ClearML...")
dataset = Dataset.get(
    dataset_project='Car Detection Project', 
    dataset_name='yolo_car_dataset'
)
dataset_path = dataset.get_local_copy()
print(f"Датасет скачан в: {dataset_path}")

data_yaml_path = os.path.join(dataset_path, 'data.yaml')

with open(data_yaml_path, 'r') as f:
    data_yaml_content = yaml.safe_load(f)

data_yaml_content['path'] = dataset_path
if 'train' in data_yaml_content:
    data_yaml_content['train'] = os.path.join(dataset_path, data_yaml_content['train'])
if 'val' in data_yaml_content:
    data_yaml_content['val'] = os.path.join(dataset_path, data_yaml_content['val'])
if 'test' in data_yaml_content and data_yaml_content['test']:
    data_yaml_content['test'] = os.path.join(dataset_path, data_yaml_content['test'])

new_yaml_path = os.path.join(dataset_path, 'data_fixed.yaml')
with open(new_yaml_path, 'w') as f:
    yaml.dump(data_yaml_content, f)

print(f"Исправленный файл конфигурации создан: {new_yaml_path}")


model = YOLO('yolov8s.pt')

print("Начинаем дообучение...")
results = model.train(
    data=new_yaml_path,   
    epochs=50,
    imgsz=640,
    batch=8,
    name='yolov8s_cars_finetuned'
)

print("Дообучение завершено! Все результаты в ClearML.")