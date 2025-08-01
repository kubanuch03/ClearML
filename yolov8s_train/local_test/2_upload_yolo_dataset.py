from clearml import Dataset, Task




DATASET_PROJECT = 'Car Detection Project'
DATASET_NAME = 'yolo_car_dataset'
PATH_TO_DATASET_FOLDER = 'car_dataset'  


Task.init(
    project_name=DATASET_PROJECT, 
    task_name=f'Upload Dataset: {DATASET_NAME}'
)

dataset = Dataset.create(
    dataset_project=DATASET_PROJECT, 
    dataset_name=DATASET_NAME
)

dataset.add_files(path=PATH_TO_DATASET_FOLDER)
dataset.upload()
dataset.finalize()
print("Датасет для YOLO успешно загружен в ClearML!")