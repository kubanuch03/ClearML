# 2_train_model.py
from clearml import Task, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # для сохранения модели






dataset = Dataset.get(dataset_project='examples-mnist', dataset_name='mnist_dataset')
local_dataset_path = dataset.get_local_copy()

print(f'Данные находятся в: {local_dataset_path}')
X = np.load(f'{local_dataset_path}/mnist_features.npy', allow_pickle=True)
y = np.load(f'{local_dataset_path}/mnist_labels.npy', allow_pickle=True)
