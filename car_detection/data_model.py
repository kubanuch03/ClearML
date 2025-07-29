import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

class MyNumberplateDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, image_size: tuple = (100, 400)):
        """
        Конструктор DataModule.
        
        :param data_dir: Путь к корневой папке с данными (где лежат папки train и val).
        :param batch_size: Размер батча (сколько картинок обрабатывать за раз).
        :param image_size: Размер, до которого нужно изменить все изображения (высота, ширина).
        """
        super().__init__()
        # Сохраняем параметры для использования в других методах
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        # Определяем трансформации (аугментацию и нормализацию)
        # Это очень важный шаг для обучения моделей изображений
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size), # Изменяем размер всех картинок
            transforms.ToTensor(),             # Превращаем картинку в тензор PyTorch
            transforms.Normalize(              # Нормализуем значения пикселей
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup(self, stage: str):
        """
        Этот метод вызывается для подготовки данных.
        'stage' может быть 'fit', 'validate', 'test', 'predict'.
        """
        # Мы создаем датасеты здесь, чтобы они не сохранялись в pickle
        # и были доступны в каждом процессе при распределенном обучении.
        
        # PyTorch `ImageFolder` автоматически найдет классы (по именам подпапок)
        # и применит наши трансформации к каждому изображению.
        if stage == "fit":
            train_path = os.path.join(self.data_dir, "train")
            val_path = os.path.join(self.data_dir, "val")
            
            self.train_dataset = ImageFolder(root=train_path, transform=self.transform)
            self.val_dataset = ImageFolder(root=val_path, transform=self.transform)
        
        # Здесь можно добавить логику для stage == "test", если у вас есть папка 'test'
        # if stage == "test":
        #     test_path = os.path.join(self.data_dir, "test")
        #     self.test_dataset = ImageFolder(root=test_path, transform=self.transform)

    def train_dataloader(self):
        """Создает и возвращает DataLoader для обучающей выборки."""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, # Очень важно перемешивать данные для обучения
            num_workers=4   # Количество параллельных потоков для загрузки данных
        )

    def val_dataloader(self):
        """Создает и возвращает DataLoader для валидационной выборки."""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, # Перемешивать валидационные данные не нужно
            num_workers=4
        )

    # def test_dataloader(self):
    #     """Создает и возвращает DataLoader для тестовой выборки."""
    #     return DataLoader(
    #         self.test_dataset, 
    #         batch_size=self.batch_size, 
    #         shuffle=False,
    #         num_workers=4
    #     )