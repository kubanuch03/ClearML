# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet18_Weights

from data_model import MyNumberplateDataModule


class MyNumberplateModel(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3, unfreeze_backbone: bool = False):
        """
        Конструктор модели.
        
        :param num_classes: Количество классов (типов номеров) для новой задачи.
        :param learning_rate: Темп обучения (learning rate).
        :param unfreeze_backbone: Если True, будет дообучаться вся модель.
                                  Если False (по умолчанию), будет обучаться только "голова".
        """
        super().__init__()
        self.save_hyperparameters()

        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if not self.hparams.unfreeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, self.hparams.num_classes)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

 
if __name__ == '__main__':
    DATA_PATH = "./data"   
    CHECKPOINT_PATH = "/checkpoint_models/numberplate_test.ckpt"  
    
    BATCH_SIZE = 64
    IMAGE_SIZE = (100, 400)  
    NUM_NEW_CLASSES = 150  
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 10
    
    datamodule = MyNumberplateDataModule(
        data_dir=DATA_PATH, 
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    try:
        print(f"Загрузка весов из чекпоинта: {CHECKPOINT_PATH}")
        model = MyNumberplateModel(num_classes=NUM_NEW_CLASSES, learning_rate=LEARNING_RATE)
        
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Веса успешно загружены. Последний слой проигнорирован из-за несовпадения размеров.")

    except FileNotFoundError:
        print("Чекпоинт не найден. Создание новой модели с нуля.")
        model = MyNumberplateModel(num_classes=NUM_NEW_CLASSES, learning_rate=LEARNING_RATE)


    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",  
        devices=2,          
        logger=True,        
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-model-{epoch:02d}-{val_loss:.2f}"
            )
        ]
    )

    # 4. Запускаем дообучение!
    print("Начинаем дообучение...")
    trainer.fit(model, datamodule=datamodule)
    print("Дообучение завершено.")
    
    final_checkpoint_path = "numberplate_finetuned_final.ckpt"
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Финальная дообученная модель сохранена в: {final_checkpoint_path}")