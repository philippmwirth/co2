# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly
from _ntx_ent_loss import NTXentLoss
from utils import BenchmarkModule
from co2 import CO2Regularizer

num_workers = 8

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
knn_k = 200
knn_t = 0.1
classes = 10
batch_size = 512

# co2 settings
alpha = 1.0
t_consistency = 1.0

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

# Use SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True
    )
)
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        transform=test_transforms,
        download=True
    )
)
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        transform=test_transforms,
        download=True
    )
)

def get_data_loaders(batch_size: int):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test
    

class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, gpus, classes, knn_k, knn_t, alpha):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simclr model based on ResNet
        self.resnet_simclr = \
            lightly.models.SimCLR(self.backbone, num_ftrs=512)
        self.criterion = NTXentLoss(temperature=0.5)
        # create co2 regularizer without memory bank (hence size=0)
        self.co2 = CO2Regularizer(alpha=alpha, memory_bank_size=0, t_consistency=t_consistency)
            
    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(y0, y1)
        reg = self.co2(y0, y1)
        self.log('train_loss_ssl', loss)
        self.log('co2_reg', reg)
        loss = loss + reg
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simclr.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


seed = 1234

# train simclr with co2
pl.seed_everything(seed)
dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
benchmark_model = SimCLRModel(dataloader_train_kNN, 1, 10, knn_k, knn_t, alpha)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
trainer.fit(
    benchmark_model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)
print(f'SimCLR w/ CO2: {benchmark_model.max_accuracy}')

# train simclr without co2
pl.seed_everything(seed)
dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
benchmark_model = SimCLRModel(dataloader_train_kNN, 1, 10, knn_k, knn_t, 0.0)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
trainer.fit(
    benchmark_model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)
print(f'SimCLR w/o CO2: {benchmark_model.max_accuracy}')
