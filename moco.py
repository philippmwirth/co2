# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly
from lightly import loss
from utils import BenchmarkModule
from co2 import CO2Regularizer

num_workers = 8
memory_bank_size = 4096

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 3
knn_k = 200
knn_t = 0.1
classes = 10

# benchmark
n_runs = 1 # optional, increase to create multiple runs and report mean + std
batch_size = 512

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
    

class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, gpus, classes, knn_k, knn_t, alpha):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a moco model based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(self.backbone, num_ftrs=512, m=0.99, batch_shuffle=True)
        # create our loss with the memory bank
        self.criterion = loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)
        # create co2 regularizer with memory bank
        self.co2 = CO2Regularizer(alpha=alpha, memory_bank_size=memory_bank_size)
            
    def forward(self, x):
        self.resnet_moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        y0, y1 = self.resnet_moco(x0, x1)
        z0, z1 = self.resnet_moco(x1, x0)
        loss = 0.5 * (self.criterion(y0, y1) + self.criterion(z0, z1))
        loss = loss + self.co2(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


seed = 1234

# train moco with co2
pl.seed_everything(seed)
dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
benchmark_model = MocoModel(dataloader_train_kNN, 1, 10, knn_k, knn_t, 1.0)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
trainer.fit(
    benchmark_model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)
print(f'MoCo w/ CO2: {benchmark_model.max_accuracy}')

# train moco without co2
pl.seed_everything(seed)
dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
benchmark_model = MocoModel(dataloader_train_kNN, 1, 10, knn_k, knn_t, 0.0)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
trainer.fit(
    benchmark_model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)
print(f'MoCo w/o CO2: {benchmark_model.max_accuracy}')