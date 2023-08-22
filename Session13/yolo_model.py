#!/usr/bin/env python3
"""
Lightning Module for Yolo v3
"""

# Third-Party Imports
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

# Local Imports
import config
from utils import check_class_accuracy
from model import ScalePrediction, ResidualBlock, CNNBlock, model_config
from loss import YoloLoss
from dataset import YOLODataset


class YOLOv3(pl.LightningModule):
    """
    PyTorch Lightning Code for YOLOv3
    """

    def __init__(self, in_channels=3, num_classes=20):
        """
        Constructor
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

        self._learning_rate = 0.03
        self.loss_fn = YoloLoss()
        self.scaled_anchors = config.SCALED_ANCHORS
        self.combined_data = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self._data_directory = None
        self.epochs = config.NUM_EPOCHS
        self.batch_size = config.BATCH_SIZE
        self.enable_gc = "batch"
        self.my_train_loss = MeanMetric()
        self.my_val_loss = MeanMetric()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in model_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats, ))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2), )
                    in_channels = in_channels * 3

        return layers

    # ##################################################################################################
    # ############################## Training Configuration Related Hooks ##############################
    # ##################################################################################################
    def configure_optimizers(self):
        """
        Method to configure the optimizer and learning rate scheduler
        """
        optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self._learning_rate,
                                                        steps_per_epoch=len(self.train_dataloader()),
                                                        epochs=self.epochs,
                                                        pct_start=0.3,
                                                        div_factor=100,
                                                        three_phase=False,
                                                        final_div_factor=100,
                                                        anneal_strategy="linear"
                                                        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @property
    def learning_rate(self) -> float:
        """
        Method to get the learning rate value
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        """
        Method to set the learning rate value
        :param value: Updated value of learning rate
        """
        self._learning_rate = value

    @property
    def data_directory(self) -> str:
        """
        Method to return data directory
        """
        return self._data_directory

    @data_directory.setter
    def data_directory(self, address: str):
        """
        Method to set the data directory path
        """
        self._data_directory = address

    def set_training_config(self, *, epochs, batch_size):
        """
        Method to set parameters required for model training
        :param epochs: Number of epochs for which model is to be trained
        :param batch_size: Batch Size
        """
        self.epochs = epochs
        self.batch_size = batch_size

    # #################################################################################################
    # ################################## Training Loop Related Hooks ##################################
    # #################################################################################################
    def training_step(self, train_batch, batch_index):
        """
        Method called on training dataset to train the model
        :param train_batch: Batch containing images and labels
        :param batch_index: Index of the batch
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = (
                self.loss_fn(logits[0], y[0], self.scaled_anchors[0])
                + self.loss_fn(logits[1], y[1], self.scaled_anchors[1])
                + self.loss_fn(logits[2], y[2], self.scaled_anchors[2])
            )
        self.my_train_loss.update(loss, x.shape[0])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        # Check Accuracy
        check_class_accuracy(model=self.forward(x), loader=self.train_dataloader(), threshold=config.CONF_THRESHOLD)

        del x, y, logits
        return loss

    def validation_step(self, train_batch, batch_index):
        """
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = (
                self.loss_fn(logits[0], y[0], self.scaled_anchors[0])
                + self.loss_fn(logits[1], y[1], self.scaled_anchors[1])
                + self.loss_fn(logits[2], y[2], self.scaled_anchors[2])
            )
        self.my_val_loss.update(loss, x.shape[0])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # Check Accuracy
        check_class_accuracy(model=self.forward(x), loader=self.test_dataloader(), threshold=config.CONF_THRESHOLD)

        del x, y, logits
        return loss

    def test_step(self, batch, batch_idx):
        """
        """
        return self.validation_step(batch, batch_idx)

    # ###########################################################################################
    # ##################################### Data Related Hooks ##################################
    # ###########################################################################################

    def prepare_data(self):
        """
        Method to download the dataset
        """
        # Since data is already downloaded
        pass

    def setup(self, stage=None):
        """
        Method to create Split the dataset into train, test and val
        """
        train_csv_path = self._data_directory + config.DATASET + "/train.csv"
        test_csv_path = self._data_directory + config.DATASET + "/test.csv"
        IMG_DIR = self._data_directory + config.IMG_DIR
        LABEL_DIR = self._data_directory + config.LABEL_DIR
        IMAGE_SIZE = config.IMAGE_SIZE

        # Assign train/val datasets for use in dataloaders
        self.train_dataset = YOLODataset(
            train_csv_path,
            transform=config.train_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR,
            anchors=config.ANCHORS,
            mosaic=0.75
        )
        self.val_dataset = YOLODataset(
            train_csv_path,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR,
            anchors=config.ANCHORS,
        )

        # Assign test dataset for use in dataloader(s)
        self.test_dataset = YOLODataset(
            test_csv_path,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR,
            anchors=config.ANCHORS,
        )

    def train_dataloader(self):
        """
        Method to return the DataLoader for Training set
        """
        if not self.train_dataset:
            self.setup()
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        """
        Method to return the DataLoader for the Validation set
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        """
        Method to return the DataLoader for the Test set
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

    # ###########################################################################################
    # ##################################### Memory Related Hooks ##################################
    # ###########################################################################################

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Garbage Collection for memory optimization
            batch:
            batch_idx:

        Returns:

        """
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Garbage Collection for memory optimization
            batch:
            batch_idx:
            dataloader_idx:

        Returns:

        """
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Garbage Collection for memory optimization
            batch:
            batch_idx:
            dataloader_idx:

        Returns:

        """
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def on_train_epoch_end(self):
        """
        Garbage Collection for memory optimization
        """
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()
        print(
            f"Epoch: {self.current_epoch}, Global Steps: {self.global_step}, Train Loss: {self.my_train_loss.compute()}")
        self.my_train_loss.reset()

    def on_validation_epoch_end(self):
        """
        Garbage Collection for memory optimization
        """
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()
        print(f"Epoch: {self.current_epoch}, Global Steps: {self.global_step}, Val Loss: {self.my_val_loss.compute()}")
        self.my_val_loss.reset()

    def on_predict_epoch_end(self):
        """
        Garbage Collection for memory optimization
        """
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()
