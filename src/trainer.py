import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from einops import rearrange

from .model import TranslationModel


class Trainer:
    def __init__(
        self, model: TranslationModel, dataloader: DataLoader, optimizer: Optimizer
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer

        self.loss = nn.CrossEntropyLoss()

    def launch_training(self):
        self.model.to("cuda")

        for src, tgt in self.dataloader:
            src, tgt = src.to("cuda"), tgt.to("cuda")

            tgt_pred = self.model(src, tgt)

            tgt = rearrange(tgt, "b l -> (b l)")
            tgt_pred = rearrange(tgt_pred, "b l v -> (b l) v")

            loss = self.loss(tgt_pred, tgt)
            loss.backward()
            self.optimizer.step()

            print(loss.item())
