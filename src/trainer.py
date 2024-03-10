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
        self.device = "cuda"

    def launch_training(self):
        self.model.to(self.device)

        for src, tgt in self.dataloader:
            src, tgt = src.to(self.device), tgt.to(self.device)

            tgt_pred = self.model(src, tgt)

            # Shift the tokens to predicts by one, so that
            # we do not predict the present.
            tgt = tgt[:, 1:]
            tgt_pred = tgt_pred[:, :-1]

            tgt = rearrange(tgt, "b l -> (b l)")
            tgt_pred = rearrange(tgt_pred, "b l v -> (b l) v")

            loss = self.loss(tgt_pred, tgt)
            loss.backward()
            self.optimizer.step()

            print(loss.item())
