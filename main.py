from functools import partial

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import TranslationDataset
from src.model import TranslationModel
from src.trainer import Trainer


def main():
    dataset = TranslationDataset.from_file("./data/fra-eng/fra-eng.txt")
    model = TranslationModel(
        len(dataset.french_vocab),
        len(dataset.english_vocab),
        dataset.french_vocab["<pad>"],
        dataset.english_vocab["<pad>"],
        92,
        nhead=2,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=92 * 4,
        dropout=0.05,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        collate_fn=partial(
            dataset.collate_fn,
            src_pad_idx=dataset.french_vocab["<pad>"],
            tgt_pad_idx=dataset.english_vocab["<pad>"],
        ),
    )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    trainer = Trainer(model, dataloader, optimizer)

    trainer.launch_training()


if __name__ == "__main__":
    main()
