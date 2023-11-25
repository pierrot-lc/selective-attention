import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D


class TranslationModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=src_pad_idx,
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=tgt_pad_idx,
        )

        self.pos_encodings = PositionalEncoding1D(d_model)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self, src: torch.LongTensor, tgt: torch.LongTensor
    ) -> torch.FloatTensor:
        src = self.src_embedding(src)
        src = src + self.pos_encodings(src)

        tgt = self.tgt_embedding(tgt)
        tgt = tgt + self.pos_encodings(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[1], device=tgt.device
        )

        tgt = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_is_causal=True)

        return self.head(tgt)
