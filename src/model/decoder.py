import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()

        self.encoder = nn.TransformerDecoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_layers,
        )
