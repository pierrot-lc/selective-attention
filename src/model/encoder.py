import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(num_layers)
            ]
        )

        self.self_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model, bias=False) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mlp, self_attention, norm in zip(
            self.mlps, self.self_attentions, self.norms
        ):
            x_attn = norm(x)
            x_attn, _ = self_attention(x_attn, x_attn, x_attn)
            x = x + x_attn

            x_mlp = norm(x)
            x_mlp = mlp(x_mlp)
            x = x + x_mlp

        return x
