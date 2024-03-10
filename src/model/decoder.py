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

        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model, bias=False) for _ in range(num_layers)]
        )
        self.memory_norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        memory = self.memory_norm(memory)

        for mlp, self_attention, cross_attention, norm in zip(
            self.mlps, self.self_attentions, self.cross_attentions, self.norms
        ):
            tgt_self_attn = norm(tgt)
            tgt_self_attn, _ = self_attention(
                tgt_self_attn, tgt_self_attn, tgt_self_attn, attn_mask=tgt_mask
            )
            tgt = tgt + tgt_self_attn

            tgt_cross_attention = norm(tgt)
            tgt_cross_attention, _ = cross_attention(
                tgt_cross_attention, memory, memory
            )
            tgt = tgt + tgt_cross_attention

            tgt_mlp = norm(tgt)
            tgt_mlp = mlp(tgt_mlp)
            tgt = tgt + tgt_mlp

        return tgt
