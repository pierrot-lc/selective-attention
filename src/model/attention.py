import einops
import torch
import torch.nn as nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
):
    return nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p
    )
