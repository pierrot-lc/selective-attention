import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as random
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from .mha import MultiheadAttention, MultiheadSelectiveAttention


class DecoderLayer(eqx.Module):
    """A transformer decoder-only block.

    You can specify the attention to use (`mha_type`):
        - "selective": Use the selective attention.
        - "standard": Use the standard attention, implemented here.
        - "equinox": Use the standard attention, implemented by equinox.
    """

    mha: MultiheadAttention | MultiheadSelectiveAttention | nn.MultiheadAttention
    ffn: nn.Sequential
    norm_1: nn.LayerNorm
    norm_2: nn.LayerNorm

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mha_type: str,
        rope: bool,
        key: random.PRNGKey,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        key, sk = random.split(key)
        match mha_type:
            case "selective":
                self.mha = MultiheadSelectiveAttention(num_heads, d_model, rope, key=sk)
            case "normal":
                self.mha = MultiheadAttention(num_heads, d_model, rope, key=sk)
            case "equinox":
                self.mha = nn.MultiheadAttention(num_heads, d_model, key=sk)
            case _:
                raise ValueError(f"Unknown mha_type: {mha_type}")

        sk_1, sk_2 = random.split(key, 2)
        self.ffn = nn.Sequential(
            [
                nn.Linear(d_model, d_model * 4, use_bias=False, key=sk_1),
                nn.Lambda(jax.nn.relu),
                nn.Linear(4 * d_model, d_model, use_bias=False, key=sk_2),
            ]
        )

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        mask: Bool[Array, "seq_len seq_len"],
    ) -> Float[Array, "seq_len d_model"]:
        """Apply the decoder layer to the input. The attention between token i and j
        is computed where the mask[i, j] is true.
        """
        x_att = self.mha(x, x, x, mask)
        x = jax.vmap(self.norm_1)(x + x_att)

        x_ffn = jax.vmap(self.ffn)(x)
        x = jax.vmap(self.norm_2)(x + x_ffn)

        return x


class DecoderTransformer(eqx.Module):
    """A transformer decoder-only model.

    You can specify the attention to use (`mha_type`):
        - "selective": Use the selective attention.
        - "standard": Use the standard attention, implemented here.
        - "equinox": Use the standard attention, implemented by equinox.
    """

    layers: nn.Sequential
    embedding: nn.Embedding
    logits: nn.Linear

    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_heads: int,
        mha_type: str,
        rope: bool,
        num_layers: int,
        num_logits: int,
        key: random.PRNGKey,
    ):
        super().__init__()

        key, sk = random.split(key)
        self.embedding = nn.Embedding(num_embeddings, d_model, key=sk)

        key, *subkeys = random.split(key, num_layers + 1)
        self.layers = nn.Sequential(
            [DecoderLayer(d_model, num_heads, mha_type, rope, sk) for sk in subkeys]
        )

        self.logits = nn.Linear(d_model, num_logits, key=key)

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Int[Array, " seq_len"]) -> Float[Array, "seq_len d_model"]:
        """Apply the decoder to the input sequence. The mask used is a causal mask."""
        mask = jnp.eye(x.shape[0], dtype=int)
        mask = jnp.cumsum(mask, axis=1).T
        mask = mask.astype(bool)

        x = jax.vmap(self.embedding)(x)

        for decoder_layer in self.layers:
            x = decoder_layer(x, mask)

        x = jax.vmap(self.logits)(x)
        return x
