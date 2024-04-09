from typing import Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as random
from beartype import beartype
from einops import rearrange
from jaxtyping import Array, Bool, Float, jaxtyped


@jax.jit
@jaxtyped(typechecker=beartype)
def qkv_attention(
    q: Float[Array, "q_seq d_model"],
    k: Float[Array, "kv_seq d_model"],
    v: Float[Array, "kv_seq d_model"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
) -> Float[Array, "q_seq d_model"]:
    logits = jnp.einsum("ij,kj->ik", q, k)
    logits = logits / jnp.sqrt(k.shape[0])

    attn_distrib = jax.nn.softmax(logits, axis=1, where=mask, initial=-jnp.inf)
    attn_result = jnp.einsum("ij,jk->ik", attn_distrib, v)

    return attn_result


class MultiheadAttention(eqx.Module):
    project_q: nn.Linear
    project_k: nn.Linear
    project_v: nn.Linear
    num_heads: int

    def __init__(self, num_heads: int, d_model: int, key: random.PRNGKey):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads

        sk = random.split(key, 3)
        self.project_q = nn.Linear(d_model, d_model, use_bias=False, key=sk[0])
        self.project_k = nn.Linear(d_model, d_model, use_bias=False, key=sk[1])
        self.project_v = nn.Linear(d_model, d_model, use_bias=False, key=sk[2])

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        q: Float[Array, "seq_len d_model"],
        k: Float[Array, "seq_len d_model"],
        v: Float[Array, "seq_len d_model"],
        mask: Bool[Array, "seq_len seq_len"],
    ) -> Float[Array, "seq_len d_model"]:
        q = jax.vmap(self.project_q)(q)
        k = jax.vmap(self.project_k)(k)
        v = jax.vmap(self.project_v)(v)

        q_heads = rearrange(q, "s (n d) -> n s d", n=self.num_heads)
        k_heads = rearrange(k, "s (n d) -> n s d", n=self.num_heads)
        v_heads = rearrange(v, "s (n d) -> n s d", n=self.num_heads)

        # Do not vmap the mask. The mask is the same accross all heads.
        qkv_attention_heads = jax.vmap(qkv_attention, in_axes=(0, 0, 0, None))
        attn_result = qkv_attention_heads(q_heads, k_heads, v_heads, mask)
        attn_result = rearrange(attn_result, "n s d -> s (n d)")
        return attn_result
