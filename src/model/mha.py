from typing import Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as random
from beartype import beartype
from einops import rearrange
from jaxtyping import Array, Bool, Float, jaxtyped

from .rope import RoPE


@jax.jit
@jaxtyped(typechecker=beartype)
def qkv_attention(
    q: Float[Array, "q_seq d_model"],
    k: Float[Array, "kv_seq d_model"],
    v: Float[Array, "kv_seq d_model"],
    mask: Bool[Array, "q_seq kv_seq"],
) -> Float[Array, "q_seq d_model"]:
    """Compute the attention between q and k, and apply it to v.
    This is the standard attention mechanism. Attention is only applied
    where mask is true. q, k and v should already be projected.

    Paper: Attention is All You Need - https://arxiv.org/abs/1706.03762
    """
    logits = jnp.einsum("ij,kj->ik", q, k)  # Dot-product attention.
    logits = logits / jnp.sqrt(k.shape[1])

    attn_distrib = jax.nn.softmax(logits, axis=1, where=mask, initial=-jnp.inf)
    attn_result = jnp.einsum("ij,jk->ik", attn_distrib, v)  # Apply attention to v.

    return attn_result


@jax.jit
@jaxtyped(typechecker=beartype)
def qkv_selective_attention(
    q: Float[Array, "q_seq d_model"],
    k: Float[Array, "q_seq kv_seq d_model"],
    v: Float[Array, "q_seq kv_seq d_model"],
    mask: Bool[Array, "q_seq kv_seq"],
) -> Float[Array, "q_seq d_model"]:
    """Compute the attention between every q and their corresponding k and v.
    Every query has its own sequence of keys and values to which it will attend.
    """
    q = rearrange(q, "s d -> s 1 d")
    mask = rearrange(mask, "s t -> s 1 t")
    attn_result = jax.vmap(qkv_attention)(q, k, v, mask)
    attn_result = rearrange(attn_result, "s 1 d -> s d")
    return attn_result


@jax.jit
@jaxtyped(typechecker=beartype)
def cross_product_matching(
    query: Float[Array, "q_seq q_size"], other: Float[Array, "o_seq o_size"]
) -> Float[Array, "q_seq o_seq q_size+o_size"]:
    """Concatenate every query element to every other element."""
    cat = lambda v1, v2: jnp.concatenate((v1, v2), axis=0)  # noqa: E731
    cat = jax.vmap(cat, in_axes=(None, 0))
    cat = jax.vmap(cat, in_axes=(0, None))
    return cat(query, other)


class MultiheadAttention(eqx.Module):
    """Project q, k and v before applying the standard attention.
    Optionally apply rotary positional encoding to q and k.
    """

    project_q: nn.Linear
    project_k: nn.Linear
    project_v: nn.Linear
    rope: Optional[RoPE]
    num_heads: int

    def __init__(self, num_heads: int, d_model: int, rope: bool, key: random.PRNGKey):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads

        sk = random.split(key, 3)
        self.project_q = nn.Linear(d_model, d_model, use_bias=False, key=sk[0])
        self.project_k = nn.Linear(d_model, d_model, use_bias=False, key=sk[1])
        self.project_v = nn.Linear(d_model, d_model, use_bias=False, key=sk[2])

        self.rope = RoPE(d_model // num_heads, max_seq_len=10000) if rope else None

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        q: Float[Array, "seq_len d_model"],
        k: Float[Array, "seq_len d_model"],
        v: Float[Array, "seq_len d_model"],
        mask: Bool[Array, "seq_len seq_len"],
    ) -> Float[Array, "seq_len d_model"]:
        # Project q k and v.
        q = jax.vmap(self.project_q)(q)
        k = jax.vmap(self.project_k)(k)
        v = jax.vmap(self.project_v)(v)

        # Separate heads.
        q_heads = rearrange(q, "s (n d) -> n s d", n=self.num_heads)
        k_heads = rearrange(k, "s (n d) -> n s d", n=self.num_heads)
        v_heads = rearrange(v, "s (n d) -> n s d", n=self.num_heads)

        if self.rope is not None:
            # Apply RoPE.
            rope = jax.vmap(self.rope)
            q_heads = rope(q_heads)
            k_heads = rope(k_heads)

        # Do not vmap the mask. The mask is the same accross all heads.
        multihead_qkv_attention = jax.vmap(qkv_attention, in_axes=(0, 0, 0, None))
        attn_result = multihead_qkv_attention(q_heads, k_heads, v_heads, mask)
        attn_result = rearrange(attn_result, "n s d -> s (n d)")
        return attn_result


class MultiheadSelectiveAttention(eqx.Module):
    """Project q, k and v before applying the selective attention.
    Optionally apply rotary positional encoding to q and k.
    """

    project_q: nn.Linear
    project_k: nn.Linear
    project_v: nn.Linear
    rope: Optional[RoPE]
    num_heads: int

    def __init__(self, num_heads: int, d_model: int, rope: bool, key: random.PRNGKey):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads

        sk = random.split(key, 3)
        self.project_q = nn.Linear(d_model, d_model, use_bias=False, key=sk[0])
        self.project_k = nn.Linear(2 * d_model, d_model, use_bias=False, key=sk[1])
        self.project_v = nn.Linear(2 * d_model, d_model, use_bias=False, key=sk[2])

        self.rope = RoPE(d_model // num_heads, max_seq_len=10000) if rope else None

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        q: Float[Array, "seq_len d_model"],
        k: Float[Array, "seq_len d_model"],
        v: Float[Array, "seq_len d_model"],
        mask: Bool[Array, "seq_len seq_len"],
    ) -> Float[Array, "seq_len d_model"]:
        # First compute the queries.
        q = jax.vmap(self.project_q)(q)

        # Match every k and v to every q.
        # Shape of [q_seq, kv_seq, d_model].
        k = cross_product_matching(q, k)
        v = cross_product_matching(q, v)

        # Project k and v.
        k = jax.vmap(jax.vmap(self.project_k))(k)
        v = jax.vmap(jax.vmap(self.project_v))(v)

        # Separate heads.
        # Shape of [num_heads, q_seq, kv_seq, d_model // num_heads].
        q_heads = rearrange(q, "s (n d) -> n s d", n=self.num_heads)
        k_heads = rearrange(k, "s1 s2 (n d) -> n s1 s2 d", n=self.num_heads)
        v_heads = rearrange(v, "s1 s2 (n d) -> n s1 s2 d", n=self.num_heads)

        if self.rope is not None:
            # Apply RoPE.
            rope = jax.vmap(self.rope)  # Multiple heads.
            q_heads = rope(q_heads)
            rope = jax.vmap(rope)  # Additional q_seq dimension.
            k_heads = rope(k_heads)

        # Finally compute cubic attention.
        multihead_qkv_attention = jax.vmap(
            qkv_selective_attention, in_axes=(0, 0, 0, None)
        )
        attn_result = multihead_qkv_attention(q_heads, k_heads, v_heads, mask)
        attn_result = rearrange(attn_result, "n s d -> s (n d)")
        return attn_result
