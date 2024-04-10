import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped


class RoPE(eqx.Module):
    cos: Float[Array, "max_seq_len d_model//2"]
    sin: Float[Array, "mas_seq_len d_model//2"]

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        assert d_model % 2 == 0

        dimensions = jnp.arange(0, d_model // 2)
        thetas = 10000 ** (-2 * dimensions / d_model)
        positions = jnp.arange(0, max_seq_len)
        pos_product = jax.vmap(lambda m, t: m * t, in_axes=(0, None))
        pos_thetas = pos_product(positions, thetas)

        self.cos = jnp.cos(pos_thetas)
        self.sin = jnp.sin(pos_thetas)

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, "seq_len d_model"]
    ) -> Float[Array, "seq_len d_model"]:
        """Apply the rotary positional encoding."""
        seq_len = x.shape[0]
        x1, x2 = jnp.split(x, 2, axis=1)

        x1_rot = x1 * self.cos[:seq_len] - x2 * self.sin[:seq_len]
        x2_rot = x1 * self.sin[:seq_len] + x2 * self.cos[:seq_len]

        x = jnp.concat((x1_rot, x2_rot), axis=1)
        return jax.lax.stop_gradient(x)