from functools import partial
from itertools import batched
from typing import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped
from optax.losses import softmax_cross_entropy_with_integer_labels
import optax

from .datasets import ShakespearDataset
from .model import DecoderOnlyTransformer


def loader(dataset: Iterable, batch_size: int, key: random.PRNGKey):
    """Yield batches of samples from the dataset."""
    sample_ids = random.permutation(key, len(dataset))

    for batch_ids in batched(sample_ids, batch_size):
        batch_samples = [dataset[id_] for id_ in batch_ids]
        batch_samples = jnp.stack(batch_samples)
        yield batch_samples


@jaxtyped(typechecker=beartype)
@partial(jax.jit, static_argnums=1)
def loss_fn(
    params: eqx.Module,
    static: eqx.Module,
    tokens: Int[Array, "batch_size seq_len"],
) -> Float[Array, ""]:
    x, y = tokens[:, :-1], tokens[:, 1:]
    model = eqx.combine(params, static)
    y_logits = jax.vmap(model)(x)
    loss = softmax_cross_entropy_with_integer_labels(y_logits, y)
    return jnp.mean(loss)


def train(
    model: DecoderOnlyTransformer,
    dataset: ShakespearDataset,
    batch_size: int,
    key: random.PRNGKey,
):
    grad_fn = jax.grad(loss_fn)
    params, static = eqx.partition(model, eqx.is_array)
    optimizer = optax.adamw(1e-4)
    opt_state = optimizer.init(params)

    for _ in range(1):
        key, sk = random.split(key)
        for batch in loader(dataset, batch_size, sk):
            grads = grad_fn(params, static, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
