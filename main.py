from pathlib import Path

import jax
import jax.random as random

from src.datasets import ShakespearDataset
from src.model import DecoderOnlyTransformer
import src.trainer as trainer


def main():
    dataset = ShakespearDataset.from_file(Path("./data/shakespear.txt"), 10)
    key = random.key(42)
    batch_size = 64
    n_iters = 10000

    key, sk = random.split(key)
    model = DecoderOnlyTransformer(
        dataset.vocab_size, 32, 2, 3, dataset.vocab_size, sk
    )

    trainer.train(model, dataset, n_iters, batch_size, key)

if __name__ == "__main__":
    main()
