from pathlib import Path

import jax

from src.datasets import ShakespearDataset
from src.model.decoder_only import DecoderOnlyTransformer


def main():
    dataset = ShakespearDataset.from_file(Path("./data/shakespear.txt"), 10)
    tokens = dataset[0]
    key = jax.random.key(42)

    model = DecoderOnlyTransformer(
        dataset.vocab_size, 32, 2, 3, dataset.vocab_size, key
    )
    out = model(tokens)
    print(out.shape)

if __name__ == "__main__":
    main()
