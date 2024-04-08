from collections import Counter
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Int, jaxtyped


class ShakespearDataset(eqx.Module):
    text: str
    encoded_text: Int[Array, "total_characters"]
    char_to_int: dict[str, int]
    int_to_char: dict[int, str]
    vocab_size: int
    seq_len: int

    def __init__(self, text: str, seq_len: int):
        self.text = text
        self.seq_len = seq_len

        uniq_chars = list(sorted(Counter(text)))
        self.vocab_size = len(uniq_chars)
        self.char_to_int = {char: id for id, char in enumerate(uniq_chars)}
        self.int_to_char = {id: char for id, char in self.char_to_int.items()}

        encoded_text = [self.char_to_int[char] for char in text]
        self.encoded_text = jnp.array(encoded_text)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, i: Int[Array, ""]) -> Int[Array, "seq_len"]:
        tokens = self.encoded_text[i : i + self.seq_len]
        return tokens

    def __len__(self) -> int:
        return max(len(self.encoded_text) - self.seq_len + 1, 1)

    @classmethod
    def from_file(cls, filepath: Path, seq_len: int) -> "ShakespearDataset":
        with open(filepath, "r") as file:
            text = file.readlines()

        text = "".join(text)
        return cls(text, seq_len)
