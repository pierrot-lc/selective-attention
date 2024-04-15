from collections import Counter
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Int, jaxtyped


class ShakespearDataset(eqx.Module):
    """A dataset for the Shakespear text generation task.

    This dataset will encode the text into integers and provide sequences of
    tokens of a given length. You can use it to extract random sequences of tokens.
    It does not use a complicated tokenization scheme, but a raw character-based encoding.
    """
    text: str
    encoded_text: Int[Array, " total_characters"]
    uniq_chars: list[str]
    char_to_int: dict[str, int]
    int_to_char: dict[int, str]
    vocab_size: int
    seq_len: int

    def __init__(self, text: str, seq_len: int, uniq_chars: Optional[list[str]] = None):
        self.text = text
        self.seq_len = seq_len

        if uniq_chars is None:
            uniq_chars = list(sorted(Counter(text)))

        self.uniq_chars = uniq_chars
        self.vocab_size = len(uniq_chars)

        self.char_to_int = {char: id for id, char in enumerate(uniq_chars)}
        self.int_to_char = {id: char for id, char in self.char_to_int.items()}

        encoded_text = [self.char_to_int[char] for char in text]
        self.encoded_text = jnp.array(encoded_text)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, i: Int[Array, ""]) -> Int[Array, " seq_len"]:
        """Return the ith sequence of tokens."""
        tokens = self.encoded_text[i : i + self.seq_len]
        return tokens

    def __len__(self) -> int:
        """Return the number of sequences that can be extracted from the text."""
        return max(len(self.encoded_text) - self.seq_len + 1, 1)

    def split(
        self, split_ratio: float
    ) -> tuple["ShakespearDataset", "ShakespearDataset"]:
        """Split the dataset chronologicaly.

        The mapping between characters and integers remains the same.
        """
        split_idx = int(len(self) * split_ratio)

        split1 = ShakespearDataset(self.text[:split_idx], self.seq_len, self.uniq_chars)
        split2 = ShakespearDataset(self.text[split_idx:], self.seq_len, self.uniq_chars)

        return split1, split2

    @classmethod
    def from_file(cls, filepath: Path, seq_len: int) -> "ShakespearDataset":
        with open(filepath, "r") as file:
            text = file.readlines()

        text = "".join(text)
        return cls(text, seq_len)
