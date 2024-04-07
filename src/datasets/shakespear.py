import equinox as eqx
from collections import Counter
from jaxtyping import Array, Int

class ShakespearDataset(eqx.Module):
    text: str
    encoded_text: Int[Array, "total_characters"]
    vocab_size: int

    def __init__(self, text: str):
        self.text = text

        uniq_chars = list(Counter(text))
