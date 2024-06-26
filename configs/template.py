from dataclasses import dataclass
from pathlib import Path

from hydra.utils import to_absolute_path


@dataclass
class ShakespearDatasetConfig:
    seq_len: int
    filepath: Path = Path("./data/shakespeare.txt")

    def __post_init__(self):
        self.filepath = Path(to_absolute_path(str(self.filepath)))


@dataclass
class DecoderTransformerConfig:
    d_model: int
    num_heads: int
    mha_type: str
    rope: bool
    num_layers: int


@dataclass
class TrainerConfig:
    learning_rate: float
    batch_size: int
    n_training_iter: int
    n_eval_iter: int
    seed: int = 42

@dataclass
class MainConfig:
    dataset: ShakespearDatasetConfig
    model: DecoderTransformerConfig
    trainer: TrainerConfig
