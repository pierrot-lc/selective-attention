import hydra
import jax.random as random
import src.trainer as trainer
from configs.template import (
    DecoderOnlyTransformerConfig,
    LabMLConfig,
    MainConfig,
    ShakespearDatasetConfig,
    TrainerConfig,
)
from hydra.core.config_store import ConfigStore
from labml import experiment
from omegaconf import DictConfig, OmegaConf
from src.datasets import ShakespearDataset
from src.model import DecoderOnlyTransformer

cs = ConfigStore.instance()
cs.store(name="main-config", node=MainConfig)


@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(dict_config: DictConfig):
    config = MainConfig(
        dataset=ShakespearDatasetConfig(**dict_config.dataset),
        model=DecoderOnlyTransformerConfig(**dict_config.model),
        trainer=TrainerConfig(**dict_config.trainer),
        labml=LabMLConfig(**dict_config.labml),
    )
    dataset = ShakespearDataset.from_file(
        config.dataset.filepath, config.dataset.seq_len
    )
    key = random.key(config.trainer.seed)

    key, sk = random.split(key)
    model = DecoderOnlyTransformer(
        dataset.vocab_size,
        config.model.d_model,
        config.model.num_heads,
        config.model.num_layers,
        dataset.vocab_size,
        sk,
    )

    with experiment.record(
        name="test",
        exp_conf=OmegaConf.to_container(dict_config),
        app_url=config.labml.app_url,
    ):
        trainer.train(
            model,
            dataset,
            config.trainer.n_training_iter,
            config.trainer.batch_size,
            key,
        )


if __name__ == "__main__":
    # Launch with hydra.
    main()
