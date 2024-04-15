# Selective Attention Transformer

This repository is an easy-to-read `jax` implementation of GPT. It is designed
to be easily understandable and hackable. This can be used as a starting point
to implement your own transformer models.

The code also implements an alternative version of the standard attention
mechanism that I called *selective attention*. The idea is that every key and
values are generated per-query. This means that while queries have their usual
shapes of `[q_seq, d_model]`, the keys and values have shapes ` [q_seq, kv_seq,
d_model]`. The initial motivation for this was that the usual keys and values
are very general and the same for every query. This means that information is
shared non-efficiently. The selective attention mechanism allows for more
fine-grained control over the information being shared.

But it turns out that the selective attention mechanism does not bring any
improvements over the standard attention mechanism. This is a bit disappointing
but I think this is still an interesting idea to explore, and a good `jax`
tutorial (at least for me).

## Implementation details

The learning dataset is the simple [Shakespeare
plays](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays). No
complex tokenization scheme is used. Every characters are mapped to its
dedicated integer. This is not efficient but it remains simple which is the
goal of this repository.

[RoPE](https://arxiv.org/abs/2104.09864) is used as positional encoding.

The shapes are checked using `beartype` and `jaxtyping`.

## Experiments

## How to use

### Install the dependencies

With `nix` (flakes required):

```console
nix develop
python3 -m venv .venv
source .venv/bin/activate
pdm sync
```

Otherwise:

```console
python3 -m venv .venv
source .venv/bin/activate
pip install pdm
pdm sync
```

Note that this will install `jax` for Nvidia GPUs with CUDA being installed
through pip. If you want to use another configuration of `jax`, you can find
the installation instructions
[here](https://jax.readthedocs.io/en/latest/installation.html).

Every dependencies are listed in the `pyproject.toml`, I use `pdm` as a package
manager but you can use whichever you want. I used python 3.12 but this code
should run fine for any python 3.11 or above.

### Run experiments

If you want to log the experiments, make sure to have a
[wandb](https://www.wandb.ai) account. The repo use
[hydra](https://hydra.cc/docs/intro/). The basic training (using the default
hyperparameters) can be run with:

```console
python3 main.py mode=offline
```

Most of the options are exposed in the `./configs/default.yaml` configuration
file which is read by hydra. You can modify this file or pass the options
directly to the command line.

Specify whether you want to log your experiments with `mode=online` or
`mode=offline`. You can specify the attention mechanism to use with
`model.mha_type={standard,selective,equinox}`. The `equinox` version is used as
a reference in this repository, but note that it does not use any positional
encoding scheme. For `standard` and `selective`, you can specify if you want
to use `rope` or not.
