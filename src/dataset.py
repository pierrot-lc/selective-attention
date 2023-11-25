from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator

SPECIALS = ["<unk>", "<pad>", "<bos>", "<eos>"]


class TranslationDataset(Dataset):
    def __init__(
        self,
        french_sentences: list[str],
        english_sentences: list[str],
        french_vocab: Vocab,
        english_vocab: Vocab,
        french_tokenizer: callable,
        english_tokenizer: callable,
    ):
        super().__init__()

        self.french_sentences = french_sentences
        self.english_sentences = english_sentences

        self.french_vocab = french_vocab
        self.english_vocab = english_vocab

        self.french_tokenizer = french_tokenizer
        self.english_tokenizer = english_tokenizer

    def __len__(self) -> int:
        return len(self.french_sentences)

    def __getitem__(self, index: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        fr_sentence = self.french_sentences[index]
        en_sentence = self.english_sentences[index]

        fr_tokens = ["<bos>"] + self.french_tokenizer(fr_sentence) + ["<eos>"]
        en_tokens = ["<bos>"] + self.english_tokenizer(en_sentence) + ["<eos>"]

        fr_tokens = self.french_vocab(fr_tokens)
        en_tokens = self.english_vocab(en_tokens)

        fr_tokens = torch.tensor(fr_tokens)
        en_tokens = torch.tensor(en_tokens)
        return fr_tokens, en_tokens

    @classmethod
    def from_file(cls, filepath: Path) -> "TranslationDataset":
        df = pd.read_csv(filepath, sep="\t", names=["english", "french", "attribution"])

        en_tokenizer = get_tokenizer("spacy", "en_core_web_sm")
        fr_tokenizer = get_tokenizer("spacy", "fr_core_news_sm")

        en_vocab = build_vocab_from_iterator(
            (en_tokenizer(sentence) for sentence in df["english"].values),
            min_freq=2,
            specials=SPECIALS,
        )
        fr_vocab = build_vocab_from_iterator(
            (fr_tokenizer(sentence) for sentence in df["french"].values),
            min_freq=2,
            specials=SPECIALS,
        )

        fr_vocab.set_default_index(fr_vocab["<unk>"])
        en_vocab.set_default_index(en_vocab["<unk>"])

        return cls(
            df["french"].values,
            df["english"].values,
            fr_vocab,
            en_vocab,
            fr_tokenizer,
            en_tokenizer,
        )

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.LongTensor, torch.LongTensor]],
        src_pad_idx: int,
        tgt_pad_idx: int,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        src_batch, tgt_batch = [], []
        for src_tokens, tgt_tokens in batch:
            src_batch.append(src_tokens)
            tgt_batch.append(tgt_tokens)

        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
        return src_batch, tgt_batch
