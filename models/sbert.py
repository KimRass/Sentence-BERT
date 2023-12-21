# References:
    # BERT from; https://github.com/KimRass/BERT/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from models.bert import BERT


def _pool(x, pooler):
    if pooler == "mean":
        x = x[:, 1: -1, :]
        x = torch.max(x, dim=1)[0]
    elif pooler == "mean":
        x = x[:, 1: -1, :]
        x = torch.mean(x, dim=1)
    else:
        x = x[:, 0, :]
    return x


class SentenceBERTForCls(nn.Module):
    def __init__(self, embedder, pooler: Literal["mean", "max", "cls"]="mean"):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler

        self.proj = nn.Linear(embedder.hidden_size * 3, 3) # $W_{t}$ in the paper.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, p, h):
        x1, x2 = (
            self.embedder(p, seg_ids=torch.zeros_like(p)),
            self.embedder(h, seg_ids=torch.zeros_like(h))
        )
        x1, x2 = (
            _pool(x1, pooler=self.pooler),
            _pool(x2, pooler=self.pooler),
        )
        x = torch.cat([x1, x2, torch.abs(x1 - x2)], dim=1)
        x = self.proj(x)
        x = self.softmax(x)
        return x

    def _get_finetuned_embedder(self):
        return self.embedder


class SentenceBERTForReg(nn.Module):
    def __init__(self, embedder, pooler: Literal["mean", "max", "cls"]="mean"):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler

    def forward(self, sent1, sent2):
        x1, x2 = (
            self.embedder(sent1, seg_ids=torch.zeros_like(sent1)),
            self.embedder(sent2, seg_ids=torch.zeros_like(sent2)),
        )
        x1, x2 = (
            _pool(x1, pooler=self.pooler),
            _pool(x2, pooler=self.pooler)
        )
        x = F.cosine_similarity(x1, x2)
        return x

    def _get_finetuned_embedder(self):
        return self.embedder


class SentenceBERTForContrastiveLearning(nn.Module):
    def __init__(self, embedder, pooler: Literal["mean", "max", "cls"]="mean", eps=1):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler
        self.eps = eps

    def forward(self, a, p, n):
        a, p, n = (
            self.embedder(a, seg_ids=torch.zeros_like(a)),
            self.embedder(p, seg_ids=torch.zeros_like(p)),
            self.embedder(n, seg_ids=torch.zeros_like(n)),
        )
        a, p, n = (
            _pool(a, pooler=self.pooler),
            _pool(p, pooler=self.pooler),
            _pool(n, pooler=self.pooler),
        )
        return a, p, n

    def _get_finetuned_embedder(self):
        return self.embedder


if __name__ == "__main__":
    VOCAB_SIZE = 30_522
    MAX_LEN = 128
    PAD_ID = 0
    bert = BERT(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, pad_id=PAD_ID)

    BATCH_SIZE = 4
    sbert_cls = SentenceBERTForCls(embedder=bert)
    sent1 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    sent2 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    out = sbert_cls(p=sent1, h=sent2)
    print(out.shape)

    sbert_reg = SentenceBERTForReg(embedder=bert)
    out = sbert_reg(sent1=sent1, sent2=sent2)
    print(out.shape)

    sbert_trip = SentenceBERTForContrastiveLearning(embedder=bert)
    a = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    p = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    n = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    out = sbert_trip(a=a, p=p, n=n)
    for i in out:
        print(i.shape)
