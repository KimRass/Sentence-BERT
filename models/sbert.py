# References:
    # BERT from; https://github.com/KimRass/BERT/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import torch.linalg as LA

from models.bert import BERT


def pool(x, pooler):
    if pooler == "max":
        x = x[:, 1: -1, :]
        x = torch.max(x, dim=1)[0]
    elif pooler == "mean":
        x = x[:, 1: -1, :]
        x = torch.mean(x, dim=1)
    elif pooler == "cls":
        x = x[:, 0, :]
    return x


class SentenceBERTForCls(nn.Module):
    def __init__(
        self,
        embedder,
        n_classes,
        pooler: Literal["mean", "max", "cls"]="mean",
    ):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler

        self.proj = nn.Linear(embedder.hidden_size * 3, n_classes) # "$W_{t}$"

    def forward(self, seq1, seq2):
        seq1 = self.embedder(seq1, seg_ids=torch.zeros_like(seq1))
        seq2 = self.embedder(seq2, seg_ids=torch.zeros_like(seq2))

        seq1 = pool(seq1, pooler=self.pooler)
        seq2 = pool(seq2, pooler=self.pooler)
        return seq1, seq2

    def get_loss(self, seq1, seq2, gt):
        seq1, seq2 = self(seq1=seq1, seq2=seq2)
        x = torch.cat([seq1, seq2, torch.abs(seq1 - seq2)], dim=1)
        x = self.proj(x)
        loss = F.cross_entropy(x, gt, reduction="mean")
        return loss


class SentenceBERTForReg(nn.Module):
    def __init__(self, embedder, pooler: Literal["mean", "max", "cls"]="mean"):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler

    def forward(self, seq1, seq2):
        seq1 = self.embedder(seq1, seg_ids=torch.zeros_like(seq1))
        seq2 = self.embedder(seq2, seg_ids=torch.zeros_like(seq2))

        seq1 = pool(seq1, pooler=self.pooler)
        seq2 = pool(seq2, pooler=self.pooler)
        return seq1, seq2

    def get_loss(self, seq1, seq2, gt):
        seq1, seq2 = self(seq1=seq1, seq2=seq2)
        # "The cosinesimilarity between the two sentence embeddings $u$ and $v$ is computed."
        cos_sim = F.cosine_similarity(seq1, seq2)
        # "We use mean-squared-error loss as the objective function."
        loss = F.mse_loss(cos_sim, gt, reduction="mean")
        return loss


class SentenceBERTForContrastiveLearning(nn.Module):
    def __init__(self, embedder, pooler: Literal["mean", "max", "cls"]="mean", eps=1):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler
        self.eps = eps

    def forward(self, anc, pos, neg):
        anc = self.embedder(anc, seg_ids=torch.zeros_like(anc))
        pos = self.embedder(pos, seg_ids=torch.zeros_like(pos))
        neg = self.embedder(neg, seg_ids=torch.zeros_like(neg))

        anc = pool(anc, pooler=self.pooler)
        pos = pool(pos, pooler=self.pooler)
        neg = pool(neg, pooler=self.pooler)
        return anc, pos, neg

    def get_loss(self, anc, pos, neg):
        anc, pos, neg = self(anc=anc, pos=pos, neg=neg)
        x = LA.vector_norm(anc - pos, ord=2, dim=1) - LA.vector_norm(anc - neg, ord=2, dim=1) + self.eps
        loss = F.relu(x)
        return loss.mean(dim=0)


if __name__ == "__main__":
    VOCAB_SIZE = 30_522
    MAX_LEN = 128
    PAD_ID = 0
    bert = BERT(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, pad_id=PAD_ID)

    BATCH_SIZE = 4
    seq1 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    seq2 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    gt = torch.randn(BATCH_SIZE, 5)
    sbert_cls = SentenceBERTForCls(embedder=bert, n_classes=5)
    out = sbert_cls(seq1=seq1, seq2=seq2)
    loss = sbert_cls.get_loss(seq1=seq1, seq2=seq2, gt=gt)
    print(loss.shape)

    seq1 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    seq2 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    gt = torch.randn(BATCH_SIZE)
    sbert_reg = SentenceBERTForReg(embedder=bert)
    loss = sbert_reg.get_loss(seq1=seq1, seq2=seq2, gt=gt)
    print(loss.shape)

    anc = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    pos = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    neg = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    sbert_trip = SentenceBERTForContrastiveLearning(embedder=bert)
    loss = sbert_trip.get_loss(anc=anc, pos=pos, neg=neg)
    print(loss.shape)
