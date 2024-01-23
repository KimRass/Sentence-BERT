# References
    # http://ixa2.si.ehu.eus/stswiki/index.php/STSb

import torch
from torch.utils.data import Dataset


class STSbDataset(Dataset):
    @staticmethod
    def _normalize_score(score):
        score -= 2.5
        score /= 2.5
        return score

    @staticmethod
    def _get_length(x):
        return (x != 0).sum(dim=1)

    def __init__(self, csv_path, tokenizer, max_len):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.scores, sents1, sents2 = list(), list(), list()
        with open(csv_path, mode="r") as f:
            for line in f:
                line = line.split("\t")
                if len(line) == 7:
                    line = line[4:]
                elif len(line) == 9:
                    line = line[4: -2]
                score, sent1, sent2 = line
                score = float(score)
                score = self._normalize_score(score)

                self.scores.append(score)
                sents1.append(sent1)
                sents2.append(sent2)
                
        self.seq1 = torch.as_tensor(
            tokenizer(sents1, padding="max_length", max_length=max_len)["input_ids"]
        )
        self.seq2 = torch.as_tensor(
            tokenizer(sents2, padding="max_length", max_length=max_len)["input_ids"]
        )

        # "We implemented a smart batching strategy: Sentences with similar lengths are grouped together"
        # in the section 7 of the paper.
        order = torch.argsort(
            self._get_length(self.seq1) + self._get_length(self.seq1)
        )
        self.seq1 = self.seq1[order]
        self.seq2 = self.seq2[order]

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.scores[idx], self.seq1[idx], self.seq2[idx]


class STSbCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        scores, sents1, sents2 = list(), list(), list()
        for score, sent1, sent2 in batch:
            scores.append(score)
            sents1.append(sent1)
            sents2.append(sent2)

        scores, sents1, sents2 = torch.as_tensor(scores), torch.stack(sents1), torch.stack(sents2)
        # "(Sentences with similar lengths) are only padded to the longest element in a mini-batch.
        # This drastically reduces computational overhead from padding tokens."
        sents1, sents2 = (
            sents1[:, : (sents1 != self.tokenizer.pad_token_id).sum(dim=1).max()],
            sents2[:, : (sents2 != self.tokenizer.pad_token_id).sum(dim=1).max()]
        )
        return scores, sents1, sents2
