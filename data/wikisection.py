# References
    # https://github.com/sebastianarnold/WikiSection.git

import torch
import json
from fastapi.encoders import jsonable_encoder
from collections import defaultdict
from tqdm.auto import tqdm
import random
from copy import deepcopy


def _group_by_sections(wiki_data, tokenizer):
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    sec2sents = defaultdict(list)
    for block in tqdm(wiki_data):
        annot = block["annotations"]
        parag = block["text"]
        for section in annot:
            sent = parag[section["begin"]: section["begin"] + section["length"]]
            token_ids = [cls_id] + tokenizer.encode(sent).ids + [sep_id]
            sec2sents[section["sectionLabel"]].append(token_ids)
    return sec2sents


def _sample_positive_sentence(sec2sents, anchor_sec, anchor):
    sents = sec2sents[anchor_sec]
    pos = deepcopy(anchor)
    while pos == anchor:
        pos = random.choice(sents)
    return pos


def _sample_negative_sentence(sec2sents, anchor_sec):
    secs = list(sec2sents)
    neg_sec = deepcopy(anchor_sec)
    while neg_sec == anchor_sec:
        neg_sec = random.choice(secs)
    neg = random.choice(sec2sents[neg_sec])
    return neg


def get_wikisection_dataset(json_path, tokenizer):
    with open(json_path, mode="r") as f:
        wiki_data = jsonable_encoder(json.load(f))
        sec2sents = _group_by_sections(wiki_data, tokenizer=tokenizer)

    ds = list()
    for anchor_sec in sec2sents.keys():
        for anchor in sec2sents[anchor_sec]:
            pos = _sample_positive_sentence(sec2sents=sec2sents, anchor_sec=anchor_sec, anchor=anchor)
            neg = _sample_negative_sentence(sec2sents=sec2sents, anchor_sec=anchor_sec)

            ds.append((anchor, pos, neg))
    return ds


class WikiSectionCollator(object):
    def __init__(self, tokenizer, max_len):
        self.max_len = max_len

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def _truncate_or_pad(self, token_ids, max_len):
        if len(token_ids) <= max_len:
            token_ids = token_ids + [self.pad_id] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[: max_len - 1] + [self.sep_id]
        return token_ids
    
    def __call__(self, batch):
        a_max_len = min(self.max_len, max([len(a) for a, _, _ in batch]))
        p_max_len = min(self.max_len, max([len(p) for _, p, _ in batch]))
        n_max_len = min(self.max_len, max([len(n) for _, _, n in batch]))

        a_ls, p_ls, n_ls = list(), list(), list()
        for a, p, n in batch:
            a = self._truncate_or_pad(token_ids=a, max_len=a_max_len)
            p = self._truncate_or_pad(token_ids=p, max_len=p_max_len)
            n = self._truncate_or_pad(token_ids=n, max_len=n_max_len)

            a_ls.append(a)
            p_ls.append(p)
            n_ls.append(n)
        return torch.as_tensor(a_ls), torch.as_tensor(p_ls), torch.as_tensor(n_ls)
