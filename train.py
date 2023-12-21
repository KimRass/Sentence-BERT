from torch.utils.data import DataLoader

from transformers import BertTokenizerFast
from models.bert import BERT
from models.sbert import (
    SentenceBERTForCls, SentenceBERTForReg, SentenceBERTForContrastiveLearning,
)
from data.sts_benchmark import STSbenchmarkDataset, STSbenchmarkCollator
from data.wikisection import get_wikisection_dataset, WikiSectionCollator
from data.nli import get_nli_dataset, NLICollator
from loss import CLS_LOSS_FN, REG_LOSS_FN, TRIP_LOSS_FN


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    BATCH_SIZE = 8
    VOCAB_SIZE = 30_522
    MAX_LEN = 128
    PAD_ID = 0
    bert = BERT(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, pad_id=PAD_ID)
    
    sbert_cls = SentenceBERTForCls(embedder=bert)

    txt_path = "/Users/jongbeomkim/Documents/datasets/multinli_1.0/multinli_1.0_train.txt"
    nli_ds = get_nli_dataset(txt_path=txt_path, tokenizer=tokenizer)
    nli_collator = NLICollator(tokenizer=tokenizer, max_len=MAX_LEN)
    nli_dl = DataLoader(
        dataset=nli_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=nli_collator,
    )
    for batch, (p, h, label) in enumerate(nli_dl, start=1):
        logit = sbert_cls(p=p, h=h)
        loss = CLS_LOSS_FN(logit, label)
        print(loss)

    sbert_reg = SentenceBERTForReg(embedder=bert)

    stsb_ds = STSbenchmarkDataset(
        csv_path="/Users/jongbeomkim/Documents/datasets/stsbenchmark/sts-train.csv",
        tokenizer=tokenizer,
    )
    stsb_collator = STSbenchmarkCollator(tokenizer=tokenizer)
    stsb_dl = DataLoader(
        stsb_ds,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        collate_fn=stsb_collator
    )
    for batch, (score, sent1, sent2) in enumerate(stsb_dl, start=1):
        logit = sbert_reg(sent1=sent1, sent2=sent2)
        loss = REG_LOSS_FN(logit, score)
        print(loss)

    sbert_trip = SentenceBERTForContrastiveLearning(embedder=bert)
    json_path = "/Users/jongbeomkim/Documents/datasets/wikisection_dataset_json/wikisection_en_city_train.json"

    wiki_ds = get_wikisection_dataset(json_path=json_path, tokenizer=tokenizer)
    wiki_collator = WikiSectionCollator(tokenizer=tokenizer, max_len=MAX_LEN)
    wiki_dl = DataLoader(
        dataset=wiki_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=wiki_collator,
    )
    for batch, (a, p, n) in enumerate(wiki_dl, start=1):
        logit = sbert_trip(a=a, p=p, n=n)
        loss = TRIP_LOSS_FN(*logit)
        print(loss)
