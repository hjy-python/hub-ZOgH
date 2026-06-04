import random
from pathlib import Path

from tqdm import tqdm

from ..data import dump_json, ensure_outputs, load_labels, load_split
from ..metrics import evaluate_tag_sequences


def _deps():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchcrf import CRF
    from transformers import AutoModel, AutoTokenizer
    return torch, nn, DataLoader, Dataset, CRF, AutoModel, AutoTokenizer


def run_bert_crf(config):
    torch, nn, DataLoader, Dataset, CRF, AutoModel, AutoTokenizer = _deps()
    ensure_outputs(config)
    labels = load_labels(config)
    label2id = {x: i for i, x in enumerate(labels)}
    id2label = {i: x for x, i in label2id.items()}

    random.seed(config.bert_crf.seed)
    torch.manual_seed(config.bert_crf.seed)
    device = torch.device("cuda" if config.bert_crf.device == "auto" and torch.cuda.is_available() else "cpu")

    local_only = Path(str(config.bert_crf.model_name)).exists()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_crf.model_name, use_fast=True, local_files_only=local_only)
    train = load_split(config, "train")
    valid = load_split(config, "validation")
    test = load_split(config, "test")

    class NerDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            enc = tokenizer(
                sample["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=config.bert_crf.max_length,
            )
            word_ids = enc.word_ids()
            tags, keep, prev = [], [], None
            for wid in word_ids:
                if wid is None:
                    tags.append(label2id["O"])
                    keep.append(0)
                elif wid != prev:
                    tags.append(label2id[sample["ner_tags"][wid]])
                    keep.append(1)
                else:
                    tags.append(label2id["O"])
                    keep.append(0)
                prev = wid
            item = {k: torch.tensor(v) for k, v in enc.items()}
            item["labels"] = torch.tensor(tags)
            item["keep"] = torch.tensor(keep)
            item["idx"] = torch.tensor(idx)
            return item

    class BertCrf(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(config.bert_crf.model_name, local_files_only=local_only)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.bert.config.hidden_size, len(labels))
            self.crf = CRF(len(labels), batch_first=True)

        def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
            h = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state
            emissions = self.classifier(self.dropout(h))
            mask = attention_mask.bool()
            pred = self.crf.decode(emissions, mask=mask)
            loss = None if labels is None else -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss, pred

    def loader(samples, shuffle=False):
        return DataLoader(NerDataset(samples), batch_size=config.bert_crf.batch_size, shuffle=shuffle)

    model = BertCrf().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.bert_crf.lr, weight_decay=config.bert_crf.weight_decay)

    def predict(samples):
        model.eval()
        all_tags = []
        with torch.no_grad():
            for batch in loader(samples):
                idx = batch.pop("idx").tolist()
                keep = batch.pop("keep").tolist()
                batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                _, decoded = model(**batch)
                for sample_idx, row, keep_row in zip(idx, decoded, keep):
                    tags = [id2label[row[j]] for j, k in enumerate(keep_row[:len(row)]) if k]
                    all_tags.append((sample_idx, tags))
        return [tags for _, tags in sorted(all_tags)]

    best_f1 = -1.0
    config.bert_crf.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.bert_crf.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader(train, shuffle=True), desc=f"epoch {epoch + 1}"):
            batch.pop("idx")
            batch.pop("keep")
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(**batch)
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()

        valid_pred = predict(valid)
        report = evaluate_tag_sequences(valid, valid_pred, "bert_crf_valid", order_definition=config.invalid_bio_order_definition)
        f1 = report["overall"]["f1"]
        print(f"epoch={epoch + 1} loss={total_loss:.4f} valid_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), config.bert_crf.output_dir / "best.pt")

    model.load_state_dict(torch.load(config.bert_crf.output_dir / "best.pt", map_location=device))
    test_pred = predict(test)
    report = evaluate_tag_sequences(test, test_pred, "bert_crf", order_definition=config.invalid_bio_order_definition)
    dump_json({k: v for k, v in report.items() if k != "records"}, config.output.reports / "bert_crf.json")
    dump_json(report["records"], config.output.predictions / "bert_crf_test.json")
    print("bert_crf f1", report["overall"]["f1"])
