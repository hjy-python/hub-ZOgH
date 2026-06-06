"""
PEOPLE'S DAILY NER 数据集类：CoNLL 格式 → BIO 标签 + BERT 子词对齐

数据格式（与 cluener 的关键区别）：
  - cluener:   {"text": "...", "label": {"name": {"张三": [[0,1]]}}}  ← span 格式
  - peoples_daily: {"tokens": ["张", "三"], "ner_tags": ["B-PER", "I-PER"]}  ← CoNLL BIO 格式

教学重点：
  1. CoNLL 格式已经是逐 token BIO 标签，无需 span_to_bio 转换
  2. 需要将字符级标签与 BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，非首子词标记为 -100
  3. 标签体系：O / B-PER / I-PER / B-ORG / I-ORG / B-LOC / I-LOC（7个标签）

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

# 标签体系（与 label_names.json 一致）
ENTITY_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    labels = list(ENTITY_TAGS)
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 的 PyTorch Dataset。

    数据格式：
      {"tokens": ["北", "京", "市"], "ner_tags": ["B-LOC", "I-LOC", "I-LOC"]}

    处理流程：
      tokens (字符列表) → BertTokenizer (is_split_into_words=True)
        → 用 word_ids() 对齐子词标签（非首子词设为 -100）
        → 返回 input_ids / attention_mask / token_type_ids / labels
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row["ner_tags"]

        # 将 BIO 标签字符串转为 id
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # 截断到 max_length - 2（留 [CLS] 和 [SEP]）
        max_tokens = self.max_length - 2
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            char_labels = char_labels[:max_tokens]

        # 用 is_split_into_words=True 让 tokenizer 按字符拆分
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 子词对齐
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)  # [CLS], [SEP], [PAD]
            elif wid != prev_word_id:
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)  # 同一字符的后续子词

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
