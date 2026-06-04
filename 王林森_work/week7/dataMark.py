import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_metric
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split

# ===================== 1. 超参数 & 配置 =====================
CONFIG = {
    "model_name": "bert-base-chinese",  # 中文用bert-base-chinese，英文用bert-base-uncased
    "data_path": "your_dataset.txt",    # 你的新数据集路径
    "max_len": 128,                     # 句子最大长度
    "batch_size": 32,
    "lr": 2e-5,
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 序列标注专用评价指标
metric = load_metric("seqeval")

# ===================== 2. 数据集读取与解析 =====================
def load_sequence_data(file_path):
    """
    加载标准序列标注数据
    数据格式：每行一个字/词，空行分隔句子，最后一列为标签
    """
    sentences = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        sent = []
        label = []
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    labels.append(label)
                    sent = []
                    label = []
            else:
                parts = line.split()
                token = parts[0]
                tag = parts[-1]
                sent.append(token)
                label.append(tag)
    return sentences, labels

# 加载数据集
sentences, labels = load_sequence_data(CONFIG["data_path"])

# 构建标签映射
all_labels = [tag for sent_labels in labels for tag in sent_labels]
label_list = sorted(list(set(all_labels)))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
num_labels = len(label_list)

print(f"标签总数: {num_labels}")
print(f"标签列表: {label_list}")

# 划分训练集/验证集
train_sents, val_sents, train_tags, val_tags = train_test_split(
    sentences, labels, test_size=0.1, random_state=42
)

# ===================== 3. 构建数据集类 =====================
class NERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, max_len, label2id):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        tags = self.tags[idx]

        # 编码（处理BERT子词拆分）
        encoding = self.tokenizer(
            tokens,
            truncation=True,
            max_length=self.max_len,
            is_split_into_words=True,
            padding="max_length",
            return_tensors="pt"
        )

        # 标签对齐（子词只保留第一个token的标签，其余设为-100）
        word_ids = encoding.word_ids()
        label_ids = []
        prev_idx = None
        for idx in word_ids:
            if idx is None:
                label_ids.append(-100)
            elif idx != prev_idx:
                label_ids.append(self.label2id[tags[idx]])
                prev_idx = idx
            else:
                label_ids.append(-100)

        # 整理输出
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return item

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

# 构建数据集
train_dataset = NERDataset(train_sents, train_tags, tokenizer, CONFIG["max_len"], label2id)
val_dataset = NERDataset(val_sents, val_tags, tokenizer, CONFIG["max_len"], label2id)

# ===================== 4. 模型 & 训练配置 =====================
model = AutoModelForTokenClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(CONFIG["device"])

# 训练参数
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    learning_rate=CONFIG["lr"],
    num_train_epochs=CONFIG["epochs"],
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 评价函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # 移除-100标签
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print("\n" + classification_report(true_labels, true_predictions, digits=4))
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ===================== 5. 开始训练 =====================
print("\n========== 开始训练新数据集 ==========")
trainer.train()

# 保存最优模型
trainer.save_model("./best_ner_model")
tokenizer.save_pretrained("./best_ner_model")
print("\n训练完成！最优模型已保存至 ./best_ner_model")

# ===================== 6. 推理测试 =====================
def predict(text, model, tokenizer, id2label, max_len=128):
    tokens = list(text)
    inputs = tokenizer(
        tokens,
        truncation=True,
        max_length=max_len,
        is_split_into_words=True,
        return_tensors="pt"
    ).to(CONFIG["device"])

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0]

    # 对齐结果
    word_ids = inputs.word_ids()
    result = []
    prev_idx = None
    for token, pred, idx in zip(tokens, predictions[1:-1], word_ids[1:-1]):
        if idx != prev_idx:
            result.append((token, id2label[pred.item()]))
            prev_idx = idx
    return result

# 测试示例
if __name__ == "__main__":
    test_text = "时间为尊，空间为王"
    pred_result = predict(test_text, model, tokenizer, id2label)
    print("\n预测结果：")
    for token, label in pred_result:
        print(f"{token}\t{label}")
