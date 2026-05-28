# -*- coding: utf-8 -*-
"""  
@Project : lycoris
@IDE : PyCharm
@File : 作业
@Author : lycoris
@Time : 2026/5/28 20:15  
@脚本说明 : 

"""
"""
文本分类方法对比实验
数据集：20 newsgroups 选取 3 个类别（rec.sport.baseball, sci.space, talk.politics.guns）
依赖安装：
pip install scikit-learn torch transformers datasets accelerate peft

关键结论
传统方法（TF‑IDF + LR）：最快，可解释，适合简单任务或基线。

TextCNN：中等性能，需要较多数据且训练较慢，但比传统方法好。

BERT 微调：效果最好，但时间最长，需要 GPU 支持。

LoRA：效果接近全参数微调，训练速度和显存占用都有改善。

Prompt‑tuning（简化版）：训练参数极少（只训练分类头），效果尚可，适合小样本场景。
"""

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.datasets import fetch_20newsgroups

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    PreTrainedTokenizerBase
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset

# ------------------------- 1. 加载与预处理数据 -------------------------
categories = ['rec.sport.baseball', 'sci.space', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
texts, labels = newsgroups.data, newsgroups.target

# 划分训练集（70%）、验证集（15%）、测试集（15%）
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"训练集大小: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")


# ------------------------- 2. 传统方法：TF-IDF + Logistic Regression -------------------------
def train_traditional():
    print("\n========== 传统方法 (TF-IDF + LR) ==========")
    start = time.time()

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    elapsed = time.time() - start

    print(f"测试准确率: {acc:.4f}, 宏平均F1: {f1:.4f}, 训练+推理时间: {elapsed:.2f}s")
    return acc, f1, elapsed


# ------------------------- 3. 深度学习：TextCNN -------------------------
# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].lower().split()[:self.max_len]
        ids = [self.vocab.get(t, 1) for t in tokens]  # 1: UNK
        ids = ids + [0] * (self.max_len - len(ids))  # 0: PAD
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=[3, 4, 5], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch, 1, seq_len, embed_dim]
        conv_outs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x)).squeeze(3)  # [batch, num_filters, seq_len-fs+1]
            pool_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outs.append(pool_out)
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)


def train_textcnn():
    print("\n========== TextCNN ==========")
    # 构建词表（简化，仅取高频词）
    from collections import Counter
    word_freq = Counter()
    for text in X_train:
        word_freq.update(text.lower().split())
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_freq.most_common(20000))}  # idx 0:PAD, 1:UNK
    vocab_size = len(vocab) + 2

    train_dataset = TextDataset(X_train, y_train, vocab)
    val_dataset = TextDataset(X_val, y_val, vocab)
    test_dataset = TextDataset(X_test, y_test, vocab)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(vocab_size, embed_dim=100, num_classes=len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start = time.time()
    epochs = 5
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        val_acc = correct / total
        print(f"Epoch {epoch + 1}: train_loss={total_loss / len(train_loader):.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_textcnn.pt")

    # 测试
    model.load_state_dict(torch.load("best_textcnn.pt"))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            output = model(x)
            pred = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
    acc = accuracy_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds, average='macro')
    elapsed = time.time() - start
    print(f"测试准确率: {acc:.4f}, 宏平均F1: {f1:.4f}, 总时间: {elapsed:.2f}s")
    return acc, f1, elapsed


# ------------------------- 4. 预训练模型微调 (DistilBERT) -------------------------
def train_bert_finetune():
    print("\n========== BERT 微调 (DistilBERT) ==========")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 转换为 HuggingFace Dataset
    train_hf = HFDataset.from_dict({"text": X_train, "label": y_train})
    val_hf = HFDataset.from_dict({"text": X_val, "label": y_val})
    test_hf = HFDataset.from_dict({"text": X_test, "label": y_test})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=256)

    train_hf = train_hf.map(tokenize_fn, batched=True)
    val_hf = val_hf.map(tokenize_fn, batched=True)
    test_hf = test_hf.map(tokenize_fn, batched=True)

    # 设置数据格式
    train_hf = train_hf.rename_column("label", "labels")
    val_hf = val_hf.rename_column("label", "labels")
    test_hf = test_hf.rename_column("label", "labels")
    train_hf.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_hf.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_hf.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 训练参数（少量epochs快速对比）
    training_args = TrainingArguments(
        output_dir="./bert_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return {"accuracy": acc, "f1_macro": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    start = time.time()
    trainer.train()
    eval_result = trainer.evaluate(test_hf)
    acc = eval_result["eval_accuracy"]
    f1 = eval_result["eval_f1_macro"]
    elapsed = time.time() - start
    print(f"测试准确率: {acc:.4f}, 宏平均F1: {f1:.4f}, 总时间: {elapsed:.2f}s")
    return acc, f1, elapsed


# ------------------------- 5. 参数高效微调 (LoRA) -------------------------
def train_lora():
    print("\n========== LoRA 微调 (基于 DistilBERT) ==========")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_hf = HFDataset.from_dict({"text": X_train, "label": y_train})
    val_hf = HFDataset.from_dict({"text": X_val, "label": y_val})
    test_hf = HFDataset.from_dict({"text": X_test, "label": y_test})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=256)

    train_hf = train_hf.map(tokenize_fn, batched=True)
    val_hf = val_hf.map(tokenize_fn, batched=True)
    test_hf = test_hf.map(tokenize_fn, batched=True)

    train_hf = train_hf.rename_column("label", "labels")
    val_hf = val_hf.rename_column("label", "labels")
    test_hf = test_hf.rename_column("label", "labels")
    train_hf.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_hf.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_hf.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # DistilBERT 的注意力线性层
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 显示可训练参数量

    training_args = TrainingArguments(
        output_dir="./lora_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs_lora',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return {"accuracy": acc, "f1_macro": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    start = time.time()
    trainer.train()
    eval_result = trainer.evaluate(test_hf)
    acc = eval_result["eval_accuracy"]
    f1 = eval_result["eval_f1_macro"]
    elapsed = time.time() - start
    print(f"测试准确率: {acc:.4f}, 宏平均F1: {f1:.4f}, 总时间: {elapsed:.2f}s")
    return acc, f1, elapsed


# ------------------------- 6. 提示学习 (Prompt-tuning) 简化示例 -------------------------
# 注意：完全实现软提示训练较复杂，这里使用一个简单的基于模板的硬提示 + 冻结主模型，
# 仅训练一个分类头，体现“少参数微调”的思想。
def train_prompt_tuning():
    print("\n========== 提示学习 (冻结模型 + 模板预测) ==========")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 设计模板："This text is about [MASK]." 然后映射 mask token 到类别词
    # 类别词: baseball -> "baseball", space -> "space", politics -> "politics"
    label_words = ["baseball", "space", "politics"]
    label2word = {0: "baseball", 1: "space", 2: "politics"}

    # 对于每条文本，构造 prompt: text + " This text is about [MASK]."
    def build_prompt(text):
        return f"{text} This text is about [MASK]."

    # 由于 DistilBERT 没有单独的 mask token，这里用 [MASK] 字符串代替实际做法。
    # 更严谨的做法是使用 RoBERTa 等 MLM 模型，此处为展示框架，我们用微调分类头的方式。
    # 实际 Prompt-tuning 需要训练连续向量，这里为了简单对比，改为：使用预训练模型提取特征，然后训练一个小型分类器，冻结主体。
    # 这近似于 “特征提取 + 分类头”，体现少样本微调效率。

    train_texts = [build_prompt(t) for t in X_train]
    val_texts = [build_prompt(t) for t in X_val]
    test_texts = [build_prompt(t) for t in X_test]

    # 使用 tokenizer 并只保存 input_ids, attention_mask
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=300, return_tensors='pt')
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=300, return_tensors='pt')
    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=300, return_tensors='pt')

    # 加载模型并冻结主体
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # 冻结所有参数
    for param in base_model.parameters():
        param.requires_grad = False
    # 只重新初始化分类头（实际 prompt tuning 会训练软提示，这里简化）
    base_model.classifier = nn.Linear(base_model.config.dim, 3)
    # 只训练分类头
    for param in base_model.classifier.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    train_dataset = torch.utils.data.TensorDataset(train_enc['input_ids'], train_enc['attention_mask'],
                                                   torch.tensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], torch.tensor(y_val))
    test_dataset = torch.utils.data.TensorDataset(test_enc['input_ids'], test_enc['attention_mask'],
                                                  torch.tensor(y_test))

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    optimizer = optim.Adam(base_model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    epochs = 5
    best_val_acc = 0
    for epoch in range(epochs):
        base_model.train()
        total_loss = 0
        for input_ids, att_mask, labels in train_loader:
            input_ids, att_mask, labels = input_ids.to(device), att_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = base_model(input_ids=input_ids, attention_mask=att_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        base_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, att_mask, labels in val_loader:
                input_ids, att_mask, labels = input_ids.to(device), att_mask.to(device), labels.to(device)
                outputs = base_model(input_ids=input_ids, attention_mask=att_mask)
                pred = outputs.logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += len(labels)
        val_acc = correct / total
        print(f"Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # 测试
    base_model.eval()
    all_preds = []
    with torch.no_grad():
        for input_ids, att_mask, labels in test_loader:
            input_ids, att_mask = input_ids.to(device), att_mask.to(device)
            outputs = base_model(input_ids=input_ids, attention_mask=att_mask)
            pred = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
    acc = accuracy_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds, average='macro')
    elapsed = time.time() - start
    print(f"测试准确率: {acc:.4f}, 宏平均F1: {f1:.4f}, 总时间: {elapsed:.2f}s")
    return acc, f1, elapsed


# ------------------------- 7. 运行对比 -------------------------
if __name__ == "__main__":
    results = {}

    # 传统方法
    results["TF-IDF + LR"] = train_traditional()

    # TextCNN
    results["TextCNN"] = train_textcnn()

    # BERT 微调
    results["BERT Fine-tune"] = train_bert_finetune()

    # LoRA
    results["LoRA"] = train_lora()

    # Prompt-tuning (简化)
    results["Prompt-tuning (简版)"] = train_prompt_tuning()

    # 打印对比表格
    print("\n" + "=" * 70)
    print("方法对比汇总")
    print("=" * 70)
    print(f"{'方法':<25} {'准确率':<10} {'宏平均F1':<10} {'训练+测试时间(s)':<15}")
    for name, (acc, f1, tm) in results.items():
        print(f"{name:<25} {acc:<10.4f} {f1:<10.4f} {tm:<15.2f}")