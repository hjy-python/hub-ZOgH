import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoTokenizer, AutoModelForCausalLM, pipeline
)

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {DEVICE}")

# ========================= 1. 超参数 ===========================
BATCH_SIZE    = 32
MAX_LEN       = 128
NUM_EPOCHS_TRANSFORMER = 20
NUM_EPOCHS_BERT        = 5
LR_TRANSFORMER = 1e-3
LR_BERT        = 2e-5
NUM_CLASSES    = 4

# ========================= 2. 加载数据 =========================
print("\n" + "=" * 55)
print("📦 1. 加载数据 (20 Newsgroups)")
print("=" * 55)

categories = [
    'alt.atheism', 'comp.graphics',
  'sci.med','soc.religion.christian'
]
news_train = fetch_20newsgroups(subset='train', categories=categories)
news_test  = fetch_20newsgroups(subset='test',  categories=categories)

print(f"训练集: {len(news_train.data)} 条 | 测试集: {len(news_test.data)} 条")
print(f"类别: {news_train.target_names}")

# ========================= 3. 构建词汇表 =======================
print("\n" + "=" * 55)
print("🔤 2. 构建词汇表")
print("=" * 55)

from collections import Counter


def build_vocab(texts, max_vocab=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


vocab = build_vocab(news_train.data, max_vocab=10000)
print(f"词汇表大小: {len(vocab)}")


def tokenize(text, vocab, max_len=MAX_LEN):
    tokens = [vocab.get(w, 1) for w in text.lower().split()[:max_len]]
    mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor(tokens), torch.tensor(mask)


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, mask = tokenize(self.texts[idx], self.vocab, self.max_len)
        return ids, mask, torch.tensor(self.labels[idx])


train_dataset = TextDataset(news_train.data, news_train.target, vocab)
test_dataset = TextDataset(news_test.data, news_test.target, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 4. 自实现 Transformer =====================
print("\n" + "=" * 55)
print("🏗️  3. 自实现 Transformer")
print("=" * 55)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape  # batch, seq_len, d_model
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # MHA + 残差 + LayerNorm
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        # FFN + 残差 + LayerNorm
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=3, n_classes=NUM_CLASSES, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.pos_enc(self.embedding(x)))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        # 取 [CLS] 位置（第一个 token）的输出做分类
        x = x[:, 0, :]
        return self.classifier(x)


# ====================== 5. 训练与评估函数 =======================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, criterion, desc=""):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ids.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += ids.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, desc=""):
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.time()
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        logits = model(ids, mask)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    infer_time = time.time() - t0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1, infer_time


def train_model(model, train_loader, test_loader, epochs, lr, name=""):
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    t0 = time.time()

    print(f"\n{'=' * 50}")
    print(f"训练: {name} | 参数量: {count_params(model):,}")
    print(f"{'=' * 50}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_acc, test_f1, _ = evaluate(model, test_loader)
        scheduler.step()
        best_acc = max(best_acc, test_acc)
        if epoch % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch:2d}/{epochs} | "
                  f"Loss={train_loss:.4f} | "
                  f"Train Acc={train_acc:.4f} | "
                  f"Test Acc={test_acc:.4f} | "
                  f"Test F1={test_f1:.4f}")

    train_time = time.time() - t0
    test_acc, test_f1, infer_time = evaluate(model, test_loader)

    result = {
        'name': name,
        'params': count_params(model),
        'train_time': round(train_time, 2),
        'infer_time': round(infer_time, 4),
        'accuracy': round(test_acc, 4),
        'f1': round(test_f1, 4),
        'best_acc': round(best_acc, 4),
    }
    print(f"\n  ✅ {name} 最终: Acc={test_acc:.4f} | F1={test_f1:.4f} | "
          f"训练耗时={train_time:.1f}s\n")
    return model, result


# ====================== 6. 训练 Transformer =====================

print("\n" + "=" * 55)
print("🚀 4. 训练自实现 Transformer")
print("=" * 55)

transformer_model = TransformerClassifier(
    vocab_size=len(vocab),
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_layers=3,
    n_classes=NUM_CLASSES,
    max_len=MAX_LEN,
    dropout=0.1,
)

transformer_model, result_transformer = train_model(
    transformer_model, train_loader, test_loader,
    epochs=NUM_EPOCHS_TRANSFORMER, lr=LR_TRANSFORMER,
    name="自实现 Transformer"
)

# ====================== 7. 微调 BERT ===========================
print("\n" + "=" * 55)
print("🤖 5. 微调预训练 BERT")
print("=" * 55)

try:
    bert_name = 'bert - base - uncased'
    tokenizer_bert = BertTokenizer.from_pretrained(bert_name)

    def tokenize_bert(texts, max_len=MAX_LEN):
        enc = tokenizer_bert(
            list(texts), padding=True, truncation=True,
            max_length=max_len, return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask']

    train_ids, train_mask_bert = tokenize_bert(news_train.data)
    test_ids, test_mask_bert = tokenize_bert(news_test.data)

    train_labels_bert = torch.tensor(news_train.target)
    test_labels_bert = torch.tensor(news_test.target)

    class BERTDataset(Dataset):
        def __init__(self, ids, mask, labels):
            self.ids = ids
            self.mask = mask
            self.labels = labels

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            return self.ids[idx], self.mask[idx], self.labels[idx]

    bert_train_set = BERTDataset(train_ids, train_mask_bert, train_labels_bert)
    bert_test_set = BERTDataset(test_ids, test_mask_bert, test_labels_bert)
    bert_train_loader = DataLoader(bert_train_set, batch_size=BATCH_SIZE, shuffle=True)
    bert_test_loader = DataLoader(bert_test_set, batch_size=BATCH_SIZE, shuffle=False)

    bert_model = BertForSequenceClassification.from_pretrained(
        bert_name, num_labels=NUM_CLASSES
    )

    bert_model, result_bert = train_model(
        bert_model, bert_train_loader, bert_test_loader,
        epochs=NUM_EPOCHS_BERT, lr=LR_BERT,
        name="BERT (预训练微调)"
    )

except ImportError:
    print("⚠️ 未安装 transformers 库，跳过 BERT 部分")
    print("   安装命令: pip install transformers")
    result_bert = None

# ====================== 8. 基于 GPT - Neo 的零样本分类 =====================
print("\n" + "=" * 55)
print("🧠 6. 基于 GPT - Neo 的零样本分类")
print("=" * 55)

# try:
#     tokenizer_gptneo = AutoTokenizer.from_pretrained("EleutherAI/gpt - neo - 1.3B")
#     model_gptneo = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt - neo - 1.3B")

#     def zero_shot_classify(texts, labels):
#         classifier = pipeline('zero - shot - classification', model=model_gptneo, tokenizer=tokenizer_gptneo)
#         all_preds = []
#         for text in texts:
#             result = classifier(text, news_train.target_names)
#             all_preds.append(news_train.target_names.index(result['labels'][0]))
#         acc = accuracy_score(labels, all_preds)
#         f1 = f1_score(labels, all_preds, average='macro')
#         return acc, f1


#     t0 = time.time()
try:
    tokenizer_gptneo = AutoTokenizer.from_pretrained("EleutherAI/gpt - neo - 1.3B")
    model_gptneo = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt - neo - 1.3B")

    def zero_shot_classify(texts, labels):
        classifier = pipeline('zero - shot - classification', model=model_gptneo, tokenizer=tokenizer_gptneo)
        all_preds = []
        for text in texts:
            result = classifier(text, news_train.target_names)
            all_preds.append(news_train.target_names.index(result['labels'][0]))
        acc = accuracy_score(labels, all_preds)
        f1 = f1_score(labels, all_preds, average='macro')
        return acc, f1


    t0 = time.time()
except Exception as e:
    print(f"发生异常: {e}")
