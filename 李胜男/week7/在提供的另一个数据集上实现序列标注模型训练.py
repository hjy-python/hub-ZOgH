import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score

# ======================== 路径配置 ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

# 使用原始路径
train_file = r"E:\八斗\week7 序列标注问题\序列标注项目\data\peoples_daily\train.json"
val_file = r"E:\八斗\week7 序列标注问题\序列标注项目\data\peoples_daily\validation.json"

print(f"训练集路径: {train_file}")
print(f"验证集路径: {val_file}")
print(f"训练集文件是否存在: {os.path.exists(train_file)}")
print(f"验证集文件是否存在: {os.path.exists(val_file)}")

# ======================== 超参数 ========================
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
USE_CRF = True

# ======================== 标签定义 ========================
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


# ======================== 数据加载 ========================
def load_json_data(file_path):
    """加载JSON数据"""
    print(f"\n正在加载文件: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return []

    print(f"数据类型: {type(data)}")

    if isinstance(data, list):
        print(f"数据长度: {len(data)}")
        if len(data) > 0:
            print(f"第一条数据示例: {data[0]}")

            # 检查数据格式
            if isinstance(data[0], dict):
                keys = list(data[0].keys())
                print(f"数据键: {keys}")

                # 处理数据格式
                processed_data = []
                for item in data:
                    if "tokens" in item and "ner_tags" in item:
                        # 将 ner_tags 重命名为 labels
                        processed_item = {
                            "tokens": item["tokens"],
                            "labels": item["ner_tags"]  # 关键修改：ner_tags -> labels
                        }
                        processed_data.append(processed_item)
                    elif "tokens" in item and "labels" in item:
                        processed_data.append(item)

                print(f"处理后的样本数: {len(processed_data)}")
                if len(processed_data) > 0:
                    print(f"第一条处理后的数据: {processed_data[0]}")

                return processed_data

    print("❌ 无法识别的数据格式")
    return []


# 加载数据
train_raw = load_json_data(train_file)
val_raw = load_json_data(val_file)

print(f"\n训练集样本数: {len(train_raw)}")
print(f"验证集样本数: {len(val_raw)}")

if len(train_raw) == 0:
    print("\n⚠️ 警告: 训练集为空! 请检查数据文件格式。")
    exit(1)


# ======================== Dataset类 ========================
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"][:self.max_len - 2]
        labels = item["labels"][:self.max_len - 2]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(LABEL2ID[labels[word_idx]])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        # 返回字典，值为列表
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding["token_type_ids"],
            "labels": aligned_labels
        }


# ======================== 自定义 collate_fn ========================
def collate_fn(batch):
    """
    自定义 collate_fn 来处理字典格式的 batch
    """
    # 将每个字段的列表转换为张量
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    token_type_ids = torch.tensor([item["token_type_ids"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels
    }


# ======================== 模型定义 ========================
class BertNER(nn.Module):
    def __init__(self, bert_path, num_labels, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state
        logits = self.classifier(self.dropout(seq_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss


class BertCRFNER(nn.Module):
    def __init__(self, bert_path, num_labels, dropout=0.1):
        super().__init__()
        try:
            from torchcrf import CRF
        except ImportError:
            raise ImportError("请先安装 pytorch-crf: pip install pytorch-crf")

        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")
        return emissions, loss

    def decode(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)


# ======================== 初始化 ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

if USE_CRF:
    model = BertCRFNER("bert-base-chinese", len(LABELS))
else:
    model = BertNER("bert-base-chinese", len(LABELS))

model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ======================== 数据加载器 ========================
train_dataset = NERDataset(train_raw, tokenizer, MAX_LEN)
val_dataset = NERDataset(val_raw, tokenizer, MAX_LEN)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn  # 添加自定义 collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn  # 添加自定义 collate_fn
)

print(f"训练批次: {len(train_loader)}")
print(f"验证批次: {len(val_loader)}")


# ======================== 训练 ========================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="训练中")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


# ======================== 评估 ========================
def evaluate(model, dataloader, device):
    model.eval()
    true_tags, pred_tags = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].cpu()

            if USE_CRF:
                preds = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                preds = torch.argmax(logits, dim=-1).cpu()

            for i in range(len(labels)):
                true_seq, pred_seq = [], []
                for j in range(len(labels[i])):
                    if labels[i][j] == -100:
                        continue
                    true_label = ID2LABEL[labels[i][j].item()]
                    pred_label = ID2LABEL[preds[i][j]]
                    true_seq.append(true_label)
                    pred_seq.append(pred_label)
                true_tags.append(true_seq)
                pred_tags.append(pred_seq)

    return true_tags, pred_tags


# ======================== 主程序 ========================
if __name__ == "__main__":
    print("开始训练...")

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 50}")

        avg_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"平均损失: {avg_loss:.4f}")

        true_tags, pred_tags = evaluate(model, val_loader, device)

        print("\n分类报告:")
        print(classification_report(true_tags, pred_tags))
        print(f"F1分数: {f1_score(true_tags, pred_tags):.4f}")

    # 保存模型
    model_path = os.path.join(SAVE_DIR, "ner_model.bin")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'labels': LABELS,
        'label2id': LABEL2ID,
        'id2label': ID2LABEL,
    }, model_path)

    print(f"\n✅ 训练完成！模型已保存到: {model_path}")
