"""
字符级 Transformer 语言模型训练脚本（GPU 版）
支持 Transformer 单向语言模型训练
含 PPL 计算 + 文本生成

用法:
    python language_model.py --epochs 30
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel


# ========================= 数据 =========================

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ========================= 自定义多头注意力 =========================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须是 num_heads 的整数倍"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        query, key, value: (B, T, D)
        attn_mask: (T, T) 或 None
        返回: (B, T, D)
        """
        batch_size, seq_len, _ = query.shape

        # 线性变换
        Q = self.q_proj(query)  # (B, T, D)
        K = self.k_proj(key)  # (B, T, D)
        V = self.v_proj(value)  # (B, T, D)

        # 重塑为多头形状: (B, T, H, HD) -> (B, H, T, HD)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数: (B, H, T, HD) @ (B, H, HD, T) -> (B, H, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用因果掩码（单向）
        if attn_mask is not None:
            # 扩展掩码以匹配多头维度
            mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力加权: (B, H, T, T) @ (B, H, T, HD) -> (B, H, T, HD)
        context = torch.matmul(attn_weights, V)

        # 重塑回原始形状: (B, H, T, HD) -> (B, T, D)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # 输出投影
        output = self.out_proj(context)

        return output, attn_weights


# ========================= 自定义前馈神经网络 =========================

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.expanded_dim = embed_dim * expansion_factor

        # 第一层线性变换
        self.fc1 = nn.Linear(embed_dim, self.expanded_dim)
        # 第二层线性变换
        self.fc2 = nn.Linear(self.expanded_dim, embed_dim)

        # 激活函数和Dropout
        self.activation = self.gelu
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_weights()

    def gelu(self, x):
        """GELU激活函数实现"""
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))

    def _init_weights(self):
        """初始化权重"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # 偏置初始化
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        返回: (batch_size, seq_len, embed_dim)
        """
        # 第一层: 扩展维度
        x = self.fc1(x)  # (B, T, embed_dim * 4)

        # 激活函数
        x = self.activation(x)

        # Dropout
        x = self.dropout(x)

        # 第二层: 恢复原始维度
        x = self.fc2(x)  # (B, T, embed_dim)

        return x


# ========================= Transformer 模型 =========================

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        # 使用自定义的多头自注意力
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 使用自定义的前馈神经网络
        self.ffn = FeedForwardNetwork(embed_dim, expansion_factor=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        """
        x: (B, T, D)
        causal_mask: (T, T)
        """
        attn_output, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask
        )

        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout, max_seq_len):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def causal_mask(self, sz):
        return torch.triu(
            torch.ones(sz, sz, device=self.pos_embed.device),
            diagonal=1
        ).bool()

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_seq_len

        x = self.embed(x) + self.pos_embed[:, :T, :]
        x = self.drop(x)

        mask = self.causal_mask(T)

        for layer in self.layers:
            x = layer(x, mask)

        logits = self.fc(self.drop(x))
        return logits


# ========================= 工具 =========================

def get_model(args, vocab_size, max_seq_len, device):
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
    )

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    return model


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, device, max_new_tokens=200, temperature=0.8):
    model.eval()
    tokens = [char2idx.get(c, 0) for c in prompt]
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return "".join(idx2char[i] for i in input_ids[0].tolist())


# ========================= 训练 =========================

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ========================= 主函数 =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="best_model.pt")
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")

    # 数据
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型（GPU）
    model = get_model(args, vocab_size, args.max_seq_len, device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, False)

        if va_ppl < best_ppl:
            best_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        marker = "  *" if va_ppl < best_ppl else ""
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳 Val PPL: {best_ppl:.2f}")

    prompt = "The "
    gen = generate(model, char2idx, idx2char, prompt, device)
    print("\n📝 生成示例:\n", gen)


if __name__ == "__main__":
    main()
