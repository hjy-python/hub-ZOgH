"""
第五周作业：
训练基于transformer的单向语言模型，并完成文本生成。
"""
import argparse
import glob
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────
# 按通配符查找并读取语料文件
def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


# 根据语料构建字符词表
def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


# 定义字符级数据集类
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


# ─────────────────────────── Transformer 大模型 ───────────────────────────
# 多头自注意力机制 (Multi-Head Self-Attention)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 的线性投影层和最终的输出投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 缩放因子，防止点积过大导致梯度消失
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 线性变换并拆分成多头
        Q = self.W_q(query).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        K = self.W_k(key).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        V = self.W_v(value).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.swapaxes(-2, -1)) / self.scale

        # 应用掩码 (如果有)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 归一化
        attn = torch.softmax(scores, dim=-1)

        # 加权求和并合并多头
        context = torch.matmul(attn, V)
        context = context.swapaxes(1, 2).reshape(batch_size, seq_len, -1)

        # 最终线性输出
        return self.W_o(context)


# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 通常中间隐藏层维度是 d_model 的 4 倍
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 线性变换 -> ReLU 激活 -> 线性变换
        return self.linear2(self.relu(self.linear1(x)))


# 位置编码 (Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# Transformer 解码器层 (Decoder Layer)
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)

        # 归一化
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        # 带掩码的自注意力 (Masked Self-Attention)
        x = x + self.dropout(self.self_attn(self.layer_norm(x), x, x, tgt_mask))

        # 前馈网络
        x = x + self.dropout(self.feed_forward(self.layer_norm(x)))
        return x


# 完整的 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=6, max_len=5000, dropout=0.1):
        super().__init__()
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠 N 个Decoder 层
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_seq_len = max_len

    def forward(self, tgt, tgt_mask=None):
        if tgt_mask is None:
            # 如果没有传入掩码，自动生成
            seq_len = tgt.size(1)
            if not hasattr(self, 'causal_mask') or self.causal_mask.size(0) < seq_len:
                # 缓存掩码以提高效率
                mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1).bool()
                self.register_buffer('causal_mask', mask, persistent=False)
            tgt_mask = self.causal_mask[:seq_len, :seq_len]

        # 嵌入与位置编码
        tgt_embed = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        # 解码器前向传播
        dec_output = tgt_embed
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, tgt_mask)

        # 输出层投影到词表大小
        return self.fc_out(dec_output)


# ─────────────────────────── 训练 / 评估 ───────────────────────────
# 定义训练/验证函数
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        print(f"警告：数据加载器为空，返回默认 Loss 10.0。请检查语料文件或减小 --val_ratio。")
        return 10.0, math.exp(10.0)  # 返回一个较大的默认值

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# 文本生成（推理阶段）
def generate_text(model, prompt, char2idx, idx2char, device, max_new_tokens, temperature):
    model.eval()
    input_ids = torch.tensor([[char2idx[c] for c in prompt]], dtype=torch.long).to(device)
    generated_ids = input_ids.clone()

    # 进入自回归生成循环
    with torch.no_grad():  # 推理阶段不需要计算梯度，节省显存
        for _ in range(max_new_tokens):
            # 截取最近的 max_seq_len 个 token，防止序列过长
            idx_cond = generated_ids[:, -model.max_seq_len:]
            current_seq_len = idx_cond.size(1)

            # 生成一个上三角为 True (或负无穷) 的掩码，防止看到未来信息
            causal_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=device), diagonal=1).bool()

            # 前向传播，获取最后一个时间步的输出 logits
            logits = model(idx_cond, tgt_mask=causal_mask)
            logits = logits[:, -1, :]

            # 引入温度系数进行缩放
            scaled_logits = logits / max(temperature, 1e-5)

            # 转化为概率分布并进行随机采样
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 将新生成的 token 拼接到当前序列末尾
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
    return "".join([idx2char[i.item()] for i in generated_ids.squeeze(0)])


def main():
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("--corpus", default="*.txt")  # 语料文件匹配规则
    parser.add_argument("--val_ratio", type=float, default=0.1)  # 验证集占语料库的比例
    parser.add_argument("--seq_len", type=int, default=128)  # 每个训练样本的长度
    parser.add_argument("--batch_size", type=int, default=64)  # 样本的批次大小
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数
    parser.add_argument("--lr", type=float, default=3e-4)  # 学习率
    parser.add_argument("--max_new_tokens", type=int, default=20)  # 最多生成多少个新字符
    parser.add_argument("--temperature", type=float, default=1.0)  # 生成时采样温度
    parser.add_argument("--save", default="transformer_best_model.pt")  # 最优模型保存路径
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU 或 CPU
    print(f"设备信息: {device}")

    # 读取所有语料文本
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    # 通过语料文本构建词表
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 按比例把整个语料库切分成训练集和验证集
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    # 把语料库转成训练集和验证集
    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 构建一个Transformer模型
    model = Transformer(vocab_size).to(device)

    # 统计模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 定义交叉熵和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_ppl = float("inf")

    # 打印训练日志表头
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    # 从第 1 轮训练到第 epochs 轮
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")
    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    while True:
        str = input("请输入提示词，使其续写（输入 'exit' 退出）：")
        if str == "exit":
            break
        sample = generate_text(model, prompt=str, char2idx=char2idx, idx2char=idx2char, device=device,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature)
        print(sample)


if __name__ == "__main__":
    main()
