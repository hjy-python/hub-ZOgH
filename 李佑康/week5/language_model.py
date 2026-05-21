import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        def transform(x, linear):
            batch_size = x.size(0)
            x = linear(x)
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)
        q = transform(x, self.linear_q)
        k = transform(x, self.linear_k)
        v = transform(x, self.linear_v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(q.size(0), -1, self.num_heads * self.d_k)
        return self.linear_out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_len=1024, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

        self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        ).unsqueeze(0).unsqueeze(0)

        x = self.token_embedding(x)  # (batch, seq) → (batch, seq, d_model)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq)
        x = self.dropout(x + self.position_embedding(positions))   # 加上可学习的位置嵌入

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.norm(x)
        return self.lm_head(x)


@torch.no_grad()
def generate(model, start_ids, max_new_tokens=100, temperature=1.0, top_k=None):
    model.eval()
    ids = start_ids.clone()

    for _ in range(max_new_tokens):
        context = ids[:, -model.max_len:] if ids.size(1) > model.max_len else ids

        logits = model(context)
        logits = logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=1)

    return ids

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])


def create_batches(data, batch_size, block_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y # x 是输入序列，y 是对应的目标序列（即输入序列右移一位）


def train(model, data, epochs=10, batch_size=64, block_size=128,
          lr=3e-4, device=None, eval_interval=500):
    if device is None:
        device = torch.device('cpu')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(data) // (batch_size * block_size)

    total_loss = 0
    start_time = time.time()

    for step in range(1, total_steps + 1):
        x, y = create_batches(data, batch_size, block_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), # (B, T, V) -> (B*T, V)
            y.reshape(-1) #(B, T) -> (B*T)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % eval_interval == 0 or step == total_steps:
            n = eval_interval if step % eval_interval == 0 else step % eval_interval or 1
            avg_loss = total_loss / n
            elapsed = time.time() - start_time
            ppl = math.exp(avg_loss)
            print(f"  step {step:6d}/{total_steps} | loss: {avg_loss:.4f} | "
                  f"ppl: {ppl:6.2f} | 耗时: {elapsed:.0f}s")
            total_loss = 0
            start_time = time.time()



def get_device():
    """选择最优设备（Mac MPS / CUDA / CPU）"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    device = get_device()
    print(f"设备: {device}\n")

    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_dir, '三国演义节选.txt')

    print(f"加载数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"  字符数: {len(text)}")

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"  词汇量: {tokenizer.vocab_size}")
    print(f"  tokens: {len(data)}\n")

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=512,
        max_len=128,
        dropout=0.3,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params / 1e6:.1f}M\n")

    # ---- 训练 ----
    print("开始训练...")
    train(model, data, epochs=100, batch_size=32, block_size=64,
          lr=1e-3, device=device, eval_interval=100)
    print("训练完成！\n")

    # ---- 保存模型 ----
    model_path = os.path.join(data_dir, 'gpt_lm.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos,
    }, model_path)
    print(f"模型已保存到 {model_path}\n")

    # ---- 文本生成示例 ----
    print("=" * 60)
    print("文本生成示例")
    print("=" * 60)

    prompts = [
        "话说天下大势"
    ]

    for prompt in prompts:
        try:
            start_ids = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
        except KeyError:
            print(f"\n--- Prompt: \"{prompt}\" --- (字符不在词表中，跳过)")
            continue
        output = generate(model, start_ids, max_new_tokens=200, temperature=0.8, top_k=40)
        generated = tokenizer.decode(output[0].tolist())
        print(f"\n--- Prompt: \"{prompt}\" ---")
        print(generated[:400])
        print("...")


if __name__ == '__main__':
    main()
