import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = hidden // n_head
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out)

class DecoderLayer(nn.Module):
    def __init__(self, hidden, n_head, ff):
        super().__init__()
        self.attn = CausalSelfAttention(hidden, n_head)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ffn(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, hidden=256, n_layer=4, n_head=4, ff=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 128, hidden)
        )
        self.layers = nn.ModuleList([
            DecoderLayer(hidden, n_head, ff)
            for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.embedding(tokens) + self.pos_embedding[:, :T]

        mask = causal_mask(T).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        logits = self.lm_head(x)
        return logits

def train_step(model, tokens, optimizer):
    model.train()
    logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def generate(model, start_tokens, max_new_tokens=50):
    model.eval()
    tokens = start_tokens.clone()

    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens

vocab_size = 1000
model = TransformerLM(vocab_size)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 假数据
tokens = torch.randint(0, vocab_size, (2, 32))

loss = train_step(model, tokens, optimizer)
print("loss:", loss)

# 生成
start = torch.randint(0, vocab_size, (1, 5))
out = generate(model, start, max_new_tokens=20)
print("generated shape:", out.shape)
