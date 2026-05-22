import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 超参数设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
seq_len = 10  # 序列长度
d_model = 128  # 模型维度
nhead = 8  # 注意力头数
num_layers = 2  # 解码器层数
lr = 1e-3  # 学习率
epochs = 50  # 训练轮数

# 简易语料（可替换成你的文本）
corpus = [
    "我 爱 深度 学习",
    "Transformer 是 强大 的 模型",
    "单向 语言 模型 用于 文本 生成",
    "今天 我们 训练 一个 生成 模型",
    "机器 学习 改变 世界"
]

# ===================== 2. 构建词典 =====================
# 分词（空格分隔）
words = []
for sentence in corpus:
    words.extend(sentence.split())
words = list(set(words))  # 去重

# 特殊标记
words = ["<pad>", "<unk>", "<sos>", "<eos>"] + words
vocab_size = len(words)
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for i, w in enumerate(words)}

# 文本转索引
def text2idx(text):
    tokens = text.split()
    return [word2idx.get(w, word2idx["<unk>"]) for w in tokens]

# 构建训练数据（输入：前n-1词，标签：后n-1词）
data = []
for sentence in corpus:
    idx = text2idx(sentence)
    x = idx[:-1]  # 输入
    y = idx[1:]   # 标签
    # 填充到固定长度
    x += [word2idx["<pad>"]] * (seq_len - len(x))
    y += [word2idx["<pad>"]] * (seq_len - len(y))
    data.append((x, y))

# 转张量
x = torch.tensor([d[0] for d in data], dtype=torch.long).to(device)
y = torch.tensor([d[1] for d in data], dtype=torch.long).to(device)

# ===================== 3. 位置编码 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

# ===================== 4. 单向 Transformer 解码器模型 =====================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 输出层（映射到词典）
        self.fc = nn.Linear(d_model, vocab_size)

    # 生成单向掩码（看不到后面的词）
    def generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask.to(device)

    def forward(self, x):
        # 嵌入 + 位置编码
        x_emb = self.embedding(x) * np.sqrt(self.d_model)
        x_emb = self.pos_encoder(x_emb)
        
        # 生成单向掩码
        tgt_mask = self.generate_mask(x.size(1))
        
        # 解码器前向传播
        output = self.transformer_decoder(
            tgt=x_emb, 
            memory=x_emb,  # 语言模型无编码器，memory用自身代替
            tgt_mask=tgt_mask
        )
        # 分类输出
        out = self.fc(output)
        return out

# ===================== 5. 初始化模型、损失、优化器 =====================
model = TransformerLM(vocab_size, d_model, nhead, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])  # 忽略padding
optimizer = optim.Adam(model.parameters(), lr=lr)

# ===================== 6. 训练模型 =====================
print("开始训练...")
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x)  # [batch, seq_len, vocab]
    
    # 调整形状计算损失
    loss = criterion(outputs.reshape(-1, vocab_size), y.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ===================== 7. 文本生成函数 =====================
def generate_text(model, start_text, max_len=15):
    model.eval()
    words_idx = text2idx(start_text)
    generated = words_idx.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
            output = model(input_tensor)
            
            # 取最后一个词的预测结果
            last_word_logits = output[0, -1, :]
            # 贪婪采样（选概率最大的词）
            pred_idx = torch.argmax(last_word_logits).item()
            
            generated.append(pred_idx)
            # 遇到结束符停止
            if pred_idx == word2idx["<eos>"]:
                break
    
    # 转文本
    result = [idx2word[idx] for idx in generated]
    return " ".join(result)

# ===================== 8. 测试生成 =====================
print("\n===== 文本生成结果 =====")
test_start = "单向 语言"
print(f"输入开头：{test_start}")
print(f"生成结果：{generate_text(model, test_start)}")