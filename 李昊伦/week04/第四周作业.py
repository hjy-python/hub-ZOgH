# coding:utf8
"""
Transformer层实现与训练示例
基于PyTorch手动实现一个完整的Transformer层
任务：五分类任务 —— x是一个5维向量，哪一维数字最大就为第几类
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# ===================== 1. 位置编码 =====================
class PositionalEncoding(nn.Module):
    """
    位置编码：为输入注入位置信息
    Transformer没有循环或卷积结构，需要手动添加位置信息
    公式：PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 注册为buffer，不参与梯度更新

    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


# ===================== 2. 多头自注意力 =====================
class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    核心思想：把注意力分成多个"头"，每个头独立计算注意力，最后拼接
    Q = X * W_q,  K = X * W_k,  V = X * W_v
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q、K、V的线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        # 1) 线性变换得到 Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) 拆分成多个头: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3) 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4) softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5) 加权求和
        context = torch.matmul(attn_weights, V)
        # context: (batch, num_heads, seq_len, d_k)

        # 6) 拼接所有头: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 7) 输出线性变换
        output = self.W_o(context)
        return output, attn_weights


# ===================== 3. 前馈神经网络 =====================
class FeedForward(nn.Module):
    """
    位置前馈网络 (Position-wise Feed-Forward Network)
    两层全连接 + ReLU激活
    FFN(x) = max(0, x * W1 + b1) * W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ===================== 4. Transformer层 =====================
class TransformerLayer(nn.Module):
    """
    一个完整的Transformer层
    结构：
    输入 -> [多头自注意力 + 残差连接 + LayerNorm] -> [前馈网络 + 残差连接 + LayerNorm] -> 输出
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()
        # 多头自注意力
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 两个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 第一步：多头自注意力 + 残差 + LayerNorm
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))  # 残差连接 + LayerNorm

        # 第二步：前馈网络 + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))  # 残差连接 + LayerNorm

        return x, attn_weights


# ===================== 5. 完整Transformer模型 =====================
class TransformerClassifier(nn.Module):
    """
    基于Transformer的五分类模型
    结构：
    输入(5维向量) -> 线性投影到d_model维度 -> Transformer层 -> 全局平均池化 -> 分类头
    """

    def __init__(self, input_dim=5, num_classes=5, d_model=64, num_heads=4,
                 d_ff=128, num_layers=1, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.d_model = d_model

        # 输入投影：将5维向量映射到d_model维
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=16)

        # Transformer层（可以堆叠多层）
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, input_dim) 即 (batch_size, 5)
        """
        # 1) 输入投影: (batch, 5) -> (batch, 1, d_model)
        x = self.input_projection(x).unsqueeze(1)  # 增加一个序列维度

        # 2) 加入位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 3) 通过Transformer层
        attn_weights = None
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)

        # 4) 全局平均池化: (batch, 1, d_model) -> (batch, d_model)
        x = x.mean(dim=1)

        # 5) 分类输出: (batch, d_model) -> (batch, num_classes)
        logits = self.classifier(x)
        return logits


# ===================== 6. 数据生成 =====================
def build_sample():
    """生成一个样本：5维随机向量，哪维最大就是第几类"""
    x = np.random.random(5).astype('float32')
    y = int(np.argmax(x))  # 哪一维最大
    return x, y


def build_dataset(sample_num):
    """批量生成样本"""
    dataset_x, dataset_y = [], []
    for _ in range(sample_num):
        x, y = build_sample()
        dataset_x.append(x)
        dataset_y.append(y)
    return np.array(dataset_x), np.array(dataset_y)


def evaluate(model, test_samples=1000):
    """评估模型准确率"""
    model.eval()
    x, y = build_dataset(test_samples)
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.LongTensor(y)

    with torch.no_grad():
        pred_logits = model(x_tensor)
        pred = torch.argmax(pred_logits, dim=1)
        correct = (pred == y_tensor).sum().item()

    accuracy = correct / test_samples
    return accuracy


# ===================== 7. 主训练流程 =====================
def main():
    # ---- 超参数 ----
    EPOCHS = 30
    BATCH_SIZE = 20
    TRAIN_SAMPLES = 5000
    LR = 1e-3
    INPUT_DIM = 5
    NUM_CLASSES = 5
    D_MODEL = 64
    NUM_HEADS = 4
    D_FF = 128
    NUM_LAYERS = 2

    print("=" * 60)
    print("Transformer层实现 - 五分类任务")
    print("=" * 60)
    print(f"模型参数: d_model={D_MODEL}, heads={NUM_HEADS}, "
          f"d_ff={D_FF}, layers={NUM_LAYERS}")
    print(f"训练参数: epochs={EPOCHS}, batch={BATCH_SIZE}, "
          f"samples={TRAIN_SAMPLES}, lr={LR}")
    print("=" * 60)

    # ---- 创建模型 ----
    model = TransformerClassifier(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # ---- 优化器 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ---- 生成数据 ----
    train_x, train_y = build_dataset(TRAIN_SAMPLES)

    # ---- 记录训练过程 ----
    loss_list = []
    accuracy_list = []

    # ---- 训练循环 ----
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0

        # 打乱数据
        indices = np.random.permutation(TRAIN_SAMPLES)
        train_x = train_x[indices]
        train_y = train_y[indices]

        for batch_index in range(TRAIN_SAMPLES // BATCH_SIZE):
            # 取一个batch
            x = torch.FloatTensor(train_x[batch_index * BATCH_SIZE:
                                          (batch_index + 1) * BATCH_SIZE])
            y = torch.LongTensor(train_y[batch_index * BATCH_SIZE:
                                         (batch_index + 1) * BATCH_SIZE])

            # 前向传播
            pred_logits = model(x)
            loss = criterion(pred_logits, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        # 每轮评估
        avg_loss = total_loss / batch_count
        accuracy = evaluate(model)
        loss_list.append(avg_loss)
        accuracy_list.append(accuracy)

        print(f"第{epoch + 1:2d}轮 | 平均loss: {avg_loss:.6f} | 准确率: {accuracy:.4f}")

    print(f"\n训练完成！最终准确率: {accuracy_list[-1]:.4f}")

    # ---- 保存模型 ----
    model_path = "transformer_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    # ---- 绘制训练曲线 ----
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), loss_list, 'b-o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), accuracy_list, 'r-o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('transformer_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("训练曲线已保存: transformer_training_curves.png")

    # ---- 演示预测 ----
    print("\n" + "=" * 60)
    print("预测演示")
    print("=" * 60)
    test_cases = [
        np.array([0.9, 0.1, 0.2, 0.3, 0.4], dtype='float32'),
        np.array([0.1, 0.8, 0.2, 0.3, 0.4], dtype='float32'),
        np.array([0.1, 0.2, 0.9, 0.3, 0.4], dtype='float32'),
        np.array([0.1, 0.2, 0.3, 0.8, 0.4], dtype='float32'),
        np.array([0.1, 0.2, 0.3, 0.4, 0.9], dtype='float32'),
    ]

    model.eval()
    for vec in test_cases:
        with torch.no_grad():
            input_tensor = torch.FloatTensor(vec).unsqueeze(0)
            pred_logits = model(input_tensor)
            pred_class = torch.argmax(pred_logits, dim=1).item()
            probabilities = F.softmax(pred_logits, dim=1).squeeze()
            max_idx = int(np.argmax(vec))
            status = "[CORRECT]" if pred_class == max_idx else "[WRONG]"
            print(f"  输入: {vec} -> 预测类别: {pred_class} "
                  f" (真实: {max_idx}) {status}")
            print(f"         各类概率: {[f'{p:.3f}' for p in probabilities.tolist()]}")

    # ---- 打印模型结构 ----
    print("\n" + "=" * 60)
    print("模型结构")
    print("=" * 60)
    print(model)


if __name__ == "__main__":
    main()
