import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""
Transformer
输入 -> embeding+position -> encoder -> decoder -> softmax -> 输出

Encoder:
multi-head self-attention -> add & norm -> feed forward -> add & norm

Decoder:
multi-head self-attention -> add & norm -> multi-head cross attention -> add & norm -> feed forward -> add & norm

Positional Encoding:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        def transform(x, linear):
            batch_size = x.size(0)
            x = linear(x)
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2) # token : head -> head : token
        q = transform(q, self.linear_q)
        k = transform(k, self.linear_k)
        v = transform(v, self.linear_v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(q.size(0), -1, self.num_heads * self.d_k) # head : token -> token : head
        return self.linear_out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x))) # 先线性变换到更高维度，增加模型表达能力，再线性变换回原维度
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_out = self.attn(x, x, x, mask) # 自注意力机制，输入的q、k、v都是x
        x = self.norm1(x + attn_out) # 残差连接 + LayerNorm
        ffn_out = self.ffn(x) # 前馈神经网络，增加模型的非线性表达能力
        return self.norm2(x + ffn_out) # 残差连接 + LayerNorm
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, k, v, mask):
        self_attn_out = self.self_attn(x, x, x, mask) # 解码器自注意力机制，输入的q、k、v都是x
        x = self.norm1(x + self_attn_out) # 残差连接 + LayerNorm
        cross_attn = self.cross_attn(x, k, v, mask) # 编码器-解码器注意力机制，q来自解码器，k和v来自编码器
        x = self.norm2(x + cross_attn) # 残差连接 + LayerNorm
        ffn_out = self.ffn(x) # 前馈神经网络，增加模型的非线性表达能力
        return self.norm3(x + ffn_out) # 残差连接 + LayerNorm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model) # 位置编码矩阵，大小为(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # 位置索引，大小为(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # 位置编码的频率，大小为(d_model/2,)
        self.pe[:, 0::2] = torch.sin(position * div_term) # 偶数维使用正弦函数，大小为(max_len, d_model/2)
        self.pe[:, 1::2] = torch.cos(position * div_term) # 奇数维使用余弦函数，大小为(max_len, d_model/2)
        self.register_buffer('pe', self.pe) # 将位置编码矩阵注册为模型的buffer，这样在保存和加载模型时会自动处理，并且在训练过程中不会被更新

    def forward(self, x):
        return x + self.pe[:x.size(1)] # 将位置编码加到输入的嵌入上
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_decoder_layers)])
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, mask=None):
        src_emb = self.dropout(self.positional_encoding(self.embedding(src))) # 对输入的源序列进行嵌入和位置编码
        tgt_emb = self.dropout(self.positional_encoding(self.embedding(tgt))) # 对输入的目标序列进行嵌入和位置编码
        
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, mask) # 依次通过编码器层
        
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, src_emb, src_emb, mask) # 依次通过解码器层，k和v来自编码器的输出
        
        return self.fc_out(tgt_emb) # 最后通过线性层映射到词汇表大小，得到每个位置的预测分布