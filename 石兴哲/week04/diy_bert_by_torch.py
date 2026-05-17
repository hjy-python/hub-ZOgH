import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertModel

from rich import print

class DIYBert(nn.Module):
    ''' 
    自定义DIY Bert模型
    但是所有的参数及词表从已经预训练的google开源的 bert-base-chinese 模型中加载
    主要模拟3个部分：
    1. embedding层
    2. transformer层
    3. pooler层
    '''
    
    def __init__(self, state_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        # 超参数初始化
        self.hidden_size             = 768
        self.num_attention_heads     = 12       # 可以理解为几个头 几个头在self-attention的时候，各算各的 再拼起来
        self.num_hidden_layers       = 12       # transformer的层数
        self.intermediate_size       = 3072     # 768 * 4
        self.max_position_embeddings = 512      # bert的缺点，固定长度
        self.type_vocab_size         = 2
        self.layer_norm_eps          = 1e-12    #epsilon 该参数作用是防止除以0 也可以用默认的 1e-5 bert对精度要求很高
        self.attention_head_size     = self.hidden_size // self.num_attention_heads  # d_k = 64 可以理解为每个头的大小 
        self.hidden_act              = "gelu"                                        # 激活函数
          # self.initializer_range = 0.02 # 初始化参数的范围

        vocab_size = self._infer_vocab_size(state_dict)
        print(f"vocab_size: {vocab_size}") # 21128 汉字比英文少

        '''Step1： 三个embedding层：TE SE PE  加上一个归一化层'''
        self.word_embeddings        = nn.Embedding(vocab_size, self.hidden_size) # 21128 * 768
        self.token_type_embeddings  = nn.Embedding(self.type_vocab_size, self.hidden_size) # 2 * 768
        self.position_embeddings    = nn.Embedding(self.max_position_embeddings, self.hidden_size) # 512 * 768
        self.embeddings_layer_norm  = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        '''Step2： transformer层 这里用的是nn.ModuleList 不是一个单层；构造方式是使用nn.ModuleDict'''
        self.transformer_layers = nn.ModuleList(
            [self._build_transformer_layer() for _ in range(self.num_hidden_layers)]
        )
        
        '''Step3： pooler层'''
        self.pooler_dense = nn.Linear(self.hidden_size, self.hidden_size)

        if state_dict is not None:
            self.load_weights(state_dict) # 加载预训练的参数 从bert-base-chinese中
        else:
            raise ValueError("state_dict is None")
        
    def forward(self, x, token_type_ids: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embedding_forward(x, token_type_ids)
        sequence_output = self.all_transformer_layer_forward(hidden_states)
        pooler_output = self.pooler_output_layer(sequence_output)
        return sequence_output, pooler_output
        
    def _infer_vocab_size(self, state_dict: Optional[Dict[str, torch.Tensor]]) -> int:
        if state_dict is None:
            return 21128
        return int(self._get_tensor(state_dict, "embeddings.word_embeddings.weight").shape[0])

    def _build_transformer_layer(self) -> nn.ModuleDict:
        '''一个transformer层中包含8个层'''
        return nn.ModuleDict(
            {
                "query": nn.Linear(self.hidden_size, self.hidden_size),
                "key":   nn.Linear(self.hidden_size, self.hidden_size),
                "value": nn.Linear(self.hidden_size, self.hidden_size),
                "attention_dense": nn.Linear(self.hidden_size, self.hidden_size),
                
                "attention_layer_norm": nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
                "intermediate_dense"  : nn.Linear(self.hidden_size, self.intermediate_size),
                "output_dense"        : nn.Linear(self.intermediate_size, self.hidden_size),
                "output_layer_norm"   : nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
            }
        )

    def _get_tensor(self, state_dict: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key in state_dict:
            return state_dict[key]
        bert_key = f"bert.{key}"
        if bert_key in state_dict:
            return state_dict[bert_key]
        raise KeyError(f"Can not find weight: {key}")

    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.word_embeddings.weight.copy_(
                self._get_tensor(state_dict, "embeddings.word_embeddings.weight")
            )
            self.position_embeddings.weight.copy_(
                self._get_tensor(state_dict, "embeddings.position_embeddings.weight")
            )
            self.token_type_embeddings.weight.copy_(
                self._get_tensor(state_dict, "embeddings.token_type_embeddings.weight")
            )
            self.embeddings_layer_norm.weight.copy_(
                self._get_tensor(state_dict, "embeddings.LayerNorm.weight")
            )
            self.embeddings_layer_norm.bias.copy_(
                self._get_tensor(state_dict, "embeddings.LayerNorm.bias")
            )

            for i, layer in enumerate(self.transformer_layers):
                layer["query"].weight.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.attention.self.query.weight")
                )
                layer["query"].bias.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.attention.self.query.bias")
                )
                layer["key"].weight.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.attention.self.key.weight")
                )
                layer["key"].bias.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.attention.self.key.bias")
                )
                layer["value"].weight.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.attention.self.value.weight")
                )
                layer["value"].bias.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.attention.self.value.bias")
                )
                layer["attention_dense"].weight.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.attention.output.dense.weight")
                )
                layer["attention_dense"].bias.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.attention.output.dense.bias")
                )
                layer["attention_layer_norm"].weight.copy_(
                    self._get_tensor(
                        state_dict,
                        f"encoder.layer.{i}.attention.output.LayerNorm.weight",
                    )
                )
                layer["attention_layer_norm"].bias.copy_(
                    self._get_tensor(
                        state_dict,
                        f"encoder.layer.{i}.attention.output.LayerNorm.bias",
                    )
                )
                layer["intermediate_dense"].weight.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.intermediate.dense.weight")
                )
                layer["intermediate_dense"].bias.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.intermediate.dense.bias")
                )
                layer["output_dense"].weight.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.output.dense.weight")
                )
                layer["output_dense"].bias.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.output.dense.bias")
                )
                layer["output_layer_norm"].weight.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.output.LayerNorm.weight")
                )
                layer["output_layer_norm"].bias.copy_(
                    self._get_tensor(state_dict, f"encoder.layer.{i}.output.LayerNorm.bias")
                )

            self.pooler_dense.weight.copy_(self._get_tensor(state_dict, "pooler.dense.weight"))
            self.pooler_dense.bias.copy_(self._get_tensor(state_dict, "pooler.dense.bias"))


    def embedding_forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = self.word_embeddings.weight.device
        input_ids = input_ids.to(device)
        print("device:", device)

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, device=device)
        else:
            token_type_ids = token_type_ids.to(device).long()

        word_embedding = self.word_embeddings(input_ids)
        position_embedding = self.position_embeddings(position_ids)
        token_type_embedding = self.token_type_embeddings(token_type_ids)

        embedding = word_embedding + position_embedding + token_type_embedding
        return self.embeddings_layer_norm(embedding)

    def all_transformer_layer_forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_hidden_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    # 单层transformer
    def single_transformer_layer_forward(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        '''单层transformer的forward实现: 
        self-attention
        residual connection & layer normalization
        feed_forward & residual connection & layer normalization
        '''
        layer = self.transformer_layers[layer_index] # 获取当前层的参数

        '''step 1: self-attention'''
        attention_output = self.self_attention_forward(
            x = x,
            query_layer = layer["query"],
            key_layer   = layer["key"],
            value_layer = layer["value"],
            attention_output_dense = layer["attention_dense"],
        )
        '''step 2: residual connection + layer normalization'''
        x = layer["attention_layer_norm"](x + attention_output)

        '''step 3: feed_forward & residual connection + layer normalization'''
        feed_forward_output = self.feed_forward(
            x = x,
            intermediate_dense = layer["intermediate_dense"],
            output_dense = layer["output_dense"],
        )
        x = layer["output_layer_norm"](x + feed_forward_output)
        return x

    def self_attention_forward(
        self,
        x: torch.Tensor,
        query_layer: nn.Linear,
        key_layer: nn.Linear,
        value_layer: nn.Linear,
        attention_output_dense: nn.Linear,
    ) -> torch.Tensor:
        '''self-attention实现'''
        
        '''把x都过一遍线性层之后，生成了Q K V， 再进行【分头】，通过transpose_for_scores函数增加一维，按照【头数】进行切分'''
        query   = self.transpose_for_scores(query_layer(x))
        key     = self.transpose_for_scores(key_layer(x))
        value   = self.transpose_for_scores(value_layer(x))
        # print("query.shape:", query.shape) # torch.Size([1, 12, 4, 64]) # [batch_size, num_attention_heads, seq_len, attention_head_size]
        # print("key.shape:", key.shape) # torch.Size([1, 12, 4, 64])
        # print("value.shape:", value.shape) # torch.Size([1, 12, 4, 64])
        
        ''' Q * K^T 形成了词表'''
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        ''' divide by sqrt(d_k) 就是固定的 8  attention_head_size=64'''
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        ''' 再过一遍softmax '''
        attention_probs  = torch.softmax(attention_scores, dim=-1)
        # print("attention_probs.shape:", attention_probs.shape) # torch.Size([1, 12, 4, 4]) # [batch_size, num_attention_heads, seq_len, seq_len]
        ''' 过完softmax之后 attention_probs * V '''
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_len = x.shape[:2]
        context = context.view(batch_size, seq_len, self.hidden_size)
        return attention_output_dense(context)

    # 
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        # print("x.shape:", x.shape)  # torch.Size([1, 4, 768])
        '''这个地方开始多头：
        x的原始形状  [1, 4, 768] batch_size, seq_len, hidden_size
        转为        [1, 4, 12, 64] batch_size, seq_len, num_attention_heads, attention_head_size
        把768维转为了 12个头，每个头64维 
        '''
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        '''返回的时候，把第2维和第3维交换一下，目的是把头放到前面，先对后面的按头切分的数据计算，计算完毕之后再做合并
        返回形状    [1, 12, 4, 64] batch_size, num_attention_heads, seq_len, attention_head_size'''
        return x.permute(0, 2, 1, 3).contiguous()

    def feed_forward(
        self,
        x: torch.Tensor,
        intermediate_dense: nn.Linear,
        output_dense: nn.Linear,
    ) -> torch.Tensor:
        '''feed_forward实现 简单 2个线性层中间夹一个激活函数gelu
        在这里就是： intermediate_dense(x) -> gelu -> output_dense(x)
        '''
        x = intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        return output_dense(x)

    def pooler_output_layer(self, x: torch.Tensor) -> torch.Tensor:
        cls_output = x[:, 0]
        return torch.tanh(self.pooler_dense(cls_output))


def main() -> None:
    pretrained_model_path = r"D:\mydocs\pretrained_models\bert-base-chinese"
    bert = BertModel.from_pretrained(pretrained_model_path, return_dict=False)
    bert.eval()
    state_dict = bert.state_dict()
    # print("state_dict keys from google bert:", state_dict.keys())

    # x = np.array([2450, 15486, 102, 2110], dtype=np.int64)
    torch_x = torch.LongTensor([2450, 15486, 102, 2110]).unsqueeze(0)

    diy_bert = DIYBert(state_dict)
    diy_bert.eval()

    with torch.no_grad():
        torch_sequence_output, torch_pooler_output = bert(torch_x)
        diy_sequence_output, diy_pooler_output = diy_bert(torch_x)

        print("torch_sequence_output:")
        print(torch_sequence_output)
        print("diy_sequence_output:")
        print(diy_sequence_output)


if __name__ == "__main__":
    main()
