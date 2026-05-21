"""
训练基于transformer的单向语言模型，并完成文本生成。
1、训练模型 (作业1)
2、文本生成 (作业2)

文本生成：支持 Beam Search、Top-P + Temperature、Top-K + Top-P（兜底）
"""

import torch
import torch.nn.functional as F
import argparse
import random

from 作业1 import TransformerModel 

def load_model_and_vocab(checkpoint_path, device):
    """加载保存的模型、词表和参数"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint["args"]
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    vocab_size = len(idx2char)
    
    model = TransformerModel(
        vocab_size=vocab_size,
        seq_len=args["seq_len"],
        embed_dim=args["embed_dim"],
        dim_feedforward=args["ff_dim"],
        num_head=args["num_head"],
        num_layers=args["num_layers"],
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, char2idx, idx2char, args

def beam_search_generate(model, char2idx, idx2char, start_str, beam_width=3, max_new_tokens=50, device='cpu'):
    """
    Beam Search 生成
    返回最佳序列的字符串和得分
    """
    # 将起始字符串转为 token ids
    start_ids = [char2idx.get(c, 0) for c in start_str]  # 未登录字符用 0 (padding)
    if len(start_ids) == 0:
        # 如果起始字符串为空
        start_ids = [0]
    input_tensor = torch.tensor([start_ids], device=device)  # (1, seq_len)
    
    # beams: 列表元素为 (序列token列表, 对数概率)
    beams = [(start_ids, 0.0)]
    completed = []
    
    for _ in range(max_new_tokens):
        new_beams = []
        for seq, score in beams:
            if len(seq) >= 1 and seq[-1] == char2idx.get('\n', 0):  # 如果已生成换行，视为结束
                completed.append((seq, score))
                continue
            # 输入当前序列（需要保证长度不超过模型支持范围，这里直接取全部）
            input_tensor = torch.tensor([seq], device=device)
            with torch.no_grad():
                logits = model(input_tensor)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]  # 最后一个位置的logits
            log_probs = F.log_softmax(next_logits, dim=-1)  # (vocab_size,)
            
            # 取前 beam_width 个候选
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                new_seq = seq + [top_indices[k].item()]
                new_score = score + top_log_probs[k].item()
                new_beams.append((new_seq, new_score))
        
        # 保留得分最高的 beam_width 条
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
    
    # 合并已完成和未完成的beams
    all_candidates = completed + beams
    if not all_candidates:
        return "", -float('inf')
    best_seq, best_score = max(all_candidates, key=lambda x: x[1])
    generated_text = ''.join([idx2char.get(idx, '') for idx in best_seq[len(start_ids):]])
    return start_str + generated_text, best_score

def top_p_temperature_generate(model, char2idx, idx2char, start_str, top_p=0.9, temperature=1.0, max_new_tokens=50, device='cpu'):
    """
    Top-P (nucleus sampling) + Temperature 生成
    """
    start_ids = [char2idx.get(c, 0) for c in start_str]
    if not start_ids:
        start_ids = [0]
    generated = start_ids[:]
    
    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([generated], device=device)
        with torch.no_grad():
            logits = model(input_tensor)
        next_logits = logits[0, -1, :] / temperature  # 温度缩放
        
        # Top-p 采样
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过 top_p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 至少保留一个token
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        next_logits[indices_to_remove] = -float('inf')
        
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        
        # 遇到换行停止（可选）
        if next_token == char2idx.get('\n', -1):
            break
    
    return start_str + ''.join([idx2char.get(idx, '') for idx in generated[len(start_ids):]])

def top_k_top_p_generate(model, char2idx, idx2char, start_str, top_k=40, top_p=0.9, temperature=1.0, max_new_tokens=50, device='cpu'):
    """
    结合 Top-K 和 Top-P 生成：先做 Top-K 过滤，再做 Top-P 采样，最后做 Temperature 调整
    """
    start_ids = [char2idx.get(c, 0) for c in start_str]
    if not start_ids:
        start_ids = [0]
    generated = start_ids[:]
    
    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([generated], device=device)
        with torch.no_grad():
            logits = model(input_tensor)
        next_logits = logits[0, -1, :] / temperature
        
        # 1) Top-K 过滤：保留概率最高的 top_k 个 token
        top_k_vals, top_k_indices = torch.topk(next_logits, top_k)
        mask = torch.full_like(next_logits, -float('inf'))
        mask.scatter_(0, top_k_indices, top_k_vals)
        next_logits = mask
        
        # 2) Top-P 采样：基于 Top-K 后的概率分布
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        next_logits[indices_to_remove] = -float('inf')
        
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        
        if next_token == char2idx.get('\n', -1):
            break
    
    return start_str + ''.join([idx2char.get(idx, '') for idx in generated[len(start_ids):]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="训练保存的模型路径")
    parser.add_argument("--start_str", type=str, default="今天", help="起始文本")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成新token数")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/mps/cpu)")
    args = parser.parse_args()
    
    # 确定设备
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # 加载模型和词表
    print(f"加载模型: {args.checkpoint}")
    model, char2idx, idx2char, model_args = load_model_and_vocab(args.checkpoint, device)
    print(f"词表大小: {len(idx2char)}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")
    print(f"起始字符串: {args.start_str}\n")
    
    # 1. Beam Search (beam_width=3)
    print("="*50)
    print("1. Beam Search (beam_width=3)")
    beam_text, score = beam_search_generate(
        model, char2idx, idx2char, args.start_str,
        beam_width=3, max_new_tokens=args.max_new_tokens, device=device
    )
    print(f"生成结果: {beam_text}\n")
    
    # 2. Top-P + Temperature
    print("="*50)
    print("2. Top-P (p=0.9) + Temperature (t=1.0)")
    tp_text = top_p_temperature_generate(
        model, char2idx, idx2char, args.start_str,
        top_p=0.9, temperature=1.0, max_new_tokens=args.max_new_tokens, device=device
    )
    print(f"生成结果: {tp_text}\n")
    
    # 3. Top-P + Top-K 兜底
    print("="*50)
    print("3. Top-K (k=40) + Top-P (p=0.9) + Temperature (t=1.0)")
    tktp_text = top_k_top_p_generate(
        model, char2idx, idx2char, args.start_str,
        top_k=10, top_p=0.9, temperature=1.0, max_new_tokens=args.max_new_tokens, device=device
    )
    print(f"生成结果: {tktp_text}\n")

if __name__ == "__main__":
    main()
