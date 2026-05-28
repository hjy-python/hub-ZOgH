"""
BERT 文本分类训练 - 联想小新CPU极速版
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from evaluate import evaluate_model

ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
BERT_PATH     = ROOT / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR    = ROOT / "outputs"
CKPT_DIR      = OUTPUT_DIR / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 文本分类训练-CPU极速版")
    parser.add_argument("--bert_path",      default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir",       default=str(DATA_DIR),  type=str)
    parser.add_argument("--output_dir",     default=str(OUTPUT_DIR), type=str)
    parser.add_argument("--pool",           default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--epochs",         default=1,   type=int)
    parser.add_argument("--batch_size",     default=16,  type=int)  # CPU最优batch
    parser.add_argument("--max_length",     default=16, type=int)   # 序列减半，计算量减半
    parser.add_argument("--lr",             default=0, type=float)  # 冻结BERT，只训分类头
    parser.add_argument("--head_lr_mult",   default=10.0,  type=float)
    parser.add_argument("--dropout",        default=0.1,  type=float)
    parser.add_argument("--warmup_ratio",   default=0.1,  type=float)
    parser.add_argument("--grad_accum",     default=2,    type=int)  # 等效batch=32
    parser.add_argument("--use_class_weight", action="store_true")
    return parser.parse_args()


def compute_loss_weights(data_dir: Path, num_labels: int, device: torch.device):
    train_file = data_dir / "train.json"
    label_file = data_dir / "label_map.json"
    
    if not train_file.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {train_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"标签映射文件不存在: {label_file}")

    with open(train_file, encoding="utf-8") as f:
        train_data = json.load(f)
    
    labels = np.array([item["label"] for item in train_data])
    classes = np.arange(num_labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    
    print("类别权重（用于加权 loss）：")
    with open(label_file, encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}
    
    for i, w in enumerate(weights):
        print(f"  {i:2d} {id2name[i]:4s}: {w:.3f}")
    
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion,
    device, epoch, total_epochs, grad_accum
):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        # 🔥 CPU极速优化：只跑前1250步，等效10000条数据
        if step >= 1250:
            break
            
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}",
                         acc=f"{total_correct/total_samples:.4f}")

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc  = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc


def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bert_path = Path(args.bert_path)
    if not bert_path.exists():
        raise FileNotFoundError(
            f"BERT 预训练模型路径不存在: {bert_path}\n"
            f"请下载 bert-base-chinese 并放到正确位置，或通过 --bert_path 指定路径"
        )

    device = torch.device("cpu")  # 强制CPU，避免不必要的检测
    print(f"使用设备: {device}")

    # 加载 label_map
    label_file = data_dir / "label_map.json"
    if not label_file.exists():
        raise FileNotFoundError(f"标签映射文件不存在: {label_file}")
    
    with open(label_file, encoding="utf-8") as f:
        label_map = json.load(f)
    
    num_labels = label_map["num_labels"]
    id2name    = {int(k): v for k, v in label_map["id2name"].items()}
    print(f"类别数: {num_labels}")

    # Tokenizer & DataLoader
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, _ = build_dataloaders(
        data_dir, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # 🔥 CPU极速优化：验证集只跑前250步
    val_loader.dataset.data = val_loader.dataset.data[:2000]

    print(f"DataLoader 构建完成  快速模式: train: 10000 条, val: 2000 条")

    # 模型
    model = build_model(args.bert_path, num_labels, pool=args.pool)
    model = model.to(device)

    # 🔥 真正冻结BERT，关闭梯度计算，速度×3~×5
    for name, param in model.named_parameters():
        if "bert" in name:
            param.requires_grad = False
    print("已冻结 BERT 主干参数，仅训练分类头")

    # Loss
    if args.use_class_weight:
        weights = compute_loss_weights(data_dir, num_labels, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("使用加权 CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用普通 CrossEntropyLoss")

    # 优化器
    bert_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "bert" in name:
            bert_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    # 强制总步数=1250，warmup=125
    total_steps = 1250
    warmup_steps = 125
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"总训练步数: {total_steps}, warmup: {warmup_steps}")

    # 训练循环
    best_val_acc = 0.0
    log_records  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum
        )
        val_metrics = evaluate_model(model, val_loader, device, id2name, print_report=True)
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["macro_f1"]
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_macro_f1": val_f1, "elapsed_s": elapsed,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
            ckpt_path = ckpt_dir / f"best_{run_tag}.pt"
            torch.save({
                "epoch":           epoch,
                "pool":            args.pool,
                "use_class_weight": args.use_class_weight,
                "state_dict":      model.state_dict(),
                "val_acc":         val_acc,
                "val_macro_f1":    val_f1,
                "args":            vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_acc={val_acc:.4f})")

    # 保存训练日志
    run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
    log_path = output_dir / f"train_log_{run_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_acc={best_val_acc:.4f}")
    print(f"训练日志 → {log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 训练出错: {type(e).__name__}: {e}")
        exit(1)
