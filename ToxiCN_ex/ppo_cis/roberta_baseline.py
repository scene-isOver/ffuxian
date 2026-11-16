"""纯 RoBERTa 单分类器基线（独立脚本，不参与 PPO 级联）。

目的：验证 ``roberta_fusion`` 的 0.7404 究竟是架构不行，还是
``freeze_layers=8`` + ``dropout=0.5`` 把 12 层模型压成了 4 层模型。
这里用完全相同的清洗、切分、分词、早停和延迟测量口径，只把模型换成
标准 RoBERTa（12 层全部微调）。

隔离性：
    * 不修改任何现有文件，不写入 scores，不参与级联；
    * 产物单独命名（roberta_plain.pt / roberta_plain_metrics.json）；
    * 想彻底回滚，删掉本文件 + 上面两个产物即可。

用法（在 ToxiCN_ex 目录下）：
    python -m ppo_cis.roberta_baseline
    python -m ppo_cis.roberta_baseline --epochs 5 --lr 2e-5 --dropout 0.1
    python -m ppo_cis.roberta_baseline --model_name D:/models/chinese-roberta-wwm-ext
"""
from __future__ import annotations

import argparse
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification, get_linear_schedule_with_warmup,
)

from .config import CLASSIFIER_DIR, ROBERTA_MODEL_NAME, DataConfig, resolve_model_name
from .data import (
    EncodedDataset, build_splits, encode_split, get_tokenizer, print_split_stats,
)
from .utils import (
    amp_context, compute_metrics, get_device, grad_scaler, measure_latency, save_json,
)


# ===================== 评估 =====================
# 与 train_classifiers.evaluate 完全一致：argmax 取预测，统一走 compute_metrics。
@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    losses, preds, labels = [], [], []
    for batch in loader:
        target = batch["labels"].to(device)
        with amp_context(device):
            logits = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device)).logits
            loss = criterion(logits, target)
        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        labels.extend(target.cpu().numpy().tolist())
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(
        description="Standalone pure-RoBERTa baseline (binary ToxiCN classification)")
    parser.add_argument("--model_name", default=ROBERTA_MODEL_NAME,
                        help="RoBERTa 预训练权重的本地目录")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--max_len", default=96, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="覆盖 hidden/attention dropout，base 模型用 0.1 更合适")
    parser.add_argument("--patience", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no_save", action="store_true",
                        help="只打印指标，不落盘 checkpoint 和 json")
    args = parser.parse_args()

    # 固定随机种子：与 train_classifiers 使用同一套全局种子，保证可比且可复现。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")

    # 1) 数据：与主流程同一个 DataConfig，保证四个划分逐条一致。
    data_cfg = DataConfig(max_len=args.max_len, seed=args.seed)
    splits = build_splits(data_cfg)
    print_split_stats(splits)

    # 2) 分词：encode_split 自带缓存，与 bert 的缓存互不冲突（键含 tokenizer 名）。
    model_path = resolve_model_name(args.model_name)
    print(f"Backbone: {model_path}")
    tokenizer = get_tokenizer(model_path)
    train_t = encode_split(splits["clf_train"], tokenizer, "clf_train", args.max_len)
    dev_t = encode_split(splits["clf_dev"], tokenizer, "clf_dev", args.max_len)
    test_t = encode_split(splits["test"], tokenizer, "test", args.max_len)

    train_loader = DataLoader(EncodedDataset(train_t), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    dev_loader = DataLoader(EncodedDataset(dev_t), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(EncodedDataset(test_t), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # 3) 模型：12 层全量微调，分类头新建；dropout 按参数覆盖。
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, local_files_only=True).to(device)
    for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
        if hasattr(model.config, attr):
            setattr(model.config, attr, args.dropout)
    if getattr(model.config, "classifier_dropout", None) is not None:
        model.config.classifier_dropout = args.dropout

    # 4) 优化器 / 调度器 / 损失，与主流程同一套（warmup 10% + 线性衰减）。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = grad_scaler(device)

    # 5) 训练主循环：dev F1 早停，与 train_classifiers 保持一致的口径。
    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, -1, 0
    for epoch in range(args.epochs):
        model.train()
        running_loss, seen_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"roberta_plain epoch {epoch + 1}/{args.epochs}",
                    leave=False)
        for batch in pbar:
            target = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device)).logits
                loss = criterion(logits, target)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            seen_batches += 1
            pbar.set_postfix(loss=f"{running_loss / seen_batches:.4f}")

        dev_metrics = evaluate(model, dev_loader, device, criterion)
        print(f"[roberta_plain] epoch {epoch + 1}: "
              f"dev_acc={dev_metrics['accuracy']:.4f} dev_f1={dev_metrics['f1']:.4f} "
              f"dev_loss={dev_metrics['loss']:.4f}")
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > args.patience:
                print(f"[roberta_plain] early stopping at epoch {epoch + 1}")
                break

    # 6) 用最好一轮的权重在 test 上评估，再测延迟/吞吐（与主流程同一函数）。
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, criterion)
    test_metrics.update(measure_latency(model, test_loader, device))
    test_metrics["best_epoch"] = best_epoch
    test_metrics["n_train"] = len(train_t["labels"])
    test_metrics["n_test"] = len(test_t["labels"])
    test_metrics["model_name"] = model_path

    print("\n" + "{:<16s}{:>9s}{:>9s}{:>9s}{:>9s}{:>13s}{:>12s}".format(
        "Method", "Acc", "Prec", "Rec", "F1", "Lat(ms)", "Throughput"))
    print("-" * 68)
    print("{:<16s}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>13.2f}{:>12.1f}".format(
        "roberta_plain", test_metrics["accuracy"], test_metrics["precision"],
        test_metrics["recall"], test_metrics["f1"],
        test_metrics["latency_ms"], test_metrics["throughput"]))

    # 7) 落盘：独立命名，不碰 metrics.json，也不影响任何现有产物。
    if not args.no_save:
        CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
        spec = {
            "name": "roberta_plain", "kind": "bert", "stage": 2,
            "model_name": model_path, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "dropout": args.dropout,
            "freeze_layers": 0, "patience": args.patience,
            "use_knowledge": False,
        }
        ckpt = CLASSIFIER_DIR / "roberta_plain.pt"
        torch.save({
            "spec": spec,
            "state_dict": model.state_dict(),
            "tokenizer_name": tokenizer.name_or_path,
            "max_len": args.max_len,
            "use_knowledge": False,
            "test_metrics": test_metrics,
        }, ckpt)
        metrics_path = CLASSIFIER_DIR / "roberta_plain_metrics.json"
        save_json(test_metrics, metrics_path)
        print(f"\nSaved: {ckpt}")
        print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
