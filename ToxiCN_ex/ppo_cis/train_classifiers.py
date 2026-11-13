"""Train the five PPO-CIS classifiers on ToxiCN (two throughput + three accuracy).

Stage 1 (high throughput): ``tiny_bert`` and ``distil_roberta``.
Stage 2 (high accuracy):   ``bert``, ``roberta_fusion`` and ``tke_bert``.

All five classifiers use the standard cross-entropy loss ``nn.CrossEntropyLoss``
over two logits (the binary cross-entropy used by the original ToxiCN_ex
``BCEWithLogitsLoss`` on a 2-dim target is the same objective in one-hot form).

Usage (from the ToxiCN_ex directory):
    python -m ppo_cis.train_classifiers
    python -m ppo_cis.train_classifiers --only tiny_bert distil_roberta --device cuda
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import (
    CLASSIFIER_DIR, DataConfig, default_classifier_specs,
    validate_classifier_models,
)
from .data import (
    EncodedDataset, build_splits, encode_split, get_tokenizer, print_split_stats,
)
from .models import build_classifier
from .utils import (
    amp_context, compute_metrics, forward_classifier, grad_scaler, get_device,
    load_json, measure_latency, save_json,
)


# ===================== 评估与单模型训练 =====================
# 在一个 loader 上跑前向，取 logits 的 argmax 作为预测，再统一交给 compute_metrics。
@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    losses, preds, labels = [], [], []
    for batch in loader:
        target = batch["labels"].to(device)
        with amp_context(device):
            logits = forward_classifier(model, batch, device)
            loss = criterion(logits, target)
        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        labels.extend(target.cpu().numpy().tolist())
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


# ===================== 单模型训练 =====================
# 训练单个分类器的完整流程，分六个阶段：
#   1) 已有权重且没加 --force 就跳过，避免重复训练；
#   2) 把 clf_train / clf_dev / test 三个划分分词并装成 DataLoader；
#   3) 建模型、AdamW 优化器、线性 warmup 调度器和损失函数；
#   4) 逐 epoch 跑训练主循环，每个 batch 做一次前向 + 反向 + 参数更新；
#   5) 每轮在 dev 上算 F1，只保留最好一轮的权重，连续无提升就早停；
#   6) 用最好权重在 test 上算最终指标、测延迟，写进 .pt 和 metrics.json。
def train_one(spec, splits, tokenizer, device, data_cfg: DataConfig,
              force: bool = False) -> Dict[str, float]:
    # ---- 1) 复用检查 ----
    # 权重已存在时直接读出旧的 test_metrics 返回，保证重复运行不会白跑一遍。
    ckpt_path = CLASSIFIER_DIR / f"{spec.name}.pt"
    if ckpt_path.exists() and not force:
        print(f"[skip] {spec.name}: {ckpt_path} already exists (use --force to retrain)")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("test_metrics", {})

    # ---- 2) 分词 + DataLoader ----
    # encode_split 内部有缓存，同一 tokenizer/max_len 组合只真正分词一次；
    # knowledge=True（只有 tke_bert）时额外生成 toxic_ids / stance_ids 两路知识 id。
    max_len = data_cfg.max_len
    knowledge = spec.use_knowledge
    train_t = encode_split(splits["clf_train"], tokenizer, "clf_train", max_len,
                           knowledge=knowledge)
    dev_t = encode_split(splits["clf_dev"], tokenizer, "clf_dev", max_len,
                         knowledge=knowledge)
    test_t = encode_split(splits["test"], tokenizer, "test", max_len,
                          knowledge=knowledge)

    train_loader = DataLoader(EncodedDataset(train_t), batch_size=spec.batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    dev_loader = DataLoader(EncodedDataset(dev_t), batch_size=spec.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(EncodedDataset(test_t), batch_size=spec.batch_size,
                             shuffle=False, num_workers=0)

    # ---- 3) 模型 / 优化器 / 调度器 / 损失 ----
    # 按 spec.kind 建对应架构（普通序列分类 / RoBERTa 融合 / TKE-BERT），再搬到目标设备。
    model = build_classifier(spec).to(device)
    # AdamW + 线性调度：前 10% 的步数把学习率从 0 线性拉到 spec.lr（warmup），
    # 之后线性衰减到 0，避免一开始就用大学习率把预训练权重冲坏。
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr,
                                  weight_decay=spec.weight_decay)
    # 总步数 = 每轮 batch 数 × 轮数；warmup 和线性衰减都以它为基准。
    total_steps = max(len(train_loader) * spec.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    # 两分类交叉熵（输入两个 logits 和 0/1 标签）。
    criterion = nn.CrossEntropyLoss()
    # 只有 CUDA 下才有 AMP 的 GradScaler；CPU 返回 None，后面走普通反向传播分支。
    scaler = grad_scaler(device)

    # 早停状态：记录 dev F1 最好的一轮，并在连续 patience 轮无提升时停止。
    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, -1, 0
    # ---- 4) 训练主循环 ----
    # 每个 epoch 都把 clf_train 完整扫一遍；epoch 结束在 dev 上评估一次。
    for epoch in range(spec.epochs):
        # 切到训练模式：启用 dropout 的随机丢弃（评估时会关掉）。
        model.train()
        running_loss = 0.0
        seen_batches = 0
        # 进度条：postfix 显示本 epoch 截至目前累计的平均 loss。
        pbar = tqdm(train_loader, desc=f"{spec.name} epoch {epoch + 1}/{spec.epochs}",
                    leave=False)
        # 每个 batch 走一遍：前向 → 算损失 → 反向 → 梯度裁剪 → 更新参数 → 推进学习率。
        for batch in pbar:
            target = batch["labels"].to(device)
            # 清空上一步残留的梯度；set_to_none=True 比写 0 更省显存也更快。
            optimizer.zero_grad(set_to_none=True)
            # 前向 + 算损失；amp_context 在 CUDA 下自动切 fp16 混合精度，CPU 下是空操作。
            with amp_context(device):
                logits = forward_classifier(model, batch, device)
                loss = criterion(logits, target)

            # 反向传播分两条路：有 AMP 走 scaler（先放大 loss 防 fp16 下溢），
            # 否则直接 loss.backward()。两条路最后都做梯度裁剪，防止梯度爆炸。
            if scaler is not None:
                scaler.scale(loss).backward()
                # 先把梯度还原回真实尺度，否则裁剪阈值 1.0 作用在被放大的梯度上没有意义。
                scaler.unscale_(optimizer)
                # 梯度范数超过 1.0 就整体等比缩小，保证每步更新幅度可控。
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                # 根据这一步是否出现 inf/nan 自动放大或缩小 scale，供下一个 batch 使用。
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            # 学习率按 batch（步）推进：warmup 和衰减都是按步数算的，不是按轮数。
            scheduler.step()

            running_loss += loss.item()
            seen_batches += 1
            pbar.set_postfix(loss=f"{running_loss / seen_batches:.4f}")

        # 一轮结束后在 dev 上完整评估一次（只前向、不更新参数），用 dev F1 决定是否保留。
        dev_metrics = evaluate(model, dev_loader, device, criterion)
        print(f"[{spec.name}] epoch {epoch + 1}: "
              f"dev_acc={dev_metrics['accuracy']:.4f} dev_f1={dev_metrics['f1']:.4f} "
              f"dev_loss={dev_metrics['loss']:.4f}")

        # 只有 dev F1 严格变好才保存快照（相等不算提升，避免在平台期反复刷新）。
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_epoch = epoch
            # 深拷贝一份到 CPU：detach 脱离计算图，clone 防止后续训练原地改掉这份快照。
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            # 没变好就累计一次；连续 patience+1 轮无提升则提前结束（省时间，也防过拟合）。
            bad_epochs += 1
            if bad_epochs > spec.patience:
                print(f"[{spec.name}] early stopping at epoch {epoch + 1}")
                break

    # ---- 5) 恢复最好一轮的权重，在 test 上做最终评估 ----
    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试集指标（Acc/Prec/Rec/F1 + loss），再测单样本延迟与吞吐。
    test_metrics = evaluate(model, test_loader, device, criterion)
    latency = measure_latency(model, test_loader, device)
    test_metrics.update(latency)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["n_train"] = len(train_t["labels"])
    test_metrics["n_test"] = len(test_t["labels"])
    test_metrics["model_name"] = spec.model_name

    # ---- 6) 落盘 ----
    # .pt 里同时存 spec / 权重 / tokenizer 名 / max_len / use_knowledge / test_metrics，
    # 这样 PPO 阶段（load_classifier、score_split）能独立还原模型，并直接读延迟值。
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "spec": asdict(spec),
        "state_dict": model.state_dict(),
        "tokenizer_name": tokenizer.name_or_path,
        "max_len": max_len,
        "use_knowledge": knowledge,
        "test_metrics": test_metrics,
    }, ckpt_path)

    # metrics.json 是五个模型共用的汇总表：读出来 → 更新自己这一条 → 整体写回。
    metrics_file = CLASSIFIER_DIR / "metrics.json"
    all_metrics = load_json(metrics_file) if metrics_file.exists() else {}
    all_metrics[spec.name] = test_metrics
    all_metrics["_meta"] = {
        "loss": "CrossEntropyLoss",
        "layout": "two-stage: tiny_bert + distil_roberta -> bert + roberta_fusion + tke_bert",
        "max_len": max_len,
    }
    save_json(all_metrics, metrics_file)

    # 终端打印最终结果：测试集 Acc / F1、单样本延迟、吞吐和权重保存位置。
    print(f"[{spec.name}] TEST acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
          f"latency={test_metrics['latency_ms']:.2f}ms "
          f"throughput={test_metrics['throughput']:.1f}/s -> {ckpt_path}")
    return test_metrics


# ===================== 主流程 =====================
# 解析参数 → 切分数据 → 生成五个 spec（可 --only 指定子集）→ 校验模型 →
# 共享 tokenizer → 逐个训练并落盘。
def main():
    parser = argparse.ArgumentParser(description="Train the five PPO-CIS classifiers")
    parser.add_argument("--bert_model", default=None, help="BERT model name or local path")
    parser.add_argument("--roberta_model", default=None, help="RoBERTa model name or local path")
    parser.add_argument("--tiny_bert_model", default=None,
                        help="stage-1 BERT-Tiny model name or local path")
    parser.add_argument("--distil_roberta_model", default=None,
                        help="stage-1 distilled RoBERTa model name or local path")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--max_len", default=96, type=int)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # 1) 选设备、切分数据并打印四个划分的样本分布。
    device = get_device(args.device)
    print(f"Device: {device}")
    data_cfg = DataConfig(max_len=args.max_len)
    splits = build_splits(data_cfg)
    print_split_stats(splits)

    # 2) 生成五个 spec（--only 时只保留指定模型），并统一校验预训练权重是否齐全。
    specs = default_classifier_specs(args.bert_model, args.roberta_model,
                                     args.tiny_bert_model, args.distil_roberta_model)
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]
    validate_classifier_models(specs)

    # 3) tokenizer 按骨干模型共享（bert 与 tke_bert 同一个），只加载一次，然后逐个训练。
    tokenizers = {}
    for spec in specs:
        tokenizer_name = spec.model_name
        if tokenizer_name not in tokenizers:
            tokenizers[tokenizer_name] = get_tokenizer(tokenizer_name)
        train_one(spec, splits, tokenizers[tokenizer_name], device, data_cfg, args.force)


if __name__ == "__main__":
    main()
