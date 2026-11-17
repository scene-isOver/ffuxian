"""把纯 RoBERTa 的分数补进 artifacts/scores/*.npz（独立脚本，可整份删除）。

为什么需要它
------------
级联池已把 ``roberta_fusion`` 换成 ``roberta_plain``，但 ``--force_scores``
会为全部 5 个分类器重算，而 artifacts/classifiers 里没有
``tiny_bert.pt`` / ``distil_roberta.pt``，一跑就崩。
本脚本只做一件事：读出旧 npz 的其余 4 列，用 ``roberta_plain.pt`` 补算
``roberta_fusion`` 那一列并原地替换，列顺序与
``config.default_classifier_specs`` 完全一致（tiny, distil, bert, roberta_plain, tke）。

安全性
------
* 覆盖前断言旧 npz 的 labels 与当前切分逐条相同，防止样本错位；
* 旧文件先备份成 ``scores/backup_<split>_with_roberta_fusion.npz``；
* 只读 .pt，不训练、不修改任何权重。

用法（在 ToxiCN_ex 目录下）
----------------------------
    python -m ppo_cis.append_scores
    python -m ppo_cis.append_scores --splits clf_dev ppo_train test
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CLASSIFIER_DIR, ClassifierSpec, DataConfig, SCORE_DIR
from .data import EncodedDataset, build_splits, encode_split, get_tokenizer
from .models import build_classifier
from .utils import (
    compute_metrics, forward_classifier, get_device, load_json, save_json,
)

TARGET = "roberta_plain"
REPLACED = "roberta_fusion"


# ===================== 加载纯 RoBERTa =====================
# roberta_plain.pt 存的是 HF 原始权重（无 encoder. 前缀），
# 这里补上前缀后装进 TransformerClassifier，与 pipeline.load_classifier 口径一致。
def load_plain_model(path: str, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = ClassifierSpec(**ckpt["spec"])
    model = build_classifier(spec)
    state = ckpt["state_dict"]
    if spec.kind == "bert" and not any(k.startswith("encoder.") for k in state):
        state = {f"encoder.{k}": v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt


@torch.no_grad()
def score_records(model, ckpt, records, split_name: str, device) -> np.ndarray:
    tokenizer = get_tokenizer(ckpt["tokenizer_name"])
    tensors = encode_split(records, tokenizer, split_name, ckpt["max_len"],
                           knowledge=ckpt.get("use_knowledge", False))
    loader = DataLoader(EncodedDataset(tensors), batch_size=128,
                        shuffle=False, num_workers=0)
    probs = []
    for batch in loader:
        logits = forward_classifier(model, batch, device)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(
        description="Append roberta_plain scores to cached npz files")
    parser.add_argument("--model_ckpt",
                        default=str(CLASSIFIER_DIR / "roberta_plain.pt"))
    parser.add_argument("--splits", nargs="*",
                        default=["clf_dev", "ppo_train", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=42, type=int,
                        help="必须与生成旧 npz 时的数据切分种子一致")
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    if not Path(args.model_ckpt).exists():
        raise SystemExit(
            f"{args.model_ckpt} 不存在。先训练纯 RoBERTa：\n"
            "  python -m ppo_cis.roberta_baseline --model_name <本地模型目录>")

    device = get_device(args.device)
    model, ckpt = load_plain_model(args.model_ckpt, device)
    print(f"Device: {device}; checkpoint: {args.model_ckpt}")

    # 复现与旧 npz 相同的数据切分（seed 必须一致，否则下面的 labels 断言会失败）。
    splits = build_splits(DataConfig(max_len=ckpt["max_len"], seed=args.seed))

    for split in args.splits:
        path = SCORE_DIR / f"{split}.npz"
        if not path.exists():
            raise SystemExit(f"{path} 不存在；先跑一次 train_ppo 生成打分缓存")
        old = np.load(path, allow_pickle=True)
        names = [str(n) for n in old["names"]]
        if TARGET in names:
            print(f"[skip] {split}: 已包含 {TARGET}")
            continue
        if REPLACED not in names:
            raise SystemExit(f"{split}: 找不到 {REPLACED} 列，names={names}")

        records = splits[split]
        labels = np.array([row["label"] for row in records], dtype=np.int64)
        if not np.array_equal(labels, old["labels"]):
            raise SystemExit(
                f"{split}: labels 与旧 npz 不一致"
                f"（{len(labels)} vs {len(old['labels'])}）。"
                f"旧 npz 可能由别的 seed 生成，请先用同一 --seed 重跑。")

        probs = score_records(model, ckpt, records, split, device)
        col = names.index(REPLACED)
        new_probs = old["probs"].copy()
        new_probs[:, col] = probs
        latencies = old["latencies"].copy()
        latencies[col] = float(ckpt["test_metrics"]["latency_sec"])
        names[col] = TARGET

        if not args.no_backup:
            backup = SCORE_DIR / f"backup_{split}_with_{REPLACED}.npz"
            if not backup.exists():
                shutil.copy2(path, backup)
                print(f"  backup -> {backup.name}")

        np.savez(path, probs=new_probs, labels=old["labels"],
                 latencies=latencies, names=np.array(names, dtype=object))
        m = compute_metrics((probs >= 0.5).astype(int), old["labels"])
        print(f"[{split}] n={len(probs)} mean_p={probs.mean():.4f} "
              f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} -> {path.name}")

    # 把 roberta_plain 的 test 指标合进 metrics.json，报告的单模型表才看得到它。
    metrics_file = CLASSIFIER_DIR / "metrics.json"
    all_metrics = load_json(metrics_file) if metrics_file.exists() else {}
    if all_metrics.get(TARGET) != ckpt["test_metrics"]:
        all_metrics[TARGET] = ckpt["test_metrics"]
        save_json(all_metrics, metrics_file)
        print(f"metrics.json <- {TARGET}")


if __name__ == "__main__":
    main()
