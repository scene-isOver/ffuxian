"""固定阈值下扫描随机种子：同一 τ 用 30 个不同种子各训练一次 PPO-CIS。

目的：看 PPO-CIS 的结果对随机初始化有多敏感（方差有多大）。

做法：
    * 阈值固定（默认 0.75），只训练 PPO，不跑任何基线、不加载分类器；
    * 30 个种子由一个元种子（--seed_pool）随机抽样得到，可复现，启动时会打印；
    * 每个种子都用同一份 ppo_train 训练、ppo_dev 选 epoch、test 评估一次；
    * 不改动任何已有文件，只新增 artifacts/reports/ppo_seed_sweep.csv。

用法（在 ToxiCN_ex 目录下）：
    python -m ppo_cis.sweep_ppo_seed
    python -m ppo_cis.sweep_ppo_seed --threshold 0.75 --num_seeds 30
    python -m ppo_cis.sweep_ppo_seed --verbose   # 打印每轮训练日志
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import PPO_DIR, PPOConfig, REPORT_DIR, RewardConfig
from .eval_metrics import four_metrics, run_policy
from .ppo import evaluate_policy, train_ppo
from .sweep_ppo_threshold import build_env, load_scores
from .utils import get_device


# 表头带单位：比例类指标无量纲，Calls 为次/样本，Lat 为毫秒，Thr 为样本/秒。
HEADER = ("seed", "tau", "Acc", "MacP", "MacR", "MacF1", "OffF1", "NonF1",
          "Calls(/sample)", "Lat(ms)", "Thr(samples/s)",
          "ValAcc", "ValF1", "ValReward", "BestEpoch")


def mean_std(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-threshold seed sweep for PPO-CIS")
    parser.add_argument("--policy", default=str(PPO_DIR / "policy.pt"),
                        help="只用于读取阶段划分 / 奖励配置 / 训练超参等结构信息")
    parser.add_argument("--threshold", default=0.75, type=float,
                        help="固定置信阈值")
    parser.add_argument("--num_seeds", default=30, type=int)
    parser.add_argument("--seed_pool", default=2024, type=int,
                        help="用于随机抽样的元种子，保证 30 个种子可复现")
    parser.add_argument("--val_metric", default="reward",
                        choices=["reward", "f1", "accuracy"],
                        help="用 ppo_dev 上的哪个指标选择最佳 epoch")
    parser.add_argument("--ppo_device", default="cpu")
    parser.add_argument("--out", default=str(REPORT_DIR / "ppo_seed_sweep.csv"))
    parser.add_argument("--verbose", action="store_true",
                        help="打印每一轮训练日志")
    args = parser.parse_args()

    # 1) 结构信息
    ckpt = torch.load(args.policy, map_location="cpu", weights_only=False)
    stage1 = list(ckpt["stage1_idx"])
    stage2 = list(ckpt["stage2_idx"])
    priority = list(ckpt.get("stage2_priority", stage2))
    ppo_cfg = PPOConfig(**ckpt["ppo_config"])
    reward_cfg = RewardConfig(**ckpt["reward_config"])

    # ppo_update 直接在 CPU 张量上更新，传 cuda 会设备不一致，这里固定 CPU。
    device = get_device(args.ppo_device)
    if device.type != "cpu":
        print(f"[warn] PPO 更新目前只支持 CPU 张量，已忽略 --ppo_device {args.ppo_device}")
        device = torch.device("cpu")

    # 2) 打分缓存
    train = load_scores("ppo_train")
    dev = load_scores("ppo_dev")
    test = load_scores("test")
    if not (train["names"] == dev["names"] == test["names"]):
        raise ValueError("三个 split 的模型列不一致，请清空 scores 后重跑 train_ppo")
    names = train["names"]

    # 3) 随机种子：元种子抽样，可复现
    pool = random.Random(args.seed_pool)
    seeds = sorted(pool.sample(range(1, 1_000_000), args.num_seeds))

    print(f"模型: {names}")
    print(f"Stage 1: {[names[i] for i in stage1]}；"
          f"Stage 2: {[names[i] for i in stage2]}；"
          f"判定优先级: {[names[i] for i in priority]}")
    print(f"数据: ppo_train={len(train['labels'])} ppo_dev={len(dev['labels'])} "
          f"test={len(test['labels'])}")
    print(f"阈值 τ={args.threshold:.2f}；种子池 {args.seed_pool}；"
          f"随机种子（{len(seeds)} 个）: {seeds}\n")

    tau = args.threshold
    train_env = build_env(train, reward_cfg, stage1, stage2, priority,
                          ppo_cfg.max_steps, tau)
    val_env = build_env(dev, reward_cfg, stage1, stage2, priority,
                        ppo_cfg.max_steps, tau)
    test_env = build_env(test, reward_cfg, stage1, stage2, priority,
                         ppo_cfg.max_steps, tau)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key = {"reward": "val_reward", "f1": "val_f1",
           "accuracy": "val_accuracy"}[args.val_metric]

    rows: List[Dict[str, float]] = []
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(HEADER))
        f.flush()

        for k, seed in enumerate(seeds, 1):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            log = print if args.verbose else (lambda *a, **kw: None)
            policy, history = train_ppo(train_env, ppo_cfg, device, reward_cfg,
                                        log=log, val_env=val_env,
                                        val_metric=args.val_metric)

            val = evaluate_policy(policy, val_env, device)
            preds, labels, lat, calls = run_policy(policy, test_env, device)
            m = four_metrics(preds, labels)
            n = len(labels)
            best_epoch = max(history, key=lambda h: h.get(key, float("-inf")))["epoch"]

            row = {
                "seed": seed,
                "tau": tau,
                "accuracy": m["accuracy"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
                "off_f1": m["off_f1"],
                "non_f1": m["non_f1"],
                "calls": float(np.mean(calls)),
                "latency_ms": float(np.mean(lat) * 1000.0),
                "throughput": float(n / max(float(np.sum(lat)), 1e-9)),
                "val_acc": float(val["accuracy"]),
                "val_f1": float(val["f1"]),
                "val_reward": float(val["mean_reward"]),
                "best_epoch": int(best_epoch),
            }
            rows.append(row)

            writer.writerow([
                seed, f"{tau:.2f}", f"{row['accuracy']:.4f}",
                f"{row['macro_precision']:.4f}", f"{row['macro_recall']:.4f}",
                f"{row['macro_f1']:.4f}", f"{row['off_f1']:.4f}",
                f"{row['non_f1']:.4f}", f"{row['calls']:.2f}",
                f"{row['latency_ms']:.2f}", f"{row['throughput']:.1f}",
                f"{row['val_acc']:.4f}", f"{row['val_f1']:.4f}",
                f"{row['val_reward']:.4f}", f"{row['best_epoch']}",
            ])
            f.flush()

            print(f"[{k:>2d}/{len(seeds)}] seed={seed:<7d} "
                  f"test: acc={row['accuracy']:.4f} macF1={row['macro_f1']:.4f} "
                  f"calls={row['calls']:.2f} thr={row['throughput']:.1f}  |  "
                  f"dev: acc={row['val_acc']:.4f} reward={row['val_reward']:.4f} "
                  f"(epoch {row['best_epoch']})")

    # 4) 汇总：均值 ± 标准差、极值、最好/最差种子
    accs = [r["accuracy"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    calls = [r["calls"] for r in rows]
    thrs = [r["throughput"] for r in rows]
    best = max(rows, key=lambda r: r["accuracy"])
    worst = min(rows, key=lambda r: r["accuracy"])

    print("\n========== 30 个种子的统计（τ={:.2f}） ==========".format(tau))
    print(f"  test Acc      : {mean_std(accs)}   "
          f"[{min(accs):.4f}, {max(accs):.4f}]")
    print(f"  test Macro-F1 : {mean_std(f1s)}   "
          f"[{min(f1s):.4f}, {max(f1s):.4f}]")
    print(f"  Calls         : {mean_std(calls)}")
    print(f"  Throughput    : {mean_std(thrs)}")
    print(f"  最好种子: seed={best['seed']}  acc={best['accuracy']:.4f}  "
          f"macF1={best['macro_f1']:.4f}  calls={best['calls']:.2f}")
    print(f"  最差种子: seed={worst['seed']}  acc={worst['accuracy']:.4f}  "
          f"macF1={worst['macro_f1']:.4f}  calls={worst['calls']:.2f}")
    print(f"  Acc 极差: {max(accs) - min(accs):.4f}")
    print("\n  按 Acc 排序:", " ".join(f"{a:.4f}" for a in sorted(accs)))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
