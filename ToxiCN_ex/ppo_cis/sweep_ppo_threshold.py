"""按阈值扫描 PPO-CIS：每个 τ 重新训练一次策略，结果写入一份新的 CSV。

目的：回答“PPO-CIS 在哪个置信阈值下效果最好”。

做法：
    * 不加载任何分类器，只读 artifacts/scores/{ppo_train,ppo_dev,test}.npz；
    * 动作掩码依赖 τ，所以每个 τ 都要重新训练策略（同一随机种子，保证可比）；
    * 训练只用 ppo_train，选 epoch 只用 ppo_dev，test 每档只评估一次；
    * 不改动任何已有文件，只新增 artifacts/reports/ppo_threshold_sweep.csv。

用法（在 ToxiCN_ex 目录下）：
    python -m ppo_cis.sweep_ppo_threshold
    python -m ppo_cis.sweep_ppo_threshold --low 0.55 --high 0.80 --step 0.01
    python -m ppo_cis.sweep_ppo_threshold --verbose   # 打印每轮训练日志

注意：CSV 里同时记录 ppo_dev 与 test 的指标。选 τ 请依据 ValAcc/ValReward，
     若按 test 的 Acc 挑最大值，就属于用测试集调参。
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import PPO_DIR, PPOConfig, REPORT_DIR, RewardConfig, SCORE_DIR
from .eval_metrics import four_metrics, run_policy
from .pipeline import PPOEnv
from .ppo import evaluate_policy, train_ppo
from .utils import get_device


# 表头带单位：比例类指标无量纲，Calls 为次/样本，Lat 为毫秒，Thr 为样本/秒。
HEADER = ("tau", "Acc", "MacP", "MacR", "MacF1", "OffF1", "NonF1",
          "Calls(/sample)", "Lat(ms)", "Thr(samples/s)",
          "ValAcc", "ValF1", "ValReward", "BestEpoch")


def load_scores(split: str) -> Dict[str, np.ndarray]:
    """直接读打分缓存，不触发任何模型推理。"""
    path = SCORE_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少打分缓存 {path}；先跑一次 train_ppo 生成 ppo_train / ppo_dev / test")
    data = np.load(path, allow_pickle=True)
    return {"probs": data["probs"], "labels": data["labels"],
            "latencies": data["latencies"],
            "names": [str(x) for x in data["names"]]}


def tau_grid(low: float, high: float, step: float) -> List[float]:
    """生成 [low, high] 上步长为 step 的阈值网格。"""
    if step <= 0:
        raise ValueError("--step 必须为正数")
    values, v = [], low
    while v <= high + 1e-9:
        values.append(round(float(v), 4))
        v += step
    return values


def build_env(scores: Dict[str, np.ndarray], reward_cfg: RewardConfig,
              stage1: List[int], stage2: List[int], priority: List[int],
              max_steps: int, tau: float) -> PPOEnv:
    return PPOEnv(scores["probs"], scores["labels"], scores["latencies"],
                  reward_cfg, stage1_idx=stage1, stage2_idx=stage2,
                  max_steps=max_steps, threshold=tau,
                  stage2_priority=priority)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the PPO-CIS threshold, retraining the policy for each tau")
    parser.add_argument("--policy", default=str(PPO_DIR / "policy.pt"),
                        help="只用于读取阶段划分 / 奖励配置 / 训练超参等结构信息")
    parser.add_argument("--low", default=0.55, type=float)
    parser.add_argument("--high", default=0.80, type=float)
    parser.add_argument("--step", default=0.01, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val_metric", default="reward",
                        choices=["reward", "f1", "accuracy"],
                        help="用 ppo_dev 上的哪个指标选择最佳 epoch")
    parser.add_argument("--ppo_device", default="cpu")
    parser.add_argument("--out", default=str(REPORT_DIR / "ppo_threshold_sweep.csv"))
    parser.add_argument("--verbose", action="store_true",
                        help="打印每一轮训练日志（默认只打印每个 tau 的汇总行）")
    args = parser.parse_args()

    # 1) 结构信息：阶段划分、Stage-2 优先级、PPO 超参、奖励参数
    ckpt = torch.load(args.policy, map_location="cpu", weights_only=False)
    stage1 = list(ckpt["stage1_idx"])
    stage2 = list(ckpt["stage2_idx"])
    priority = list(ckpt.get("stage2_priority", stage2))
    ppo_cfg = PPOConfig(**ckpt["ppo_config"])
    reward_cfg = RewardConfig(**ckpt["reward_config"])
    device = get_device(args.ppo_device)

    # 2) 打分缓存
    train = load_scores("ppo_train")
    dev = load_scores("ppo_dev")
    test = load_scores("test")
    if not (train["names"] == dev["names"] == test["names"]):
        raise ValueError("三个 split 的模型列不一致，请清空 scores 后重跑 train_ppo")
    names = train["names"]

    grid = tau_grid(args.low, args.high, args.step)
    print(f"模型: {names}")
    print(f"Stage 1: {[names[i] for i in stage1]}；"
          f"Stage 2: {[names[i] for i in stage2]}；"
          f"判定优先级: {[names[i] for i in priority]}")
    print(f"数据: ppo_train={len(train['labels'])} ppo_dev={len(dev['labels'])} "
          f"test={len(test['labels'])}")
    print(f"阈值网格: {len(grid)} 档（{grid[0]:.2f} ~ {grid[-1]:.2f}，步长 {args.step:g}）\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key = {"reward": "val_reward", "f1": "val_f1",
           "accuracy": "val_accuracy"}[args.val_metric]

    rows: List[Dict[str, float]] = []
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(HEADER))
        f.flush()

        for k, tau in enumerate(grid, 1):
            # 同一随机种子，保证各 τ 之间的差异只来自阈值本身
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            train_env = build_env(train, reward_cfg, stage1, stage2, priority,
                                  ppo_cfg.max_steps, tau)
            val_env = build_env(dev, reward_cfg, stage1, stage2, priority,
                                ppo_cfg.max_steps, tau)
            test_env = build_env(test, reward_cfg, stage1, stage2, priority,
                                 ppo_cfg.max_steps, tau)

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
                f"{tau:.2f}", f"{row['accuracy']:.4f}",
                f"{row['macro_precision']:.4f}", f"{row['macro_recall']:.4f}",
                f"{row['macro_f1']:.4f}", f"{row['off_f1']:.4f}",
                f"{row['non_f1']:.4f}", f"{row['calls']:.2f}",
                f"{row['latency_ms']:.2f}", f"{row['throughput']:.1f}",
                f"{row['val_acc']:.4f}", f"{row['val_f1']:.4f}",
                f"{row['val_reward']:.4f}", f"{row['best_epoch']}",
            ])
            f.flush()

            print(f"[{k:>2d}/{len(grid)}] tau={tau:.2f}  "
                  f"test: acc={row['accuracy']:.4f} macF1={row['macro_f1']:.4f} "
                  f"calls={row['calls']:.2f} thr={row['throughput']:.1f}  |  "
                  f"dev: acc={row['val_acc']:.4f} reward={row['val_reward']:.4f} "
                  f"(epoch {row['best_epoch']})")

    # 3) 汇总：按 dev 指标排序（用于选 τ），test 指标一并列出
    dev_best = max(rows, key=lambda r: r["val_acc"])
    test_best = max(rows, key=lambda r: r["accuracy"])
    print("\n按 ppo_dev 准确率选出的最优阈值：")
    print(f"  tau={dev_best['tau']:.2f}  dev_acc={dev_best['val_acc']:.4f}  "
          f"test_acc={dev_best['accuracy']:.4f}  macF1={dev_best['macro_f1']:.4f}  "
          f"calls={dev_best['calls']:.2f}  thr={dev_best['throughput']:.1f}")
    print("按 test 准确率挑出的最大值（仅供参考，不能用于选型）：")
    print(f"  tau={test_best['tau']:.2f}  test_acc={test_best['accuracy']:.4f}  "
          f"macF1={test_best['macro_f1']:.4f}  calls={test_best['calls']:.2f}  "
          f"thr={test_best['throughput']:.1f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
