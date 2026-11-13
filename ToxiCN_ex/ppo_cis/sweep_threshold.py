"""在验证集上扫 PPO-CIS 的置信阈值 τ，再用同一个策略报告测试集结果。

作用：把 τ 的选择放到不参与 PPO 训练的 split（默认 clf_dev）上做，避免用
测试集挑超参造成的数据泄漏。脚本只读 ``policy.pt`` 和 ``scores/<split>.npz``，
不加载分类器、不重训 PPO。

用法（在 ToxiCN_ex 目录下，先跑过 train_ppo）：
    python -m ppo_cis.sweep_threshold
    python -m ppo_cis.sweep_threshold --grid 0.55:0.85:0.05
    python -m ppo_cis.sweep_threshold --select_split clf_dev --report_split test
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import PPO_DIR, RewardConfig, SCORE_DIR
from .pipeline import PPOEnv
from .ppo import ActorCritic, evaluate_policy
from .utils import get_device

METRIC_KEYS = ("accuracy", "precision", "recall", "f1",
               "avg_classifiers", "avg_latency_ms", "throughput")


def load_scores(split: str) -> Dict[str, np.ndarray]:
    """直接读打分缓存，不触发任何模型推理。"""
    path = SCORE_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少打分缓存 {path}。先跑一次 train_ppo（会生成 ppo_train/test），"
            f"或运行 fit_temperature 生成 clf_dev。"
        )
    data = np.load(path, allow_pickle=True)
    return {"probs": data["probs"], "labels": data["labels"],
            "latencies": data["latencies"]}


def parse_grid(text: str) -> List[float]:
    """解析 ``low:high:step`` 形式的阈值网格。"""
    low, high, step = (float(x) for x in text.split(":"))
    values = []
    v = low
    while v <= high + 1e-9:
        values.append(round(v, 4))
        v += step
    return values


def build_env(scores: Dict[str, np.ndarray], reward_cfg: RewardConfig,
              threshold: float, ckpt: Dict) -> PPOEnv:
    s1 = ckpt["stage1_idx"]
    s2 = ckpt["stage2_idx"]
    return PPOEnv(scores["probs"], scores["labels"], scores["latencies"],
                  reward_cfg, stage1_idx=s1, stage2_idx=s2,
                  max_steps=ckpt["ppo_config"].get("max_steps", 8),
                  threshold=threshold,
                  stage2_priority=ckpt.get("stage2_priority") or s2)


def print_header(title: str) -> None:
    print("\n" + title)
    print("{:>6s}{:>9s}{:>9s}{:>9s}{:>9s}{:>8s}{:>11s}{:>12s}".format(
        "tau", "Acc", "Prec", "Rec", "F1", "Calls", "Lat(ms)", "Thr"))


def print_row(label: str, metrics: Dict[str, float]) -> None:
    print("{:>6s}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>8.2f}{:>11.2f}{:>12.1f}".format(
        label, metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["avg_classifiers"], metrics["avg_latency_ms"],
        metrics["throughput"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the PPO-CIS confidence threshold on a validation split")
    parser.add_argument("--policy", default=str(PPO_DIR / "policy.pt"))
    parser.add_argument("--select_split", default="clf_dev",
                        help="用来挑 τ 的 split（不应是 test）")
    parser.add_argument("--report_split", default="test",
                        help="最终报告结果的 split")
    parser.add_argument("--grid", default="0.55:0.85:0.05",
                        help="阈值网格，格式 low:high:step")
    parser.add_argument("--select_metric", default="accuracy",
                        choices=("accuracy", "f1"),
                        help="用来挑最优 τ 的指标，默认 accuracy")
    parser.add_argument("--min_throughput", default=0.0, type=float,
                        help="吞吐下限（样本/秒）；只在这些行里挑最优 τ，0 表示不限制")
    parser.add_argument("--ppo_device", default="cpu")
    args = parser.parse_args()

    device = get_device(args.ppo_device)
    ckpt = torch.load(args.policy, map_location="cpu", weights_only=False)

    policy = ActorCritic(ckpt["state_dim"], ckpt["action_dim"], ckpt["hidden_dim"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(device).eval()

    reward_cfg = RewardConfig(**ckpt["reward_config"])
    default_tau = float(ckpt.get("threshold", 0.70))

    select_scores = load_scores(args.select_split)
    report_scores = load_scores(args.report_split)

    # 在验证集上扫 τ，按 --select_metric 选最优（默认准确率）
    metric = args.select_metric
    grid = parse_grid(args.grid)
    print_header(f"阈值扫描（选择集 {args.select_split}, "
                 f"N={len(select_scores['labels'])}）")
    rows = []
    for tau in grid:
        env = build_env(select_scores, reward_cfg, tau, ckpt)
        metrics = evaluate_policy(policy, env, device)
        print_row("{:.2f}".format(tau), metrics)
        rows.append((tau, metrics))

    # 只在满足吞吐下限的行里挑；全部不满足时退回整张表
    candidates = [item for item in rows
                  if args.min_throughput <= 0.0
                  or item[1]["throughput"] >= args.min_throughput]
    if candidates:
        tau_best, best_metrics = max(candidates, key=lambda item: item[1][metric])
        if args.min_throughput > 0.0:
            print("\n吞吐下限 {:.1f}：{} / {} 个 τ 满足".format(
                args.min_throughput, len(candidates), len(rows)))
    else:
        tau_best, best_metrics = max(rows, key=lambda item: item[1][metric])
        print("\n警告：没有 τ 满足吞吐下限 {:.1f}，改为在全部网格中挑最优".format(
            args.min_throughput))

    print("最优 τ = {:.2f}（{}(验证) {} = {:.4f}, throughput = {:.1f}）".format(
        tau_best, args.select_split, metric, best_metrics[metric],
        best_metrics["throughput"]))

    # 用同一个策略在报告集上对比：checkpoint 里的默认 τ vs 验证集选出的 τ
    print_header(f"报告集 {args.report_split}（N={len(report_scores['labels'])}）")
    env = build_env(report_scores, reward_cfg, default_tau, ckpt)
    print_row("{:.2f}*".format(default_tau), evaluate_policy(policy, env, device))
    env = build_env(report_scores, reward_cfg, tau_best, ckpt)
    print_row("{:.2f}".format(tau_best), evaluate_policy(policy, env, device))
    print("\n* 为 checkpoint 中保存的默认 τ")

    final = evaluate_policy(
        policy, build_env(report_scores, reward_cfg, tau_best, ckpt), device)
    print("\n选定 τ={:.2f} 在 {} 上的完整指标：".format(tau_best, args.report_split))
    for key in METRIC_KEYS:
        print("  {:<16s}{:.4f}".format(key, final[key]))
    print("\n要把它写进正式报告，重跑 train_ppo 时加 --threshold {:.2f}".format(tau_best))


if __name__ == "__main__":
    main()
