"""Score with frozen classifiers, train the two-stage PPO-Clip policy, compare.

The output directory ``ppo_cis/artifacts/reports`` receives, after a run:
    ppo_cis_results.csv    machine-readable comparison table
    ppo_cis_results.md     Markdown table (readable result file)
    ppo_cis_results.json   full payload incl. PPO training history
    ppo_cis_comparison.png F1 / latency / call-count bar chart

Usage (from the ToxiCN_ex directory, after train_classifiers):
    python -m ppo_cis.train_ppo
    python -m ppo_cis.train_ppo --ppo_device cpu --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .config import (
    CLASSIFIER_DIR, DataConfig, PPO_DIR, PPOConfig, REPORT_DIR, RewardConfig,
    StaticConfig, default_classifier_specs, stage_indices,
    validate_classifier_models,
)
from .data import build_splits, print_split_stats
from .pipeline import PPOEnv, score_split, static_baselines
from .ppo import evaluate_policy, train_ppo
from .plot_results import make_plots
from .utils import get_device, load_json, save_json

METHOD_COLUMNS = ["accuracy", "precision", "recall", "f1",
                  "avg_classifiers", "avg_latency_ms", "throughput"]


# 打印对比表：方法 / Acc / Prec / Rec / F1 / 调用数 / 延迟 / 吞吐。
# 终端宽度不够时数字会折行，完整表格见 ppo_cis_results.md。
def print_table(results: Dict[str, Dict[str, float]]) -> None:
    header = "{:<24s}{:>9s}{:>9s}{:>9s}{:>9s}{:>8s}{:>12s}{:>12s}".format(
        "Method", "Acc", "Prec", "Rec", "F1", "Calls", "Lat(ms)", "Throughput")
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print("{:<24s}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>8.2f}{:>12.2f}{:>12.1f}".format(
            name, metrics.get("accuracy", 0), metrics.get("precision", 0),
            metrics.get("recall", 0), metrics.get("f1", 0),
            metrics.get("avg_classifiers", 0), metrics.get("avg_latency_ms", 0),
            metrics.get("throughput", 0)))


# 生成 Markdown 版结果表：上半部分是 5 个分类器明细，下半部分是测试集方法对比。
def write_markdown(results: Dict[str, Dict[str, float]], classifier_metrics: Dict,
                   specs, path) -> None:
    stage_by_name = {spec.name: spec.stage for spec in specs}
    stage1_names = ", ".join(f"`{s.name}`" for s in specs if s.stage == 1)
    stage2_names = ", ".join(f"`{s.name}`" for s in specs if s.stage == 2)
    lines = [
        "# PPO-CIS two-stage cascade — results",
        "",
        f"Stage 1 (throughput): {stage1_names}. "
        f"Stage 2 (accuracy): {stage2_names}.",
        "",
        "## Classifiers",
        "",
        "| Classifier | Stage | Acc | F1 | Latency (ms) | Throughput |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in classifier_metrics.items():
        if name.startswith("_") or not isinstance(metrics, dict):
            continue
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.2f} | {:.1f} |".format(
            name, stage_by_name.get(name, metrics.get("stage", "")),
            metrics.get("accuracy", 0), metrics.get("f1", 0),
            metrics.get("latency_ms", 0), metrics.get("throughput", 0)))

    lines += [
        "",
        "## Test-set comparison",
        "",
        "| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results.items():
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:.2f} | {:.1f} |".format(
            name, metrics.get("accuracy", 0), metrics.get("precision", 0),
            metrics.get("recall", 0), metrics.get("f1", 0),
            metrics.get("avg_classifiers", 0), metrics.get("avg_latency_ms", 0),
            metrics.get("throughput", 0)))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ===================== 主流程 =====================
# 数据切分 → 冻结打分 → 训练 PPO → 测试评估 → 静态基线对比 → 落盘 CSV/MD/JSON/PNG
def main():
    parser = argparse.ArgumentParser(
        description="Train PPO-CIS on the two-stage ToxiCN cascade")
    parser.add_argument("--bert_model", default=None)
    parser.add_argument("--roberta_model", default=None)
    parser.add_argument("--tiny_bert_model", default=None)
    parser.add_argument("--distil_roberta_model", default=None)
    parser.add_argument("--device", default="auto", help="device for classifier inference")
    parser.add_argument("--ppo_device", default="cpu", help="PPO MLP is tiny; cpu is enough")
    parser.add_argument("--max_len", default=96, type=int)
    parser.add_argument("--train_json", default=None,
                        help="训练集 JSON 路径（默认 ToxiCN/data/train.json）")
    parser.add_argument("--test_json", default=None,
                        help="测试集 JSON 路径（默认 ToxiCN/data/test.json）")
    parser.add_argument("--val_metric", default="reward",
                        choices=["reward", "f1", "accuracy"],
                        help="用 ppo_dev 上的哪个指标选择最佳 epoch")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--update_epochs", default=10, type=int)
    parser.add_argument("--clip", default=0.2, type=float)
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="learning rate of the PPO MLP")
    parser.add_argument("--aux_coef", default=0.10, type=float,
                        help="weight of the auxiliary cross-entropy loss")
    parser.add_argument("--entropy_coef", default=0.05, type=float,
                        help="entropy bonus; raise it if the policy stops exploring early exits")
    parser.add_argument("--explore_decide", default=0.05, type=float,
                        help="fraction of rollouts forced to a uniformly random legal decision")
    parser.add_argument("--time_budget", default=0.0015, type=float,
                        help="u in the time reward, in seconds (1.5 ms reference budget)")
    parser.add_argument("--time_theta", default=0.5, type=float,
                        help="theta_1: weight of the per-sample latency penalty")
    parser.add_argument("--throughput_scale", default=0.0, type=float,
                        help="weight of the Eq.(6) batch throughput bonus (report-only when 0)")
    parser.add_argument("--confidence_threshold", default=0.70, type=float,
                        help="threshold of the fixed cascade baseline")
    parser.add_argument("--threshold", default=0.70, type=float,
                        help="confidence gate of the PPO action mask")
    parser.add_argument("--stage2_priority", default="roberta_plain,bert,tke_bert",
                        help="Stage 2 判定优先级：第一个是主判定模型，其后是灰区参考模型")
    parser.add_argument("--force_scores", action="store_true")
    parser.add_argument("--no_plot", action="store_true", help="skip PNG generation")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    # 固定随机种子：保证策略网络初始化、动作采样、探索抽样和 batch 打乱都可复现。
    # 数据切分的随机性由 DataConfig(seed=...) 单独控制，两者用同一个 --seed。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    ppo_device = get_device(args.ppo_device)
    print(f"Classifier device: {device}; PPO device: {ppo_device}")

    # 1) 清洗并切分数据：clf_train / clf_dev / ppo_train / ppo_dev / test
    data_cfg = DataConfig(max_len=args.max_len, seed=args.seed,
                          **({"train_json": Path(args.train_json)}
                             if args.train_json else {}),
                          **({"test_json": Path(args.test_json)}
                             if args.test_json else {}))
    splits = build_splits(data_cfg)
    print_split_stats(splits)

    specs = default_classifier_specs(args.bert_model, args.roberta_model,
                                     args.tiny_bert_model, args.distil_roberta_model)
    validate_classifier_models(specs)
    s1, s2 = stage_indices(specs)
    print(f"Stage 1: {[specs[i].name for i in s1]}")
    print(f"Stage 2: {[specs[i].name for i in s2]}")

    # Stage 2 判定优先级：按名字映射成列下标，未列出的 Stage-2 模型排在最后。
    name_to_idx = {spec.name: i for i, spec in enumerate(specs)}
    wanted = [n.strip() for n in args.stage2_priority.split(",") if n.strip()]
    stage2_priority = [name_to_idx[n] for n in wanted if n in name_to_idx]
    stage2_priority += [i for i in s2 if i not in stage2_priority]
    print(f"Stage 2 priority: {[specs[i].name for i in stage2_priority]}")

    # 2) 用冻结分类器给 ppo_train / ppo_dev / test 打分（有缓存则复用，--force_scores 强制重算）
    ppo_scores = score_split(specs, splits["ppo_train"], "ppo_train", device,
                             args.force_scores)
    ppo_dev_scores = score_split(specs, splits["ppo_dev"], "ppo_dev", device,
                                 args.force_scores)
    test_scores = score_split(specs, splits["test"], "test", device,
                              args.force_scores)

    # 3) 组装配置：PPO 超参、奖励参数、静态基线阈值
    ppo_cfg = PPOConfig(epochs=args.epochs, batch_size=args.batch_size,
                        update_epochs=args.update_epochs, clip=args.clip, lr=args.lr,
                        aux_coef=args.aux_coef, entropy_coef=args.entropy_coef,
                        explore_decide=args.explore_decide, seed=args.seed)
    reward_cfg = RewardConfig(time_budget=args.time_budget,
                              time_theta=args.time_theta,
                              throughput_scale=args.throughput_scale)
    static_cfg = StaticConfig(confidence_threshold=args.confidence_threshold)

    train_env = PPOEnv(ppo_scores["probs"], ppo_scores["labels"],
                       ppo_scores["latencies"], reward_cfg,
                       stage1_idx=s1, stage2_idx=s2,
                       max_steps=ppo_cfg.max_steps, threshold=args.threshold,
                       stage2_priority=stage2_priority)
    # 策略验证集：不参与梯度更新，只用于按平均奖励选择 epoch。
    val_env = PPOEnv(ppo_dev_scores["probs"], ppo_dev_scores["labels"],
                     ppo_dev_scores["latencies"], reward_cfg,
                     stage1_idx=s1, stage2_idx=s2,
                     max_steps=ppo_cfg.max_steps, threshold=args.threshold,
                     stage2_priority=stage2_priority)
    # 4) 训练 PPO 策略（5 个分类器全程冻结）
    policy, history = train_ppo(train_env, ppo_cfg, ppo_device, reward_cfg,
                                val_env=val_env, val_metric=args.val_metric)
    ppo_val_metrics = evaluate_policy(policy, val_env, ppo_device)
    print("PPO validation (best epoch): acc={:.4f} f1={:.4f} calls={:.2f} "
          "lat={:.2f}ms thr={:.1f}".format(
              ppo_val_metrics["accuracy"], ppo_val_metrics["f1"],
              ppo_val_metrics["avg_classifiers"], ppo_val_metrics["avg_latency_ms"],
              ppo_val_metrics["throughput"]))

    PPO_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "state_dim": train_env.obs_dim,
        "action_dim": train_env.action_dim,
        "hidden_dim": ppo_cfg.hidden_dim,
        "specs": [asdict(spec) for spec in specs],
        "stage1_idx": s1,
        "stage2_idx": s2,
        "ppo_config": asdict(ppo_cfg),
        "reward_config": asdict(reward_cfg),
        "threshold": args.threshold,
        "stage2_priority": stage2_priority,
        "history": history,
    }, PPO_DIR / "policy.pt")

    test_env = PPOEnv(test_scores["probs"], test_scores["labels"],
                      test_scores["latencies"], reward_cfg,
                      stage1_idx=s1, stage2_idx=s2,
                      max_steps=ppo_cfg.max_steps, threshold=args.threshold,
                      stage2_priority=stage2_priority)
    # 5) 在测试集上用贪心策略评估 PPO-CIS
    ppo_metrics = evaluate_policy(policy, test_env, ppo_device)

    # 6) 计算全部静态基线，并把 PPO-CIS 一起放进对比表
    results = static_baselines(test_scores["probs"], test_scores["labels"],
                               test_scores["latencies"],
                               [spec.name for spec in specs], static_cfg,
                               stage1_idx=s1, stage2_idx=s2)
    results["PPO-CIS"] = ppo_metrics

    print("\nTest-set comparison:")
    print_table(results)

    classifier_metrics = {}
    metrics_path = CLASSIFIER_DIR / "metrics.json"
    if metrics_path.exists():
        classifier_metrics = load_json(metrics_path)

    # 验证集上表现最好的一轮（train_ppo 返回的就是这一轮权重）。
    val_key = {"reward": "val_reward", "f1": "val_f1",
               "accuracy": "val_accuracy"}[args.val_metric]
    best_epoch = max(history, key=lambda h: h.get(val_key, float("-inf")))["epoch"]

    # 7) 汇总数据统计、配置、训练曲线和结果，写出 JSON / CSV / MD / PNG
    payload = {
        "data": splits["stats"],
        "stage1": [specs[i].name for i in s1],
        "stage2": [specs[i].name for i in s2],
        "classifiers": classifier_metrics,
        "static_config": asdict(static_cfg),
        "ppo_config": asdict(ppo_cfg),
        "reward_config": asdict(reward_cfg),
        "threshold": args.threshold,
        "val_metric": args.val_metric,
        "ppo_history": history,
        "ppo_best_epoch": best_epoch,
        "ppo_val": ppo_val_metrics,
        "results": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(payload, REPORT_DIR / "ppo_cis_results.json")

    with open(REPORT_DIR / "ppo_cis_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + METHOD_COLUMNS)
        for name, metrics in results.items():
            writer.writerow([name] + [metrics.get(col, "") for col in METHOD_COLUMNS])

    write_markdown(results, classifier_metrics, specs,
                   REPORT_DIR / "ppo_cis_results.md")

    print(f"\nSaved: {PPO_DIR / 'policy.pt'}")
    print(f"Saved: {REPORT_DIR / 'ppo_cis_results.csv'}")
    print(f"Saved: {REPORT_DIR / 'ppo_cis_results.md'}")
    print(f"Saved: {REPORT_DIR / 'ppo_cis_results.json'}")

    if not args.no_plot:
        try:
            png = make_plots()
            print(f"Saved: {png}")
        except Exception as exc:      # matplotlib missing / headless backend issues
            print(f"[plot] skipped: {exc}")


if __name__ == "__main__":
    main()
