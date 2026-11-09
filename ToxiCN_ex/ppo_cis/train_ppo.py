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
from dataclasses import asdict
from typing import Dict

import torch

from .config import (
    CLASSIFIER_DIR, DataConfig, PPO_DIR, PPOConfig, REPORT_DIR, RewardConfig,
    StaticConfig, default_classifier_specs, stage_indices,
)
from .data import build_splits, print_split_stats
from .pipeline import PPOEnv, score_split, static_baselines
from .ppo import evaluate_policy, train_ppo
from .plot_results import make_plots
from .utils import get_device, load_json, save_json

METHOD_COLUMNS = ["accuracy", "precision", "recall", "f1",
                  "avg_classifiers", "avg_latency_ms", "throughput"]


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


def write_markdown(results: Dict[str, Dict[str, float]], classifier_metrics: Dict,
                   specs, path) -> None:
    stage_by_name = {spec.name: spec.stage for spec in specs}
    lines = [
        "# PPO-CIS two-stage cascade — results",
        "",
        "Stage 1 (throughput): `tiny_bert`, `distil_roberta`. "
        "Stage 2 (accuracy): `bert`, `roberta_fusion`, `tke_bert`.",
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
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--update_epochs", default=4, type=int)
    parser.add_argument("--clip", default=0.2, type=float)
    parser.add_argument("--aux_coef", default=0.10, type=float,
                        help="weight of the auxiliary cross-entropy loss")
    parser.add_argument("--entropy_coef", default=0.01, type=float,
                        help="entropy bonus; raise it if the policy stops exploring early exits")
    parser.add_argument("--explore_decide", default=0.25, type=float,
                        help="fraction of rollouts forced to decide early (anti-collapse)")
    parser.add_argument("--time_budget", default=1.0, type=float,
                        help="u in Eq.(2) time reward, in seconds (paper: 1.0)")
    parser.add_argument("--time_theta", default=10.0, type=float,
                        help="theta_1 in Eq.(2) (paper: 10)")
    parser.add_argument("--throughput_scale", default=1.0, type=float,
                        help="weight of the Eq.(6) batch throughput bonus")
    parser.add_argument("--confidence_threshold", default=0.70, type=float,
                        help="threshold of the fixed cascade baseline")
    parser.add_argument("--threshold", default=0.70, type=float,
                        help="confidence gate of the PPO action mask")
    parser.add_argument("--force_scores", action="store_true")
    parser.add_argument("--no_plot", action="store_true", help="skip PNG generation")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    device = get_device(args.device)
    ppo_device = get_device(args.ppo_device)
    print(f"Classifier device: {device}; PPO device: {ppo_device}")

    data_cfg = DataConfig(max_len=args.max_len, seed=args.seed)
    splits = build_splits(data_cfg)
    print_split_stats(splits)

    specs = default_classifier_specs(args.bert_model, args.roberta_model,
                                     args.tiny_bert_model, args.distil_roberta_model)
    s1, s2 = stage_indices(specs)
    print(f"Stage 1: {[specs[i].name for i in s1]}")
    print(f"Stage 2: {[specs[i].name for i in s2]}")

    ppo_scores = score_split(specs, splits["ppo_train"], "ppo_train", device,
                             args.force_scores)
    test_scores = score_split(specs, splits["test"], "test", device,
                              args.force_scores)

    ppo_cfg = PPOConfig(epochs=args.epochs, batch_size=args.batch_size,
                        update_epochs=args.update_epochs, clip=args.clip,
                        aux_coef=args.aux_coef, entropy_coef=args.entropy_coef,
                        explore_decide=args.explore_decide, seed=args.seed)
    reward_cfg = RewardConfig(time_budget=args.time_budget,
                              time_theta=args.time_theta,
                              throughput_scale=args.throughput_scale)
    static_cfg = StaticConfig(confidence_threshold=args.confidence_threshold)

    train_env = PPOEnv(ppo_scores["probs"], ppo_scores["labels"],
                       ppo_scores["latencies"], reward_cfg,
                       stage1_idx=s1, stage2_idx=s2,
                       max_steps=ppo_cfg.max_steps, threshold=args.threshold)
    policy, history = train_ppo(train_env, ppo_cfg, ppo_device, reward_cfg)

    PPO_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "state_dim": train_env.K,
        "action_dim": train_env.action_dim,
        "hidden_dim": ppo_cfg.hidden_dim,
        "specs": [asdict(spec) for spec in specs],
        "stage1_idx": s1,
        "stage2_idx": s2,
        "ppo_config": asdict(ppo_cfg),
        "reward_config": asdict(reward_cfg),
        "threshold": args.threshold,
        "history": history,
    }, PPO_DIR / "policy.pt")

    test_env = PPOEnv(test_scores["probs"], test_scores["labels"],
                      test_scores["latencies"], reward_cfg,
                      stage1_idx=s1, stage2_idx=s2,
                      max_steps=ppo_cfg.max_steps, threshold=args.threshold)
    ppo_metrics = evaluate_policy(policy, test_env, ppo_device)

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

    payload = {
        "data": splits["stats"],
        "stage1": [specs[i].name for i in s1],
        "stage2": [specs[i].name for i in s2],
        "classifiers": classifier_metrics,
        "static_config": asdict(static_cfg),
        "ppo_config": asdict(ppo_cfg),
        "reward_config": asdict(reward_cfg),
        "threshold": args.threshold,
        "ppo_history": history,
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
