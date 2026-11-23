"""运行四个外部基线并与 PPO-CIS 放进同一张表（只读 scores/*.npz）。

用法（在 ToxiCN_ex 目录下）：
    python -m ppo_cis.run_baselines
    python -m ppo_cis.run_baselines --policy ppo_cis/artifacts/ppo/policy.pt

产物（arts/reports 下）：
    baselines_test.csv   机器可读对比表（表头带单位）
    baselines_test.md    Markdown 表
    baselines_test.json  完整结果 + 每个基线在 dev 上选出的超参

口径说明：
    * 所有基线都跑在同一个 PPOEnv 上，动作掩码、阶段二判定、延迟累计、
      终止奖励与 PPO-CIS 完全一致；
    * 超参只在 ppo_dev 上按“平均回合奖励”选择，test 只评估一次；
    * RouteLLM / AutoMix / Conformal 的“专家”是阶段二分类器而非人类或 LLM，
      均属 adapted，表内方法名已标注。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .baselines_extra import (
    AutoMixPOMDP, CascadeCtx, ConformalDeferral, RouteLLMBinary,
    RouteLLMSequential, SemanticAgreement, rollout,
)
from .config import (
    PPO_DIR, REPORT_DIR, RewardConfig, SCORE_DIR, default_classifier_specs, stage_indices,
)
from .eval_metrics import four_metrics, run_policy
from .pipeline import PPOEnv
from .ppo import ActorCritic
from .utils import save_json


# 表头统一带单位：比例类指标无量纲，Calls 为次/样本，Lat 为毫秒，Thr 为样本/秒。
HEADER = ("Method", "Acc", "MacP", "MacR", "MacF1", "OffP", "OffR", "OffF1",
          "NonP", "NonR", "NonF1", "Calls(/sample)", "Lat(ms)", "Thr(samples/s)")


def load_scores(split: str) -> Dict[str, np.ndarray]:
    path = SCORE_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少打分缓存 {path}；先跑一次 train_ppo 生成 ppo_train / ppo_dev / test")
    data = np.load(path, allow_pickle=True)
    return {"probs": data["probs"], "labels": data["labels"],
            "latencies": data["latencies"],
            "names": [str(x) for x in data["names"]]}


def metrics_row(name: str, out: Dict[str, np.ndarray]) -> Dict[str, float]:
    m = four_metrics(out["preds"], out["labels"])
    m.update(method=name,
             calls=float(np.mean(out["calls"])),
             latency_ms=float(np.mean(out["latency"]) * 1000.0),
             throughput=float(len(out["labels"]) / max(float(np.sum(out["latency"])), 1e-9)))
    return m


def print_table(rows: List[Dict[str, float]]) -> None:
    fmt = "{:<22s}" + "{:>7s}" * 10 + "{:>8s}{:>10s}{:>14s}"
    print("\n" + fmt.format(*HEADER))
    print("-" * 128)
    for r in rows:
        print(fmt.format(
            r["method"][:22],
            f"{r['accuracy']:.4f}", f"{r['macro_precision']:.4f}",
            f"{r['macro_recall']:.4f}", f"{r['macro_f1']:.4f}",
            f"{r['off_precision']:.4f}", f"{r['off_recall']:.4f}", f"{r['off_f1']:.4f}",
            f"{r['non_precision']:.4f}", f"{r['non_recall']:.4f}", f"{r['non_f1']:.4f}",
            f"{r['calls']:.2f}", f"{r['latency_ms']:.2f}", f"{r['throughput']:.1f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="外部级联/路由基线对比")
    parser.add_argument("--policy", default=str(PPO_DIR / "policy.pt"))
    parser.add_argument("--train_split", default="ppo_train")
    parser.add_argument("--dev_split", default="ppo_dev")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out_prefix", default="baselines")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1) 从策略文件读取阶段划分、阈值、优先级；缺失时退回默认分类器定义。
    policy_path = Path(args.policy)
    ckpt = None
    if policy_path.exists():
        ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
        stage1 = list(ckpt["stage1_idx"])
        stage2 = list(ckpt["stage2_idx"])
        threshold = float(ckpt.get("threshold", 0.70))
        priority = list(ckpt.get("stage2_priority", stage2))
        reward_cfg = RewardConfig(**ckpt["reward_config"])
        max_steps = int(ckpt["ppo_config"]["max_steps"])
    else:
        print(f"注意：未找到 {policy_path}，按默认分类器定义推断阶段划分")
        specs = default_classifier_specs()
        stage1, stage2 = stage_indices(specs)
        threshold, priority = 0.70, list(stage2)
        reward_cfg = RewardConfig()
        max_steps = 8

    train = load_scores(args.train_split)
    dev = load_scores(args.dev_split)
    test = load_scores(args.test_split)
    names = test["names"]
    if train["probs"].shape[1] != len(names):
        raise ValueError("三个 split 的模型列数不一致，请清空 scores 后重跑 train_ppo")
    if not np.allclose(train["latencies"], test["latencies"]):
        raise ValueError("三个 split 的延迟不一致，请清空 scores 后重跑 train_ppo")
    print(f"models={names}; stage1={[names[i] for i in stage1]}; "
          f"stage2={[names[i] for i in stage2]}; threshold={threshold}")

    ctx = CascadeCtx(latencies=test["latencies"], cfg=reward_cfg,
                     stage1=stage1, stage2=stage2, max_steps=max_steps,
                     threshold=threshold, priority=priority)

    rows: List[Dict[str, float]] = []
    tuning: Dict[str, dict] = {}

    # 2) RouteLLM（监督行为克隆，sequential 版）
    print("\n[1/4] RouteLLM sequential：用 ppo_train 生成奖励最优动作标签")
    rl_seq = RouteLLMSequential(seed=args.seed)
    tuning["routellm_seq_fit"] = rl_seq.fit(train["probs"], train["labels"], ctx)
    env = ctx.env(test["probs"], test["labels"])
    out = rollout(env, rl_seq.choose)
    rows.append(metrics_row("routellm_seq(adapted)", out))
    dev_reward = float(rollout(ctx.env(dev["probs"], dev["labels"]), rl_seq.choose)["rewards"].mean())
    print(f"  dev reward = {dev_reward:.4f}")

    # 3) RouteLLM（binary 近似，附录用）
    print("\n[2/4] RouteLLM binary：先调最小模型，再决定是否改调主模型")
    rl_bin = RouteLLMBinary(seed=args.seed)
    tuning["routellm_binary"] = rl_bin.fit(
        train["probs"], train["labels"], dev["probs"], dev["labels"], ctx)
    out = rollout(ctx.env(test["probs"], test["labels"]), rl_bin.choose)
    rows.append(metrics_row("routellm_binary(adapted)", out))
    print(f"  tau = {rl_bin.threshold:.2f}")

    # 4) AutoMix（POMDP + 子集 DP）
    print("\n[3/4] AutoMix：用 ppo_train 估计观测模型并做子集 DP")
    automix = AutoMixPOMDP(seed=args.seed)
    tuning["automix_fit"] = automix.fit(train["probs"], train["labels"], ctx)
    out = rollout(ctx.env(test["probs"], test["labels"]), automix.choose)
    rows.append(metrics_row("automix_pomdp(adapted)", out))
    dev_reward = float(rollout(ctx.env(dev["probs"], dev["labels"]), automix.choose)["rewards"].mean())
    print(f"  dev reward = {dev_reward:.4f}")

    # 5) Semantic Agreement（训练无关，只调一致率阈值）
    print("\n[4/4] Semantic Agreement：在 ppo_dev 上扫一致率阈值")
    sem = SemanticAgreement()
    tuning["semantic_agreement"] = sem.fit(dev["probs"], dev["labels"], ctx)
    out = rollout(ctx.env(test["probs"], test["labels"]), sem.choose)
    rows.append(metrics_row("semantic_agree(adapted)", out))
    print(f"  tau = {sem.tau:.2f}")

    # 6) Conformal L2D（alpha 固定 0.1，在 ppo_dev 上校准）
    print("\n[+] Conformal L2D：在 ppo_dev 上做 split-conformal 校准")
    conf = ConformalDeferral(alpha=0.1)
    tuning["conformal_l2d"] = conf.fit(dev["probs"], dev["labels"], ctx)
    out = rollout(ctx.env(test["probs"], test["labels"]), conf.choose)
    rows.append(metrics_row("conformal_l2d(adapted)", out))
    print(f"  q = {conf.q:.4f}, rejector = {names[conf.rejector]}")

    # 7) PPO-CIS 参照行
    if ckpt is not None:
        print("\n[ref] PPO-CIS 贪心评估")
        policy = ActorCritic(int(ckpt["state_dim"]), int(ckpt["action_dim"]),
                             int(ckpt.get("hidden_dim", 64)))
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
        env = PPOEnv(test["probs"], test["labels"], test["latencies"], reward_cfg,
                     stage1_idx=stage1, stage2_idx=stage2,
                     max_steps=max_steps, threshold=threshold,
                     stage2_priority=priority)
        preds, labels, lat, calls = run_policy(policy, env, torch.device("cpu"))
        m = four_metrics(preds, labels)
        m.update(method="PPO-CIS", calls=calls,
                 latency_ms=float(lat.mean() * 1000.0),
                 throughput=float(len(labels) / max(float(lat.sum()), 1e-9)))
        rows.append(m)

    # 8) 打印与落盘
    print_table(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"{args.out_prefix}_{args.test_split}.json"
    save_json({"split": args.test_split, "n": int(len(test["labels"])),
               "n_positive": int(test["labels"].sum()),
               "models": names, "stage1": [names[i] for i in stage1],
               "stage2": [names[i] for i in stage2],
               "threshold": threshold,
               "priority": [names[i] for i in priority],
               "units": {"Calls": "次/样本", "Lat": "毫秒", "Thr": "样本/秒"},
               "tuning": tuning, "rows": rows}, json_path)

    lines = [
        f"# 外部基线对比（{args.test_split}）", "",
        f"- 样本数：{len(test['labels'])}（有毒 {int(test['labels'].sum())}，"
        f"无毒 {len(test['labels']) - int(test['labels'].sum())}）",
        f"- 阶段一：{', '.join(names[i] for i in stage1)}；"
        f"阶段二：{', '.join(names[i] for i in stage2)}",
        f"- 置信阈值：{threshold}",
        "- 单位：Calls 为次/样本，Lat 为毫秒，Thr 为样本/秒；比例类指标无量纲",
        "- 超参只在 ppo_dev 上按平均回合奖励选择，test 只评估一次",
        "- 各基线的“专家”为阶段二分类器而非人类或大模型，均属 adapted",
        "",
        "| Method | Acc | MacP | MacR | MacF1 | OffP | OffR | OffF1 "
        "| NonP | NonR | NonF1 | Calls(/sample) | Lat(ms) | Thr(samples/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['accuracy']:.4f} | {r['macro_precision']:.4f} | "
            f"{r['macro_recall']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['off_precision']:.4f} | {r['off_recall']:.4f} | {r['off_f1']:.4f} | "
            f"{r['non_precision']:.4f} | {r['non_recall']:.4f} | {r['non_f1']:.4f} | "
            f"{r['calls']:.2f} | {r['latency_ms']:.2f} | {r['throughput']:.1f} |")
    md_path = REPORT_DIR / f"{args.out_prefix}_{args.test_split}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = REPORT_DIR / f"{args.out_prefix}_{args.test_split}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(HEADER))
        for r in rows:
            writer.writerow([
                r["method"], f"{r['accuracy']:.4f}",
                f"{r['macro_precision']:.4f}", f"{r['macro_recall']:.4f}",
                f"{r['macro_f1']:.4f}",
                f"{r['off_precision']:.4f}", f"{r['off_recall']:.4f}", f"{r['off_f1']:.4f}",
                f"{r['non_precision']:.4f}", f"{r['non_recall']:.4f}", f"{r['non_f1']:.4f}",
                f"{r['calls']:.2f}", f"{r['latency_ms']:.2f}", f"{r['throughput']:.1f}",
            ])

    print(f"\nSaved: {md_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
