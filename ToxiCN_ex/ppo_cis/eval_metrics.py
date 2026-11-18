"""四组口径的评测脚本（对齐 COLD 论文的 Table 2）。

COLD 论文报告的是 Acc / Macro P-R-F1 / Offensive P-R-F1 / Non-Offensive P-R-F1。
本项目原本只输出 binary（正类）指标，无法直接对照。本脚本直接读取
artifacts/scores/<split>.npz 里已缓存的概率，不重训任何模型，一次性输出
上述四组指标，并额外给出 Calls / Latency / Throughput，方便展示效率优势。

独立性：
    * 只读 scores/*.npz 与 ppo/policy.pt，不写任何训练产物；
    * 结果写到 artifacts/reports/eval_metrics_<split>.{md,json}；
    * 想回滚直接删除本文件与这两个产物即可。

用法（在 ToxiCN_ex 目录下）：
    python -m ppo_cis.eval_metrics
    python -m ppo_cis.eval_metrics --split clf_dev
    python -m ppo_cis.eval_metrics --policy ppo_cis/artifacts/ppo/policy.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import PPO_DIR, REPORT_DIR, SCORE_DIR, RewardConfig
from .pipeline import PPOEnv
from .ppo import ActorCritic
from .utils import save_json


# ===================== 四组指标 =====================
# 与 COLD 论文一致：整体 Acc + Macro 平均 + 正类（offensive）+ 负类（non-offensive）。
def four_metrics(preds, labels) -> Dict[str, float]:
    p = np.asarray(preds).astype(int)
    y = np.asarray(labels).astype(int)

    def prf(pos: int):
        return (
            float(precision_score(y, p, pos_label=pos, zero_division=0)),
            float(recall_score(y, p, pos_label=pos, zero_division=0)),
            float(f1_score(y, p, pos_label=pos, zero_division=0)),
        )

    pos_p, pos_r, pos_f1 = prf(1)
    neg_p, neg_r, neg_f1 = prf(0)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(precision_score(y, p, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, p, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "off_precision": pos_p, "off_recall": pos_r, "off_f1": pos_f1,
        "non_precision": neg_p, "non_recall": neg_r, "non_f1": neg_f1,
    }


def add_eff(metrics: Dict[str, float], calls: float, latency_s: float,
            n: int) -> Dict[str, float]:
    metrics["calls"] = float(calls)
    metrics["latency_ms"] = float(latency_s * 1000.0)
    metrics["throughput"] = float(n / max(latency_s * n, 1e-9))
    return metrics


# ===================== 策略推理 =====================
# 与 ppo.evaluate_policy 完全一致的贪心 rollout，但额外返回逐样本预测，
# 以便计算 macro / 负类指标。
@torch.no_grad()
def run_policy(policy: ActorCritic, env, device: torch.device):
    env.reset(torch.arange(env.probs.shape[0]))
    for _ in range(env.max_steps):
        masks = env.action_mask()
        if not bool(masks.any()):
            break
        logits, _, _ = policy(env.obs().to(device), masks.to(device))
        env.step(logits.argmax(dim=1).cpu())
    env.finalize()
    return (env.pred.numpy(), env.cur_labels.numpy(),
            env.sample_latency().numpy(), float(env.calls.float().mean()))


# ===================== 静态基线 =====================
# 与 pipeline.static_baselines 保持同一套规则，只是额外保留 preds。
def static_preds(probs: np.ndarray, latencies: np.ndarray,
                 stage1_idx: Sequence[int], stage2_idx: Sequence[int],
                 xi: float) -> Dict[str, tuple]:
    n, k = probs.shape
    out: Dict[str, tuple] = {}

    # 顺序执行：全部调用一遍，用最后一个 Stage-2 模型判定。
    seq_pred = (probs[:, stage2_idx[-1]] >= 0.5).astype(int)
    out["static_sequential"] = (seq_pred, np.ones(n),
                                float(np.sum(latencies)))

    # 固定阈值级联：Stage-1 高置信无毒出口，其余升级；Stage-2 同阈值判定。
    cascade_pred = np.zeros(n, dtype=int)
    calls = np.zeros(n, dtype=int)
    lat = np.zeros(n, dtype=float)
    for i in range(n):
        exited = escalated = False
        for j in stage1_idx:
            calls[i] += 1
            lat[i] += latencies[j]
            p = probs[i, j]
            if p <= 1.0 - xi:
                cascade_pred[i] = 0
                exited = True
                break
            if p >= xi:
                escalated = True
                break
        else:
            escalated = True
        if exited:
            continue
        decided = False
        for j in stage2_idx:
            calls[i] += 1
            lat[i] += latencies[j]
            p = probs[i, j]
            if p >= xi:
                cascade_pred[i] = 1
                decided = True
                break
            if p <= 1.0 - xi:
                cascade_pred[i] = 0
                decided = True
                break
        if not decided:
            cascade_pred[i] = int(probs[i, stage2_idx].mean() >= 0.5)
    out["static_cascade"] = (cascade_pred, calls, lat)

    # 多模型投票
    votes = (probs >= 0.5).sum(axis=1)
    out["majority_vote"] = ((votes >= int(np.ceil(k / 2))).astype(int),
                            np.full(n, float(k)), float(np.sum(latencies)))
    out["soft_vote"] = ((probs.mean(axis=1) >= 0.5).astype(int),
                        np.full(n, float(k)), float(np.sum(latencies)))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Report Acc / Macro / Offensive / Non-Offensive metrics (COLD style)")
    parser.add_argument("--split", default="test", help="test / clf_dev / ppo_dev / ppo_train")
    parser.add_argument("--policy", default=str(PPO_DIR / "policy.pt"),
                        help="PPO 策略文件，缺失时只输出单模型与静态基线")
    parser.add_argument("--confidence_threshold", default=None, type=float,
                        help="静态级联阈值；默认从 policy.pt 读取，读不到用 0.70")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    score_path = SCORE_DIR / f"{args.split}.npz"
    if not score_path.exists():
        raise FileNotFoundError(f"未找到打分缓存：{score_path}（先跑 train_ppo 生成）")

    data = np.load(score_path, allow_pickle=True)
    probs = data["probs"]
    labels = data["labels"]
    latencies = data["latencies"]
    names = [str(x) for x in data["names"]]
    n, k = probs.shape
    print(f"split={args.split} n={n} models={names}")

    # 阶段划分与阈值优先取自策略文件，保证和训练时完全一致。
    ckpt: Optional[dict] = None
    policy_path = args.policy
    if policy_path and Path(policy_path).exists():
        ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
        stage1_idx = list(ckpt["stage1_idx"])
        stage2_idx = list(ckpt["stage2_idx"])
        xi = (args.confidence_threshold if args.confidence_threshold is not None
              else float(ckpt.get("threshold", 0.70)))
    else:
        print("注意：未找到策略文件，跳过 PPO-CIS 行；阶段按列序推断。")
        stage1_idx, stage2_idx = [0, 1], list(range(2, k))
        xi = args.confidence_threshold if args.confidence_threshold is not None else 0.70

    rows: List[Dict[str, float]] = []

    # 1) 五个单模型
    for j, name in enumerate(names):
        preds = (probs[:, j] >= 0.5).astype(int)
        m = four_metrics(preds, labels)
        m.update(method=name, calls=1.0, latency_ms=float(latencies[j] * 1000),
                 throughput=float(1.0 / max(latencies[j], 1e-9)))
        rows.append(m)

    # 2) 静态基线
    for name, (preds, calls, lat) in static_preds(
            probs, latencies, stage1_idx, stage2_idx, xi).items():
        m = four_metrics(preds, labels)
        m.update(method=name,
                 calls=float(np.mean(calls)),
                 latency_ms=float(np.mean(lat) * 1000),
                 throughput=float(n / max(float(np.sum(lat)), 1e-9)))
        rows.append(m)

    # 3) PPO-CIS
    if ckpt is not None:
        reward_cfg = RewardConfig(**ckpt["reward_config"])
        env = PPOEnv(probs, labels, latencies, reward_cfg,
                     stage1_idx=stage1_idx, stage2_idx=stage2_idx,
                     max_steps=int(ckpt["ppo_config"]["max_steps"]),
                     threshold=float(ckpt.get("threshold", 0.70)),
                     stage2_priority=list(ckpt.get("stage2_priority", stage2_idx)))
        policy = ActorCritic(int(ckpt["state_dim"]), int(ckpt["action_dim"]),
                             int(ckpt.get("hidden_dim", 64)))
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
        device = torch.device(args.device)
        preds, true_labels, lat, calls = run_policy(policy, env, device)
        m = four_metrics(preds, true_labels)
        m.update(method="PPO-CIS", calls=calls,
                 latency_ms=float(lat.mean() * 1000),
                 throughput=float(len(true_labels) / max(lat.sum(), 1e-9)))
        rows.append(m)

    # 4) 打印与落盘
    header = ("Method", "Acc", "MacP", "MacR", "MacF1",
              "OffP", "OffR", "OffF1", "NonP", "NonR", "NonF1",
              "Calls", "Lat(ms)", "Thr")
    fmt = "{:<18s}" + "{:>7s}" * 10 + "{:>7s}" + "{:>9s}" + "{:>9s}"
    print("\n" + fmt.format(*header))
    print("-" * 122)
    for r in rows:
        print(fmt.format(
            r["method"][:18],
            f"{r['accuracy']:.4f}", f"{r['macro_precision']:.4f}",
            f"{r['macro_recall']:.4f}", f"{r['macro_f1']:.4f}",
            f"{r['off_precision']:.4f}", f"{r['off_recall']:.4f}", f"{r['off_f1']:.4f}",
            f"{r['non_precision']:.4f}", f"{r['non_recall']:.4f}", f"{r['non_f1']:.4f}",
            f"{r['calls']:.2f}", f"{r['latency_ms']:.2f}", f"{r['throughput']:.1f}"))

    report = {"split": args.split, "n": int(n), "n_positive": int(labels.sum()),
              "confidence_threshold": float(xi), "rows": rows}
    json_path = REPORT_DIR / f"eval_metrics_{args.split}.json"
    save_json(report, json_path)

    lines = [
        f"# Four-way metrics ({args.split})", "",
        f"- samples: {n} (offensive {int(labels.sum())}, "
        f"non-offensive {n - int(labels.sum())})",
        f"- confidence_threshold: {xi}", "",
        "| Method | Acc | Macro P | Macro R | Macro F1 | Off P | Off R | Off F1 "
        "| Non P | Non R | Non F1 | Calls | Lat(ms) | Throughput |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['accuracy']:.4f} | {r['macro_precision']:.4f} | "
            f"{r['macro_recall']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['off_precision']:.4f} | {r['off_recall']:.4f} | {r['off_f1']:.4f} | "
            f"{r['non_precision']:.4f} | {r['non_recall']:.4f} | {r['non_f1']:.4f} | "
            f"{r['calls']:.2f} | {r['latency_ms']:.2f} | {r['throughput']:.1f} |")
    md_path = REPORT_DIR / f"eval_metrics_{args.split}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
