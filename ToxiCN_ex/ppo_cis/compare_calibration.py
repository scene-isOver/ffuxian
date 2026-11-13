"""温度校准前后对比：只用 numpy，不需要 GPU、不需要重训、不改任何现有代码。

读入 artifacts/scores/test.npz（原始概率）和 artifacts/scores/calibration.json
（五个温度），把同一套判定规则分别在"原始概率"和"校准后概率"上跑一遍，
看温度校准到底有没有用、如果有用是在哪个环节。

对比的内容：
1. 单分类器 F1（理论上必须完全一致，用来验证温度不改 argmax）；
2. 三个融合方法在固定阈值下的 F1；
3. static_cascade 与 Stage-2 均值融合的 τ 扫描。

用法（工作目录必须在 ToxiCN_ex）：

    python -m ppo_cis.compare_calibration
    python -m ppo_cis.compare_calibration --split ppo_train
    python -m ppo_cis.compare_calibration --scores 某个.npz --calibration 某个.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .config import SCORE_DIR, default_classifier_specs, stage_indices

EPS = 1e-7


# ===================== 概率 + 温度 =====================
def to_logit(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """p' = sigmoid(logit(p) / T)；T=1 时原样返回。"""
    if abs(temperature - 1.0) < 1e-9:
        return np.asarray(probs, dtype=np.float64).copy()
    z = np.clip(to_logit(probs) / max(float(temperature), 1e-6), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def load_calibration(path: Path, names: Sequence[str]) -> Dict[str, float]:
    """读 calibration.json；缺文件或缺某个模型时按 T=1（即不校准）处理。"""
    if not path.exists():
        print(f"警告：找不到 {path}，全部按 T=1 处理（前后结果会完全相同）")
        return {name: 1.0 for name in names}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    table = payload.get("temperatures", payload)
    out = {}
    for name in names:
        if name in table:
            out[name] = float(table[name])
        else:
            print(f"警告：{name} 没有温度记录，按 T=1 处理")
            out[name] = 1.0
    return out


# ===================== 指标 =====================
def metrics(preds: np.ndarray, labels: np.ndarray,
            latencies: np.ndarray = None, calls: np.ndarray = None) -> Dict[str, float]:
    """手写混淆矩阵，避免依赖 sklearn / torch。"""
    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    n = max(len(labels), 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out = {"accuracy": (tp + tn) / n, "precision": prec, "recall": rec, "f1": f1}
    if latencies is not None:
        out["avg_latency_ms"] = float(np.mean(latencies) * 1000)
    if calls is not None:
        out["avg_classifiers"] = float(np.mean(calls))
        out["throughput"] = float(len(labels) / max(np.sum(latencies), 1e-9))
    return out


# ===================== 判定规则（与 pipeline.py 保持一致） =====================
def static_cascade(probs: np.ndarray, labels: np.ndarray, latencies: np.ndarray,
                   stage1: Sequence[int], stage2: Sequence[int], xi: float):
    """固定阈值级联：Stage 1 高置信无毒直接出口，否则升级；Stage 2 单侧判定。"""
    n = probs.shape[0]
    preds = np.zeros(n, dtype=np.int64)
    calls = np.zeros(n, dtype=int)
    lat = np.zeros(n, dtype=float)
    for i in range(n):
        exited = False
        for j in stage1:
            calls[i] += 1
            lat[i] += latencies[j]
            p = probs[i, j]
            if p <= 1.0 - xi:          # 高置信无毒 -> 出口
                preds[i] = 0
                exited = True
                break
            if p >= xi:                # 高置信有毒 -> 升级
                break
        if exited:
            continue

        decided = False
        for j in stage2:
            calls[i] += 1
            lat[i] += latencies[j]
            p = probs[i, j]
            if p >= xi:
                preds[i] = 1
                decided = True
                break
            if p <= 1.0 - xi:
                preds[i] = 0
                decided = True
                break
        if not decided:
            preds[i] = int(probs[i, stage2].mean() >= 0.5)
    return preds, calls, lat


def mean_fusion(probs: np.ndarray, idx: Sequence[int], xi: float) -> np.ndarray:
    """对指定模型取 p(toxic) 均值，再用单侧阈值判定（等价 PPO-CIS 的 Stage 2 规则）。"""
    mean_toxic = probs[:, idx].mean(axis=1)
    return (1.0 - mean_toxic < xi).astype(np.int64)


# ===================== 主流程 =====================
def main() -> None:
    parser = argparse.ArgumentParser(description="温度校准前后对比")
    parser.add_argument("--split", default="test", choices=["test", "ppo_train", "clf_dev"])
    parser.add_argument("--scores", default=None, help="默认 artifacts/scores/<split>.npz")
    parser.add_argument("--calibration", default=str(SCORE_DIR / "calibration.json"))
    parser.add_argument("--tau", default=0.70, type=float, help="固定对比用的阈值")
    parser.add_argument("--taus", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    args = parser.parse_args()

    scores_path = Path(args.scores) if args.scores else SCORE_DIR / f"{args.split}.npz"
    if not scores_path.exists():
        raise FileNotFoundError(f"缺少 {scores_path}，先跑一次 train_ppo 生成打分缓存")

    data = np.load(scores_path, allow_pickle=True)
    probs = np.asarray(data["probs"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    latencies = np.asarray(data["latencies"], dtype=np.float64)
    names = [str(n) for n in data["names"]]

    temps = load_calibration(Path(args.calibration), names)
    cal_probs = np.stack([apply_temperature(probs[:, j], temps[names[j]])
                          for j in range(len(names))], axis=1)

    # 用 config 里的 stage 划分（按名字匹配，不依赖 npz 的列顺序）
    specs = default_classifier_specs()
    stage_of = {spec.name: spec.stage for spec in specs}
    stage1 = [j for j, n in enumerate(names) if stage_of.get(n) == 1]
    stage2 = [j for j, n in enumerate(names) if stage_of.get(n) == 2]
    if not stage1 or not stage2:
        stage1, stage2 = stage_indices(specs)
        print("警告：名字与 config 对不上，退回按 config 的下标划分")

    print(f"\n数据: {scores_path}  ({len(labels)} 条, {len(names)} 个分类器)")
    print("温度: " + ", ".join(f"{n}={temps[n]:.2f}" for n in names))
    print(f"Stage 1: {[names[j] for j in stage1]}    Stage 2: {[names[j] for j in stage2]}")

    # ---- 1) 单分类器：校准前后必须完全一致 ----
    print("\n== 1) 单分类器（温度不改变 argmax，两列应完全相同）==")
    print("{:<18}{:>10}{:>10}{:>9}".format("Classifier", "F1(前)", "F1(后)", "差值"))
    print("-" * 47)
    for j, name in enumerate(names):
        f1_raw = metrics((probs[:, j] >= 0.5).astype(int), labels)["f1"]
        f1_cal = metrics((cal_probs[:, j] >= 0.5).astype(int), labels)["f1"]
        print("{:<18}{:>10.4f}{:>10.4f}{:>9.4f}".format(name, f1_raw, f1_cal, f1_cal - f1_raw))

    # ---- 2) 固定阈值下的融合方法 ----
    xi = args.tau
    print(f"\n== 2) 融合方法 @ τ={xi:.2f} ==")
    print("{:<26}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
        "Method", "F1(前)", "F1(后)", "ΔF1", "Prec(后)", "Rec(后)"))
    print("-" * 71)

    rows = []
    # 全部 5 个模型软投票（阈值 0.5）
    rows.append(("soft_vote(全部)", (probs.mean(axis=1) >= 0.5).astype(int),
                 (cal_probs.mean(axis=1) >= 0.5).astype(int), None, None))
    # Stage 2 三个模型均值 + 单侧阈值
    rows.append((f"mean_stage2(τ={xi:.2f})", mean_fusion(probs, stage2, xi),
                 mean_fusion(cal_probs, stage2, xi), None, None))
    # 固定阈值级联
    p_raw, c_raw, l_raw = static_cascade(probs, labels, latencies, stage1, stage2, xi)
    p_cal, c_cal, l_cal = static_cascade(cal_probs, labels, latencies, stage1, stage2, xi)
    rows.append((f"static_cascade(τ={xi:.2f})", p_raw, p_cal, (c_raw, l_raw), (c_cal, l_cal)))

    for name, pr, pc, extra_raw, extra_cal in rows:
        m_raw = metrics(pr, labels)
        m_cal = metrics(pc, labels)
        print("{:<26}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}".format(
            name, m_raw["f1"], m_cal["f1"], m_cal["f1"] - m_raw["f1"],
            m_cal["precision"], m_cal["recall"]))

    # ---- 3) τ 扫描 ----
    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    print("\n== 3) τ 扫描：static_cascade / mean_stage2 ==")
    print("{:>6}{:>12}{:>12}{:>9}{:>12}{:>12}{:>9}{:>8}{:>8}".format(
        "τ", "casc F1前", "casc F1后", "Δ", "mean F1前", "mean F1后", "Δ", "calls前", "calls后"))
    print("-" * 103)
    best = {"raw": (-1, 0.0), "cal": (-1, 0.0)}
    for tau in taus:
        pr, cr, _ = static_cascade(probs, labels, latencies, stage1, stage2, tau)
        pc, cc, _ = static_cascade(cal_probs, labels, latencies, stage1, stage2, tau)
        fr, fc = metrics(pr, labels)["f1"], metrics(pc, labels)["f1"]
        mr, mc = metrics(mean_fusion(probs, stage2, tau), labels)["f1"], \
                 metrics(mean_fusion(cal_probs, stage2, tau), labels)["f1"]
        print("{:>6.2f}{:>12.4f}{:>12.4f}{:>9.4f}{:>12.4f}{:>12.4f}{:>9.4f}{:>8.2f}{:>8.2f}".format(
            tau, fr, fc, fc - fr, mr, mc, mc - mr, cr.mean(), cc.mean()))
        if fr > best["raw"][1]:
            best["raw"] = (tau, fr)
        if fc > best["cal"][1]:
            best["cal"] = (tau, fc)

    print("\n== 结论 ==")
    print(f"static_cascade 最佳：校准前 τ={best['raw'][0]:.2f} F1={best['raw'][1]:.4f}  |  "
          f"校准后 τ={best['cal'][0]:.2f} F1={best['cal'][1]:.4f}  "
          f"(Δ={best['cal'][1] - best['raw'][1]:+.4f})")
    print(f"单模型最强 F1 = {max(metrics((probs[:, j] >= 0.5).astype(int), labels)['f1'] for j in range(len(names))):.4f}")
    print("注意：这里没有 PPO 策略，无法复现 PPO-CIS 的完整两阶段结果；"
          "它只能说明'概率校准本身'对阈值类规则的影响。")


if __name__ == "__main__":
    main()
