"""温度校准：在验证集上为五个分类器各拟合一个温度 T。

温度校准（temperature scaling）只调整一个标量：
    p' = sigmoid( logit(p) / T )
- T > 1：概率被压平（模型过度自信时选到）；
- T < 1：概率被拉尖（模型不够自信时选到）；
- T = 1：保持原样。

它不改变 p >= 0.5 的判定，所以单模型的 Acc / F1 不变；改变的是所有"按阈值
做决策"的地方：Stage 1 出口门控、Stage 2 融合、τ 扫描。

用法（工作目录必须是 ToxiCN_ex）：

    python -m ppo_cis.fit_temperature
    python -m ppo_cis.fit_temperature --split ppo_train --device cpu
    python -m ppo_cis.fit_temperature --force_scores

默认在 clf_dev（960 条验证集）上拟合。如果 artifacts/scores/clf_dev.npz
不存在，脚本会先用五个冻结分类器给它打一次分（这一步需要 torch 和模型权重）。

产物：artifacts/scores/calibration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import SCORE_DIR

# 温度搜索网格：0.2 ~ 10.0，步长 0.02。
GRID_START, GRID_STOP, GRID_STEP = 0.2, 10.0, 0.02
EPS = 1e-7


# ===================== 概率换算与指标 =====================
def to_logit(probs: np.ndarray) -> np.ndarray:
    """p -> logit：二分类下 logit = ln(p / (1 - p))，等价于两个原始 logits 之差。"""
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """套温度：logit 除以 T 再过 sigmoid，得到校准后的 p(toxic)。"""
    z = to_logit(probs) / max(float(temperature), 1e-6)
    z = np.clip(z, -30.0, 30.0)      # 防止 exp 溢出；超出 ±30 的概率已经饱和到 0 / 1
    return 1.0 / (1.0 + np.exp(-z))


def nll(probs: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """负对数似然：这组概率对真实标签有多"惊讶"，越小越好。"""
    p = np.clip(apply_temperature(probs, temperature), EPS, 1.0 - EPS)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """期望校准误差：按置信度分桶，比较"平均置信度"和"实际正确率"的差距。

    0 表示完全校准；越接近 1 说明模型说的和实际差得越远。
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    conf = np.maximum(probs, 1.0 - probs)
    correct = ((probs >= 0.5).astype(np.int64) == labels).astype(np.float64)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = (np.asarray(probs) >= 0.5).astype(np.int64)
    return float((preds == np.asarray(labels)).mean())


# ===================== 温度拟合 =====================
def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """在网格上扫 T，取验证集 NLL 最小的那个（一维搜索，不需要梯度）。"""
    grid = np.arange(GRID_START, GRID_STOP + 1e-9, GRID_STEP)
    losses = np.array([nll(probs, labels, t) for t in grid])
    return float(grid[int(np.argmin(losses))])


# ===================== 打分缓存的读取 / 生成 =====================
def load_or_score(split: str, args) -> tuple:
    """优先读 artifacts/scores/<split>.npz；没有就现跑一次 score_split。"""
    path = SCORE_DIR / f"{split}.npz"
    if path.exists() and not args.force_scores:
        data = np.load(path, allow_pickle=True)
        return data["probs"], data["labels"], [str(n) for n in data["names"]]

    # 只有缺缓存时才导入 torch 相关模块，保证纯读取模式下不需要 GPU 环境。
    from .config import DataConfig, default_classifier_specs, validate_classifier_models
    from .data import build_splits
    from .pipeline import score_split
    from .utils import get_device

    print(f"未找到 {path}，先用五个冻结分类器打分……")
    device = get_device(args.device)
    splits = build_splits(DataConfig(max_len=args.max_len))
    if split not in splits or split == "stats":
        raise ValueError(f"未知划分 {split}，可选：clf_dev / ppo_train / test")
    specs = default_classifier_specs(args.bert_model, args.roberta_model,
                                     args.tiny_bert_model, args.distil_roberta_model)
    validate_classifier_models(specs)
    scores = score_split(specs, splits[split], split, device, force=True)
    return scores["probs"], scores["labels"], [str(n) for n in scores["names"]]


# ===================== 主流程 =====================
def main() -> None:
    parser = argparse.ArgumentParser(description="为五个分类器拟合温度")
    parser.add_argument("--split", default="clf_dev",
                        choices=["clf_dev", "ppo_train", "test"],
                        help="拟合用的打分缓存（默认验证集 clf_dev）")
    parser.add_argument("--device", default="auto", help="需要现打分时才用到")
    parser.add_argument("--max_len", default=96, type=int)
    parser.add_argument("--bert_model", default=None)
    parser.add_argument("--roberta_model", default=None)
    parser.add_argument("--tiny_bert_model", default=None)
    parser.add_argument("--distil_roberta_model", default=None)
    parser.add_argument("--force_scores", action="store_true",
                        help="忽略已有 npz，强制重新打分")
    parser.add_argument("--out", default=str(SCORE_DIR / "calibration.json"))
    args = parser.parse_args()

    if args.split == "test":
        print("警告：test 不应用于拟合任何参数（会造成测试集泄漏），"
              "这里的结果只能当诊断看。")

    probs, labels, names = load_or_score(args.split, args)
    labels = np.asarray(labels, dtype=np.int64)
    print(f"\n拟合集: {args.split}    样本数: {len(labels)}    分类器: {len(names)}")

    header = "{:<18}{:>7}{:>10}{:>10}{:>10}{:>10}{:>12}".format(
        "Classifier", "T", "NLL(前)", "NLL(后)", "ECE(前)", "ECE(后)", "Acc(不变)")
    print(header)
    print("-" * len(header))

    temperatures = {}
    details = {}
    for j, name in enumerate(names):
        p = probs[:, j]
        temperature = fit_temperature(p, labels)
        temperatures[name] = round(temperature, 3)
        before_nll, after_nll = nll(p, labels, 1.0), nll(p, labels, temperature)
        before_ece = ece(p, labels)
        after_ece = ece(apply_temperature(p, temperature), labels)
        acc = accuracy(p, labels)
        details[name] = {
            "temperature": round(temperature, 3),
            "nll_before": round(before_nll, 6),
            "nll_after": round(after_nll, 6),
            "ece_before": round(before_ece, 6),
            "ece_after": round(after_ece, 6),
            "accuracy": round(acc, 6),
        }
        print("{:<18}{:>7.2f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>12.4f}".format(
            name, temperature, before_nll, after_nll,
            before_ece, after_ece, acc))

    payload = {
        "_meta": {
            "method": "temperature scaling (minimize NLL)",
            "split": args.split,
            "n_samples": int(len(labels)),
            "grid": [GRID_START, GRID_STOP, GRID_STEP],
            "note": "p_calibrated = sigmoid(logit(p) / T); argmax at 0.5 unchanged",
        },
        "temperatures": temperatures,
        "details": details,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n已写入: {out_path}")
    print("说明：温度只影响阈值类决策（出口门控 / 融合 / τ 扫描），"
          "不改变 τ=0.5 下的单模型指标。")


if __name__ == "__main__":
    main()
