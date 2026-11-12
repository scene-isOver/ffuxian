"""错误相关性分析：判断五个分类器是否足够互补。

用法（工作目录必须是 ToxiCN_ex）：

    python -m ppo_cis.analyze_errors
    python -m ppo_cis.analyze_errors --split ppo_train

读取 ``artifacts/scores/<split>.npz``（由 ``score_split`` 生成），对每个分类器
构造错误向量 ``E_i(x) = 1[pred_i(x) != y(x)]``，然后输出：

1. 单分类器的 Acc / Prec / Rec / F1 / 错误率；
2. 两两错误向量的 Pearson 相关矩阵（``corr(E_i, E_j)``）；
3. 两两预测一致率、错误重合率（Jaccard）、独立纠错率；
4. 每个分类器“独占错误”的比例（只有它错、其他都对）。

相关矩阵非对角元素普遍 > 0.8 时，说明各模型误差高度相关、互补性不足，
单纯加模型或做融合的收益有限；此时提升准确率应该靠更强的单模型或换架构。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCORE_DIR = Path(__file__).resolve().parent / "artifacts" / "scores"


def _metrics(pred: np.ndarray, label: np.ndarray):
    """手写二分类指标，避免依赖 torch / sklearn。"""
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    acc = (tp + tn) / max(len(label), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return acc, prec, rec, f1


def load_scores(split: str):
    path = SCORE_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少打分缓存 {path}。请先运行 "
            f"`python -m ppo_cis.train_ppo --force_scores` 生成。")
    data = np.load(path, allow_pickle=True)
    return data["probs"], data["labels"], [str(n) for n in data["names"]]


def main():
    parser = argparse.ArgumentParser(description="五个分类器的错误相关性分析")
    parser.add_argument("--split", default="test", choices=["test", "ppo_train"],
                        help="要分析的打分缓存")
    parser.add_argument("--tau", default=0.5, type=float,
                        help="判定阈值：p(toxic) >= tau 记为有毒")
    args = parser.parse_args()

    probs, labels, names = load_scores(args.split)
    preds = (probs >= args.tau).astype(int)
    errors = (preds != labels[:, None]).astype(float)      # N × K
    n, k = errors.shape

    print(f"\n划分: {args.split}    样本数: {n}    分类器数: {k}    阈值: {args.tau}")
    print(f"有毒样本: {int(labels.sum())}    无毒样本: {int((1 - labels).sum())}")

    # 1) 单分类器指标
    print("\n单分类器表现")
    print(f"{'模型':<18}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'错误率':>9}")
    print("-" * 59)
    for j, name in enumerate(names):
        acc, prec, rec, f1 = _metrics(preds[:, j], labels)
        print(f"{name:<18}{acc:>8.4f}{prec:>8.4f}"
              f"{rec:>8.4f}{f1:>8.4f}{errors[:, j].mean():>9.4f}")

    # 2) 错误相关性矩阵
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(errors.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    print("\n错误相关性矩阵（Pearson，越高说明错得越像）")
    print(f"{'':<18}" + "".join(f"{name[:9]:>10}" for name in names))
    for i, name in enumerate(names):
        print(f"{name:<18}" + "".join(f"{corr[i, j]:>10.3f}" for j in range(k)))

    off_diag = corr[~np.eye(k, dtype=bool)]
    print(f"\n非对角平均相关: {off_diag.mean():.3f}    最大: {off_diag.max():.3f}")

    # 3) 两两一致率 / 错误重合率 / 独立纠错率
    print("\n两两关系（一致率 / 错误重合率 / 互相纠错率）")
    print(f"{'':<18}" + "".join(f"{name[:9]:>10}" for name in names))
    for i, name in enumerate(names):
        cells = []
        for j in range(k):
            if i == j:
                cells.append(f"{'-':>10}")
                continue
            same = float((preds[:, i] == preds[:, j]).mean())
            both_err = float(((errors[:, i] == 1) & (errors[:, j] == 1)).sum())
            union_err = float(((errors[:, i] == 1) | (errors[:, j] == 1)).sum())
            overlap = both_err / union_err if union_err else 0.0
            cells.append(f"{same:.2f}/{overlap:.2f}".rjust(10))
        print(f"{name:<18}" + "".join(cells))

    # 4) 独占错误率：只有它错、其余都对
    err_count = errors.sum(axis=1)
    print("\n独占错误率（只有它错、其他都对，占它全部错误的比例）")
    for j, name in enumerate(names):
        own_err = errors[:, j] == 1
        if own_err.sum() == 0:
            print(f"{name:<18}{'无错误':>10}")
            continue
        exclusive = float(((err_count == 1) & own_err).sum() / own_err.sum())
        print(f"{name:<18}{exclusive:>10.4f}")

    # 5) 结论提示
    mean_corr = off_diag.mean()
    print()
    if mean_corr >= 0.80:
        print(f"结论：平均错误相关 {mean_corr:.3f}，高度相关，互补性不足。"
              "加模型/做融合收益有限，提升准确率应优先考虑更强的单模型或更换架构。")
    elif mean_corr >= 0.60:
        print(f"结论：平均错误相关 {mean_corr:.3f}，中等相关，融合仍有一定空间，"
              "但收益不会太大。")
    else:
        print(f"结论：平均错误相关 {mean_corr:.3f}，互补性较好，级联/融合值得投入。")


if __name__ == "__main__":
    main()
