"""One-command pipeline: train classifiers -> train PPO-CIS -> evaluate -> plot.

Usage (from the ToxiCN_ex directory):
    python -m ppo_cis.run_all
    python -m ppo_cis.run_all --step classifiers --only tiny_bert distil_roberta
    python -m ppo_cis.run_all --step ppo
    python -m ppo_cis.run_all --step plot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TOXICN_EX_DIR = Path(__file__).resolve().parent.parent


# 用当前 Python 解释器执行 `python -m 模块`，工作目录固定在 ToxiCN_ex。
def _run(module: str, extra: list) -> None:
    cmd = [sys.executable, "-m", module] + extra
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(TOXICN_EX_DIR), check=True)


# ===================== 主流程 =====================
# 按 --step 依次调度三个阶段：train_classifiers → train_ppo → plot_results，
# 公共参数（模型路径、max_len）透传给子进程。
def main():
    parser = argparse.ArgumentParser(description="Run the full PPO-CIS pipeline")
    parser.add_argument("--step", choices=["all", "classifiers", "ppo", "plot"],
                        default="all")
    parser.add_argument("--bert_model", default=None)
    parser.add_argument("--roberta_model", default=None)
    parser.add_argument("--tiny_bert_model", default=None)
    parser.add_argument("--distil_roberta_model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ppo_device", default="cpu")
    parser.add_argument("--max_len", default=96, type=int)
    parser.add_argument("--epochs", default=10, type=int, help="PPO epochs")
    parser.add_argument("--time_budget", default=0.0015, type=float,
                        help="u in the time reward, seconds (1.5 ms reference budget)")
    parser.add_argument("--time_theta", default=0.5, type=float)
    parser.add_argument("--throughput_scale", default=0.0, type=float,
                        help="Eq.(6) bonus weight; 0 keeps it report-only")
    parser.add_argument("--aux_coef", default=0.10, type=float)
    parser.add_argument("--entropy_coef", default=0.05, type=float)
    parser.add_argument("--explore_decide", default=0.10, type=float)
    parser.add_argument("--threshold", default=0.70, type=float)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shared = ["--max_len", str(args.max_len)]
    if args.bert_model:
        shared += ["--bert_model", args.bert_model]
    if args.roberta_model:
        shared += ["--roberta_model", args.roberta_model]
    if args.tiny_bert_model:
        shared += ["--tiny_bert_model", args.tiny_bert_model]
    if args.distil_roberta_model:
        shared += ["--distil_roberta_model", args.distil_roberta_model]

    # 1) 训练五个分类器（--only 可只训部分，--force 覆盖已有权重）
    if args.step in ("all", "classifiers"):
        extra = shared + ["--device", args.device]
        if args.only:
            extra += ["--only"] + args.only
        if args.force:
            extra.append("--force")
        _run("ppo_cis.train_classifiers", extra)

    # 2) 训练 PPO 策略并输出对比表；--force 会转成 --force_scores 重算打分缓存
    if args.step in ("all", "ppo"):
        extra = shared + [
            "--device", args.device,
            "--ppo_device", args.ppo_device,
            "--epochs", str(args.epochs),
            "--time_budget", str(args.time_budget),
            "--time_theta", str(args.time_theta),
            "--throughput_scale", str(args.throughput_scale),
            "--aux_coef", str(args.aux_coef),
            "--entropy_coef", str(args.entropy_coef),
            "--explore_decide", str(args.explore_decide),
            "--threshold", str(args.threshold),
        ]
        if args.force:
            extra.append("--force_scores")
        _run("ppo_cis.train_ppo", extra)      # writes CSV/MD/JSON and the PNG

    # 3) 单独重新绘图（结果表已存在时用）
    if args.step == "plot":
        _run("ppo_cis.plot_results", [])


if __name__ == "__main__":
    main()
