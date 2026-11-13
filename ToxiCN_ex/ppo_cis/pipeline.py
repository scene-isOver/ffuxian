"""Frozen-classifier scoring, the two-stage PPO-CIS environment and static baselines.

PPO-CIS works on the outputs of the *frozen* classifiers, so we pre-compute one
toxicity probability per (sample, classifier) and cache it.  The RL part then
simulates the cascade without re-running the heavy transformer encoders.

Environment (two stages, following the paper's MDP):
    state   : beta vector of length K; called classifiers hold p(toxic),
              uncalled classifiers hold -1.
    actions : 0..K-1 = call classifier k, K = "toxic", K+1 = "nontoxic".
    stage 1 : 只输出“无毒”；融合概率达阈值即可出口，调用过任意模型后也可随时
              升级到第二阶段（策略自己决定什么时候跳过剩余小模型）。
    stage 2 : 按同一阈值单侧判定：无毒概率达阈值判无毒，否则判有毒。
    reward  : 逐样本时间惩罚（time_reward）+ 分类奖励 r / -iota*r
              + 可选的 Eq.(6) 吞吐奖励（默认权重 0，仅作报告指标）。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CLASSIFIER_DIR, ClassifierSpec, RewardConfig, SCORE_DIR, StaticConfig
from .models import build_classifier
from .data import EncodedDataset, encode_split, get_tokenizer
from .utils import compute_metrics, forward_classifier


# ---------------------------------------------------------------- scoring -----

# ===================== 分类器加载与打分 =====================
# 从 artifacts/classifiers/<name>.pt 还原模型与规格，设为 eval 模式（冻结，不参与训练）
def load_classifier(name: str, device: torch.device):
    path = CLASSIFIER_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {path}. Train the classifiers first with "
            f"`python -m ppo_cis.train_classifiers`."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = ClassifierSpec(**ckpt["spec"])
    model = build_classifier(spec)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# 用冻结的分类器给整个 split 打分，得到 (N, K) 的毒性概率矩阵；
# 结果缓存到 artifacts/scores/<split>.npz，模型名一致就复用（注意：不校验样本数）。
@torch.no_grad()
def score_split(specs: Sequence[ClassifierSpec], records: List[dict], split_name: str,
                device: torch.device, force: bool = False) -> Dict[str, np.ndarray]:
    """Return probs (N, K), labels (N,), latencies (K,), classifier names."""
    out_path = SCORE_DIR / f"{split_name}.npz"
    names = [spec.name for spec in specs]
    if out_path.exists() and not force:
        data = np.load(out_path, allow_pickle=True)
        if list(data["names"]) == names:
            return {k: data[k] for k in data.files}

    probs = np.zeros((len(records), len(specs)), dtype=np.float32)
    latencies = np.zeros(len(specs), dtype=np.float32)
    labels = np.array([row["label"] for row in records], dtype=np.int64)

    for j, spec in enumerate(specs):
        model, ckpt = load_classifier(spec.name, device)
        tokenizer = get_tokenizer(ckpt["tokenizer_name"])
        tensors = encode_split(records, tokenizer, split_name, ckpt["max_len"],
                               knowledge=ckpt.get("use_knowledge", False))
        loader = DataLoader(EncodedDataset(tensors), batch_size=128,
                            shuffle=False, num_workers=0)
        batch_probs = []
        for batch in loader:
            logits = forward_classifier(model, batch, device)
            batch_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        probs[:, j] = np.concatenate(batch_probs)
        latencies[j] = float(ckpt["test_metrics"]["latency_sec"])
        print(f"  scored {split_name} with {spec.name}: "
              f"mean_p={probs[:, j].mean():.4f}, latency={latencies[j] * 1000:.2f}ms")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, probs=probs, labels=labels, latencies=latencies,
             names=np.array(names, dtype=object))
    return {"probs": probs, "labels": labels, "latencies": latencies,
            "names": np.array(names, dtype=object)}


# --------------------------------------------------------------- baselines ----

# 统一指标口径：Acc/Prec/Rec/F1 + 平均调用数 + 平均延迟(ms) + 吞吐（样本/秒）
def _metrics(preds: np.ndarray, labels: np.ndarray, latencies: np.ndarray,
             calls: np.ndarray) -> Dict[str, float]:
    metrics = compute_metrics(preds, labels)
    metrics["avg_classifiers"] = float(np.mean(calls))
    metrics["avg_latency_ms"] = float(np.mean(latencies) * 1000)
    metrics["throughput"] = float(len(labels) / max(np.sum(latencies), 1e-9))
    return metrics


# ===================== 静态基线 =====================
# 依次算：5 个单分类器、顺序调用、固定阈值级联、多数投票、软投票，用于和 PPO-CIS 对比。
def static_baselines(probs: np.ndarray, labels: np.ndarray, latencies: np.ndarray,
                     names: Sequence[str], cfg: StaticConfig,
                     stage1_idx: Sequence[int], stage2_idx: Sequence[int]
                     ) -> Dict[str, Dict[str, float]]:
    n, k = probs.shape
    results: Dict[str, Dict[str, float]] = {}

    # 1) every single classifier
    for j, name in enumerate(names):
        preds = (probs[:, j] >= 0.5).astype(int)
        results[name] = _metrics(preds, labels, np.full(n, latencies[j]), np.ones(n))

    # 2) sequential cascade: call every stage-1 classifier, then every stage-2 one
    results["static_sequential"] = _metrics(
        (probs[:, stage2_idx[-1]] >= 0.5).astype(int), labels,
        np.full(n, float(np.sum(latencies))), np.full(n, k))

    # 3) 固定阈值级联（核心基线）：第一阶段高置信无毒直接出口，其余升级；
    #    第二阶段按同一阈值判定，两阶段都没定论时用大模型均值兜底。
    # 3) fixed-threshold cascade: high-confidence nontoxic exits stage 1,
    #    everything else escalates; stage 2 makes the final decision
    xi = cfg.confidence_threshold
    preds = np.zeros(n, dtype=int)
    calls = np.zeros(n, dtype=int)
    sample_latency = np.zeros(n, dtype=float)
    for i in range(n):
        escalated = False
        exited = False
        for j in stage1_idx:
            calls[i] += 1
            sample_latency[i] += latencies[j]
            p = probs[i, j]
            if p <= 1.0 - xi:          # high-confidence nontoxic -> exit
                preds[i] = 0
                exited = True
                break
            if p >= xi:                # high-confidence toxic -> escalate
                escalated = True
                break
        else:
            escalated = True           # stage-1 exhausted without an exit
        if exited:
            continue

        decided = False
        for j in stage2_idx:
            calls[i] += 1
            sample_latency[i] += latencies[j]
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
            preds[i] = int(probs[i, stage2_idx].mean() >= 0.5)

    results["static_cascade"] = _metrics(preds, labels, sample_latency, calls)

    # 4) majority / soft vote over all classifiers
    votes = (probs >= 0.5).sum(axis=1)
    majority_preds = (votes >= int(np.ceil(k / 2))).astype(int)
    results["majority_vote"] = _metrics(
        majority_preds, labels, np.full(n, float(np.sum(latencies))), np.full(n, k))
    soft_preds = (probs.mean(axis=1) >= 0.5).astype(int)
    results["soft_vote"] = _metrics(
        soft_preds, labels, np.full(n, float(np.sum(latencies))), np.full(n, k))

    return results


# ----------------------------------------------------------------- rewards ----

def time_reward(total_latency_sec: float, cfg: RewardConfig) -> float:
    """单条样本的处理时间惩罚（逐样本）。

    论文 Eq. (2) 是 ``phi + theta * log2(u - t)``，但它按秒设计：实际延迟只有
    1 ms 量级，``t << u`` 时整个可行区间只变化 0.05，几乎没有梯度；而且
    ``t -> u`` 处会发散到 -inf，低预算下不可用。这里改成线性、无量纲的惩罚
    ``-theta * t / u``：t 越快惩罚越小，量级与分类奖励可比。
    """
    return cfg.time_phi - cfg.time_theta * total_latency_sec / max(cfg.time_budget, cfg.eps)


def throughput_bonus(total_latency_sec: float, n_samples: int,
                     cfg: RewardConfig) -> float:
    """Eq. (6): ``psi_M = M / sum(t_i)`` for a rollout batch, scaled by the config."""
    return cfg.throughput_scale * n_samples / max(total_latency_sec, cfg.eps)


# ------------------------------------------------------------ environment -----_

class PPOEnv:
    """One episode per sample with a two-stage cascade.

    Actions (K+2): 0..K-1 select classifier k, K = "toxic", K+1 = "nontoxic".
    In stage 1 "toxic" means *escalate to stage 2*; in stage 2 it terminates.
    """

    def __init__(self, probs: np.ndarray, labels: np.ndarray, latencies: np.ndarray,
                 reward_cfg: RewardConfig, stage1_idx: Sequence[int],
                 stage2_idx: Sequence[int], max_steps: int = 8,
                 threshold: float = 0.7):
        self.probs = torch.as_tensor(probs, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.latencies = torch.as_tensor(latencies, dtype=torch.float32)
        self.cfg = reward_cfg
        self.K = self.probs.shape[1]
        self.action_dim = self.K + 2
        self.stage1_idx = list(stage1_idx)
        self.stage2_idx = list(stage2_idx)
        self.max_steps = max_steps
        self.threshold = threshold
        self.n = 0

    # ---------------------------------------------------------- lifecycle ----
    def reset(self, indices) -> torch.Tensor:
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.cur_probs = self.probs[self.indices]
        self.cur_labels = self.labels[self.indices]
        self.n = self.cur_probs.shape[0]
        self.state = torch.full((self.n, self.K), -1.0)
        self.used = torch.zeros((self.n, self.K), dtype=torch.bool)
        self.stage = torch.ones(self.n, dtype=torch.long)
        self.done = torch.zeros(self.n, dtype=torch.bool)
        self.pred = torch.full((self.n,), -1, dtype=torch.long)
        self.calls = torch.zeros(self.n, dtype=torch.long)
        self.elapsed = torch.zeros(self.n, dtype=torch.float32)
        return self.obs()

    def obs(self) -> torch.Tensor:
        """状态：β 向量（K 维）+ 一个阶段标志位。

        β 向量的前两位是小模型的概率、后三位是大模型的概率；当“两个小模型都调过、
        还没升级”和“刚升级、还没调大模型”时 β 完全相同，仅靠 β 无法区分这两
        种处境（前者还能出口判无毒，后者必须先调大模型）。因此额外拼接一维阶段
        标志，让策略和价值函数能判断“现在是不是已经进入第二阶段”。
        """
        stage_flag = (self.stage == 2).float().unsqueeze(1)
        return torch.cat([self.state, stage_flag], dim=1)

    @property
    def obs_dim(self) -> int:
        return self.K + 1

    def _stage_columns(self, i: int) -> List[int]:
        return self.stage1_idx if int(self.stage[i]) == 1 else self.stage2_idx

    def action_mask(self) -> torch.Tensor:
        """返回每条样本当前的合法动作掩码。

        第一阶段：只输出“无毒”。融合后的无毒概率达到阈值就允许出口；只要已经
        调用过至少一个模型，就允许随时升级到第二阶段（不再要求把第一阶段用满，
        这样策略才能学“什么时候跳过剩下的小模型”）。

        第二阶段：按同一个阈值做单侧判定——无毒概率达阈值判无毒，否则判有毒。
        """
        mask = torch.zeros((self.n, self.action_dim), dtype=torch.bool)
        for i in range(self.n):
            if self.done[i]:
                continue
            cols = self._stage_columns(i)
            for k in cols:
                if not self.used[i, k]:
                    mask[i, k] = True

            used_cols = [k for k in cols if bool(self.used[i, k])]
            if not used_cols:
                continue
            mean_toxic = float(torch.stack(
                [self.state[i, k] for k in used_cols]).mean())
            mean_nontoxic = 1.0 - mean_toxic

            if int(self.stage[i]) == 1:
                if mean_nontoxic >= self.threshold:
                    mask[i, self.K + 1] = True      # 融合概率可信 -> 判无毒出口
                mask[i, self.K] = True              # 调用过模型即可升级
            else:
                if mean_nontoxic >= self.threshold:
                    mask[i, self.K + 1] = True      # 单侧阈值 -> 判无毒
                else:
                    mask[i, self.K] = True          # 其余一律判有毒
        return mask

    # ------------------------------------------------------------- rewards ----
    # 终止奖励 = 分类奖励（对 +r，错 -iota*r）+ 本条样本的时间惩罚（-theta*t/u）
    def _terminal_reward(self, i: int, pred: int) -> float:
        label = int(self.cur_labels[i])
        r = self.cfg.correct_reward
        reward = r if pred == label else -self.cfg.penalty_factor * r
        reward += time_reward(float(self.elapsed[i]), self.cfg)
        return reward

    # 把某条样本标记为结束：结算奖励、记录预测值（每条只结算一次）
    def _terminate(self, i: int, pred: int, rewards: torch.Tensor) -> None:
        if self.done[i]:
            return
        rewards[i] += self._terminal_reward(i, pred)
        self.pred[i] = pred
        self.done[i] = True

    def _force_decision(self, i: int, rewards: torch.Tensor) -> None:
        """步数用尽时的兜底：按与掩码完全相同的单侧阈值规则判定。"""
        cols = [k for k in self._stage_columns(i) if bool(self.used[i, k])]
        if not cols:
            cols = [k for k in range(self.K) if bool(self.used[i, k])]
        if not cols:
            self._terminate(i, 0, rewards)
            return
        mean_nontoxic = 1.0 - float(self.cur_probs[i, cols].mean())
        self._terminate(i, 0 if mean_nontoxic >= self.threshold else 1, rewards)

    # ---------------------------------------------------------------- step ----
    # 执行一步：判定/升级类动作结算或切换阶段；模型类动作记录概率、调用数和延迟。
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.zeros(self.n)
        for i in range(self.n):
            if self.done[i]:
                continue
            action = int(actions[i])

            if action >= self.K:
                if int(self.stage[i]) == 1:
                    if action == self.K:          # "toxic" in stage 1 = escalate
                        self.stage[i] = 2
                    else:                          # confident nontoxic -> exit
                        self._terminate(i, 0, rewards)
                else:
                    self._terminate(i, 1 if action == self.K else 0, rewards)
                continue

            k = action
            self.state[i, k] = self.cur_probs[i, k]
            self.used[i, k] = True
            self.calls[i] += 1
            self.elapsed[i] += float(self.latencies[k])
            # 不再在“阶段用尽”时强制切换：升级与判定都交给策略按掩码决定，
            # 否则打完两个小模型就再也没有“出口判无毒”的机会。
        return rewards, self.done.clone()

    # 每条样本当前累计的延迟（秒），评估时用来算平均延迟和吞吐
    def sample_latency(self) -> torch.Tensor:
        return self.elapsed.clone()

    # 收尾：对达到 max_steps 仍没判定的样本强制做出判定
    def finalize(self) -> None:
        """Force a decision for any sample left undecided after ``max_steps``."""
        rewards = torch.zeros(self.n)
        for i in range(self.n):
            if not self.done[i]:
                self._force_decision(i, rewards)
