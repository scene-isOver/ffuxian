"""Frozen-classifier scoring, the two-stage PPO-CIS environment and static baselines.

PPO-CIS works on the outputs of the *frozen* classifiers, so we pre-compute one
toxicity probability per (sample, classifier) and cache it.  The RL part then
simulates the cascade without re-running the heavy transformer encoders.

Environment (two stages, following the paper's MDP):
    state   : beta vector of length K; called classifiers hold p(toxic),
              uncalled classifiers hold -1.
    actions : 0..K-1 = call classifier k,
              K = stage-1 escalate, K+1 = stage-1 nontoxic,
              K+2 = stage-2 toxic, K+3 = stage-2 nontoxic.
    stage 1 : 只输出“无毒”；融合概率达阈值即可出口，调用过任意模型后也可随时
              升级到第二阶段（策略自己决定什么时候跳过剩余小模型）。
    stage 2 : 按同一阈值单侧判定：无毒概率达阈值判无毒，否则判有毒。
    reward  : 逐样本时间惩罚（time_reward）+ 分类奖励 r / -iota*r
              + 可选的 Eq.(6) 吞吐奖励（默认权重 0，仅作报告指标）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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
    state = ckpt["state_dict"]
    # roberta_baseline.py 保存的是 HF 原始权重（key 不含 encoder. 前缀），
    # 而 build_classifier 返回的是 TransformerClassifier 包装，这里补一次前缀。
    if spec.kind == "bert" and not any(k.startswith("encoder.") for k in state):
        state = {f"encoder.{k}": v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt


# 用冻结的分类器给整个 split 打分，得到 (N, K) 的毒性概率矩阵；
# 结果缓存到 artifacts/scores/<split>.npz，模型名一致就复用（注意：不校验样本数）。
@torch.no_grad()
def score_split(specs: Sequence[ClassifierSpec], records: List[dict], split_name: str,
                device: torch.device, force: bool = False) -> Dict[str, np.ndarray]:
    """Return probs (N, K), labels (N,), latencies (K,), classifier names."""
    out_path = SCORE_DIR / f"{split_name}.npz"
    names = [spec.name for spec in specs]
    cur_labels = np.array([row["label"] for row in records], dtype=np.int64)
    if out_path.exists() and not force:
        data = np.load(out_path, allow_pickle=True)
        # 缓存必须同时匹配模型名和标签序列：换数据集/换切分后标签会变，
        # 只比对模型名会静默复用旧概率，导致评估数据与打印的划分对不上。
        same_names = list(data["names"]) == names
        same_labels = (data["labels"].shape == cur_labels.shape
                       and np.array_equal(data["labels"], cur_labels))
        if same_names and same_labels:
            return {k: data[k] for k in data.files}
        print(f"  [cache] {out_path.name} 与当前数据不一致，重新打分")

    probs = np.zeros((len(records), len(specs)), dtype=np.float32)
    latencies = np.zeros(len(specs), dtype=np.float32)
    labels = cur_labels

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

# 判定阈值写死为 0.7：出口与有毒/无毒的结论都由这条规则给出，
# 策略只负责学“调哪个模型、什么时候跳级”，阈值不参与学习。
FIXED_THRESHOLD = 0.7


class PPOEnv:
    """One episode per sample with a two-stage cascade.

    动作空间（K+4）：
        0..K-1  调用第 k 个分类器
        K       第一阶段：升级到第二阶段
        K+1     第一阶段：判定无毒并结束
        K+2     第二阶段：判定有毒
        K+3     第二阶段：判定无毒

    第一阶段的“升级”与第二阶段的“判定”使用不同的输出编号：两者语义不同
    （升级不是一次预测），共用一个编号会让策略把一个学到的偏好同时用在两处，
    导致第二阶段退化成常数判定。

    判定阈值固定为 0.7（``FIXED_THRESHOLD``）：有毒/无毒的结论由这条规则给出，
    策略只学习调用顺序与跳级时机，不学习阈值。
    """

    def __init__(self, probs: np.ndarray, labels: np.ndarray, latencies: np.ndarray,
                 reward_cfg: RewardConfig, stage1_idx: Sequence[int],
                 stage2_idx: Sequence[int], max_steps: int = 8,
                 threshold: float = FIXED_THRESHOLD,
                 stage2_priority: Optional[Sequence[int]] = None):
        self.probs = torch.as_tensor(probs, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.latencies = torch.as_tensor(latencies, dtype=torch.float32)
        self.cfg = reward_cfg
        self.K = self.probs.shape[1]
        # 四个判定动作的编号，见类文档字符串
        self.escalate_action = self.K
        self.exit_action = self.K + 1
        self.toxic_action = self.K + 2
        self.nontoxic_action = self.K + 3
        self.action_dim = self.K + 4
        self.stage1_idx = list(stage1_idx)
        self.stage2_idx = list(stage2_idx)
        self.max_steps = max_steps
        self.threshold = FIXED_THRESHOLD
        # Stage 2 判定优先级：rank 越小越优先，第一个是主判定模型。
        # 默认沿用 stage2_idx 的顺序，train_ppo 会按 --stage2_priority 传入。
        priority = list(stage2_priority) if stage2_priority else self.stage2_idx
        self.stage2_rank = {j: r for r, j in enumerate(priority)}
        # 主判定模型固定为 Stage 2 里优先级最高的那个（默认 bert），不随调用顺序变化。
        self.stage2_primary = (min(self.stage2_idx,
                                   key=lambda j: self.stage2_rank.get(j, len(priority)))
                               if self.stage2_idx else None)
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
        # 每条样本的终止奖励，供验证集按“平均奖励”选择 epoch。
        self.ep_reward = torch.zeros(self.n, dtype=torch.float32)
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

    # ===================== Stage 2 判定（主模型固定优先） =====================
    # 主判定模型固定为 stage2_priority 里排名最高的那个（默认 bert），与调用顺序无关：
    #   0) 主模型还没被调用 -> 不判定，逼策略先把主模型调出来；
    #   1) 主模型 p >= threshold      -> 判有毒；
    #   2) 主模型 p <= 1 - threshold  -> 判无毒；
    #   3) 落在灰区 (1-threshold, threshold) 时，与已调用的参考模型取平均，以 0.5 切开；
    #   4) 灰区且没有参考模型可参照 -> 暂不判定，让策略继续调用。
    # 返回 (pred, decided)；decided=False 表示当前信息不足以做结论。
    def _stage2_decision(self, i: int, used_cols: Sequence[int]) -> Tuple[int, bool]:
        primary = self.stage2_primary
        if primary is None or not bool(self.used[i, primary]):
            return 0, False
        p1 = float(self.state[i, primary])
        if p1 >= self.threshold:
            return 1, True
        if 1.0 - p1 >= self.threshold:
            return 0, True
        refs = sorted([j for j in used_cols
                       if j != primary and j in self.stage2_rank],
                      key=lambda j: self.stage2_rank[j])
        if refs:
            p2 = float(self.state[i, refs[0]])
            return (1 if 0.5 * (p1 + p2) >= 0.5 else 0), True
        if all(bool(self.used[i, j]) for j in self.stage2_idx):
            return (1 if p1 >= 0.5 else 0), True
        return 0, False

    def action_mask(self) -> torch.Tensor:
        """返回每条样本当前的合法动作掩码。

        阈值（0.7）写死，结论由规则给出，策略只决定调用顺序与跳级时机：

        第一阶段：只输出“无毒”。融合后的无毒概率达到阈值才允许出口；只要已经
        调用过至少一个模型，就允许随时升级到第二阶段（策略自己学什么时候跳过
        剩余的小模型）。
        第二阶段：主判定模型按阈值给出结论（灰区时参考第二优先级的模型），
        掩码只开放该结论对应的动作。
        两个阶段使用互不相同的判定编号（K+1 vs K+2/K+3），避免共用输出列。
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

            if int(self.stage[i]) == 1:
                mean_toxic = float(self.state[i, used_cols].mean())
                mean_nontoxic = 1.0 - mean_toxic
                if mean_nontoxic >= self.threshold:
                    mask[i, self.exit_action] = True      # 融合概率可信 -> 判无毒出口
                mask[i, self.escalate_action] = True      # 调用过模型即可升级
            else:
                # 第二阶段：由固定阈值规则给结论，策略只负责把模型调出来
                pred, decided = self._stage2_decision(i, used_cols)
                if decided:
                    mask[i, self.toxic_action if pred == 1 else self.nontoxic_action] = True
        return mask

    # ------------------------------------------------------------- rewards ----
    # 终止奖励 = 分类奖励（对 +r，错 -iota*r）+ 本条样本的时间惩罚（-theta*t/u）
    def _terminal_reward(self, i: int, pred: int) -> float:
        label = int(self.cur_labels[i])
        r = self.cfg.correct_reward
        if pred == label:
            reward = r
        else:
            # 漏判有毒（FN）比误判无毒（FP）危害更大：原文 Eq.(1) 允许对 FN
            # 单独加重惩罚，避免策略为了省一次调用而提前判无毒。
            factor = (self.cfg.fn_penalty_factor if label == 1
                      else self.cfg.penalty_factor)
            reward = -factor * r
        reward += time_reward(float(self.elapsed[i]), self.cfg)
        return reward

    # 把某条样本标记为结束：结算奖励、记录预测值（每条只结算一次）
    def _terminate(self, i: int, pred: int, rewards: torch.Tensor) -> None:
        if self.done[i]:
            return
        value = self._terminal_reward(i, pred)
        rewards[i] += value
        self.ep_reward[i] = value
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
        if int(self.stage[i]) == 2:
            pred, decided = self._stage2_decision(i, cols)
            if not decided:
                # 灰区且没有第二个模型可参照：优先用主模型自己的 0.5，保证一定有结果
                order = sorted([j for j in cols if j in self.stage2_rank],
                               key=lambda j: (j != self.stage2_primary,
                                              self.stage2_rank[j]))
                pred = 1 if (order and float(self.state[i, order[0]]) >= 0.5) else 0
            self._terminate(i, pred, rewards)
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
                if action == self.escalate_action:
                    self.stage[i] = 2
                elif action == self.exit_action:      # 第一阶段：判无毒并结束
                    self._terminate(i, 0, rewards)
                elif action == self.toxic_action:     # 第二阶段：判有毒
                    self._terminate(i, 1, rewards)
                else:                                 # 第二阶段：判无毒
                    self._terminate(i, 0, rewards)
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
