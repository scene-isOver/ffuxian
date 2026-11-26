"""四个外部级联/路由基线（只读 scores/*.npz，不重训任何分类器）。

对应论文与落地方式：
    RouteLLM (ICLR 2025)            监督路由器：用 ppo_train 生成“奖励最优动作”
                                    标签，训练逻辑回归做行为克隆（sequential 版）；
                                    binary 版受限于没有文本特征，退化为“先调最小
                                    模型，再决定是否改调 roberta_plain”。
    AutoMix (NeurIPS 2024)          POMDP：用 ppo_train 估计“出口判对概率”和
                                    “升级判对概率”，再做子集动态规划求最优策略。
    Semantic Agreement (EMNLP 2025) 训练无关：阶段一小模型投票一致率达标就出口。
    Conformal L2D (TMLR 2026)       split-conformal：在 ppo_dev 上校准预测集，
                                    集合不唯一就升级到阶段二。

所有基线共用同一个 PPOEnv，因此动作掩码、阶段二判定规则、延迟累计、
终止奖励的口径与 PPO-CIS 完全一致，结果可直接放进同一张表。

全部超参只在 ppo_dev 上按“平均回合奖励”选择，test 只报告一次。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RewardConfig
from .pipeline import PPOEnv, time_reward


# ===================== 公共工具 =====================

@dataclass
class CascadeCtx:
    """把阶段划分、阈值、延迟等冻结参数打包，供所有基线构造环境。"""
    latencies: np.ndarray
    cfg: RewardConfig
    stage1: List[int]
    stage2: List[int]
    max_steps: int
    threshold: float
    priority: List[int]

    @property
    def kappa(self) -> float:
        """每秒钟的时间惩罚系数 theta/u，与奖励函数保持一致。"""
        return self.cfg.time_theta / max(self.cfg.time_budget, self.cfg.eps)

    def env(self, probs: np.ndarray, labels: np.ndarray) -> PPOEnv:
        return PPOEnv(probs, labels, self.latencies, self.cfg,
                      stage1_idx=self.stage1, stage2_idx=self.stage2,
                      max_steps=self.max_steps, threshold=self.threshold,
                      stage2_priority=self.priority)

    @property
    def primary(self) -> int:
        rank = {j: r for r, j in enumerate(self.priority)}
        return min(self.stage2, key=lambda j: rank.get(j, len(rank)))


# 终止奖励：判对 +r，判错 -iota*r，再减去累计延迟惩罚。
def terminal_reward(pred: int, label: int, elapsed: float,
                    cfg: RewardConfig) -> float:
    r = cfg.correct_reward
    base = r if pred == label else -cfg.penalty_factor * r
    return base + time_reward(elapsed, cfg)


# 期望终止奖励：已知“判对概率 q”时的奖励期望，用于 POMDP 的 DP。
def expected_reward(q: float, elapsed: float, cfg: RewardConfig) -> float:
    r = cfg.correct_reward
    return q * r + (1.0 - q) * (-cfg.penalty_factor * r) + time_reward(elapsed, cfg)


# 复刻 PPOEnv._stage2_decision：主模型优先，灰区找第二意见，全调用则退回 0.5。
def stage2_prediction(row: np.ndarray, used: Sequence[int],
                      stage2: Sequence[int], rank: Dict[int, int],
                      threshold: float) -> Optional[int]:
    primary = min(stage2, key=lambda j: rank.get(j, len(rank)))
    if primary not in used:
        return None
    p1 = float(row[primary])
    if p1 >= threshold:
        return 1
    if 1.0 - p1 >= threshold:
        return 0
    refs = sorted([j for j in used if j != primary], key=lambda j: rank.get(j, len(rank)))
    if refs:
        return 1 if 0.5 * (p1 + float(row[refs[0]])) >= 0.5 else 0
    if all(j in used for j in stage2):
        return 1 if p1 >= 0.5 else 0
    return None


# 复刻 PPOEnv.action_mask：返回当前状态下所有合法动作。
def legal_actions(row: np.ndarray, used: Sequence[int], stage: int,
                  ctx: CascadeCtx) -> List[int]:
    k = len(row)
    cols = ctx.stage1 if stage == 1 else ctx.stage2
    legal = [j for j in cols if j not in used]
    used_cols = [j for j in cols if j in used]
    if not used_cols:
        return legal
    if stage == 1:
        mean_nontoxic = 1.0 - float(np.mean([row[j] for j in used_cols]))
        if mean_nontoxic >= ctx.threshold:
            legal.append(k + 1)
        legal.append(k)
    else:
        pred = stage2_prediction(row, used, ctx.stage2,
                                 {j: r for r, j in enumerate(ctx.priority)},
                                 ctx.threshold)
        if pred is not None:
            # 第二阶段判定使用独立编号：k+2 = 有毒，k+3 = 无毒
            legal.append(k + 2 if pred == 1 else k + 3)
    return legal


# 状态观测：未调用的模型填 -1，末尾拼阶段标志，与 PPOEnv.obs 完全一致。
def state_obs(row: np.ndarray, used: Sequence[int], stage: int) -> np.ndarray:
    obs = np.full(len(row), -1.0, dtype=np.float32)
    for j in used:
        obs[j] = float(row[j])
    return np.concatenate([obs, [1.0 if stage == 2 else 0.0]])


# 统一 rollout：由 choose_fn 决定每一步动作，其余全部交给 PPOEnv。
def rollout(env: PPOEnv, choose_fn) -> Dict[str, np.ndarray]:
    n = env.probs.shape[0]
    env.reset(torch.arange(n))
    for _ in range(env.max_steps):
        masks = env.action_mask()
        if not bool(masks.any()):
            break
        env.step(choose_fn(env, masks))
    env.finalize()
    return {
        "preds": env.pred.numpy(),
        "labels": env.cur_labels.numpy(),
        "latency": env.sample_latency().numpy(),
        "calls": env.calls.numpy().astype(float),
        "rewards": env.ep_reward.numpy(),
    }


def _first_legal(mask_row: np.ndarray, prefer: Sequence[int]) -> int:
    for a in prefer:
        if bool(mask_row[a]):
            return int(a)
    idx = np.flatnonzero(mask_row)
    return int(idx[0]) if idx.size else 0


class _ConstProba:
    """训练集只有单一类别时的兜底“模型”，避免 sklearn 直接报错。"""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, x):
        n = len(x)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def _fit_binary(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return _ConstProba(float(y.mean()) if len(y) else 0.0)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(x, y)
    return model


# ===================== 1) Semantic Agreement =====================

class SemanticAgreement:
    """阶段一小模型投票一致率 >= tau 就出口，否则升级。

    原文：Semantic Agreement Enables Efficient Open-Ended LLM Cascades
          (EMNLP 2025 Industry)。原方法比较生成文本的语义等价，这里改成
          比分类器二值判决的一致率，属于 adapted。
    """

    def __init__(self, tau: float = 0.75):
        self.tau = tau

    def fit(self, probs, labels, ctx: CascadeCtx) -> Dict[str, float]:
        best_tau, best_reward = self.tau, -1e18
        history = {}
        for tau in np.round(np.arange(0.50, 1.0001, 0.05), 2):
            self.tau = float(tau)
            env = ctx.env(probs, labels)
            out = rollout(env, self.choose)
            mean_reward = float(out["rewards"].mean())
            history[f"{tau:.2f}"] = mean_reward
            if mean_reward > best_reward:
                best_reward, best_tau = mean_reward, float(tau)
        self.tau = best_tau
        return {"tau": best_tau, "dev_reward": best_reward, "sweep": history}

    def choose(self, env: PPOEnv, masks: torch.Tensor) -> torch.Tensor:
        m = masks.numpy()
        actions = np.zeros(env.n, dtype=np.int64)
        k = env.K
        for i in range(env.n):
            if not bool(m[i].any()):
                continue
            stage = int(env.stage[i])
            if stage == 2:
                order = sorted(env.stage2_idx,
                               key=lambda j: env.stage2_rank.get(j, 99))
                actions[i] = _first_legal(m[i], [k + 2, k + 3] + order)
                continue
            used1 = [j for j in env.stage1_idx if bool(env.used[i, j])]
            if used1:
                votes = [1 if float(env.state[i, j]) >= 0.5 else 0 for j in used1]
                major = 1 if sum(votes) >= int(np.ceil(len(votes) / 2)) else 0
                agree = float(np.mean([v == major for v in votes]))
                if agree >= self.tau and bool(m[i, k + 1]):
                    actions[i] = k + 1          # 一致 -> 判无毒出口
                    continue
            uncalled = [j for j in env.stage1_idx if not bool(env.used[i, j])]
            if uncalled:
                actions[i] = _first_legal(m[i], uncalled)
            else:
                actions[i] = _first_legal(m[i], [k])
        return torch.as_tensor(actions)


# ===================== 2) Conformal L2D =====================

class ConformalDeferral:
    """split-conformal 推迟：预测集不唯一就升级。

    原文：Learning to Defer with an Uncertain Rejector via Conformal Prediction
          (TMLR 2026)。原文的“专家”是人类，这里换成阶段二分类器，属于 adapted；
    alpha 固定 0.1，不参与调参，保证校准集的统计含义不被破坏。
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q: Optional[float] = None
        self.rejector: Optional[int] = None

    def fit(self, probs, labels, ctx: CascadeCtx) -> Dict[str, float]:
        # 用最便宜的小模型做 rejector
        self.rejector = int(min(ctx.stage1, key=lambda j: ctx.latencies[j]))
        p = probs[:, self.rejector].astype(np.float64)
        y = labels.astype(np.int64)
        scores = 1.0 - np.where(y == 1, p, 1.0 - p)
        n = len(scores)
        rank = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        rank = min(max(rank, 1), n)
        self.q = float(np.sort(scores)[rank - 1])
        return {"alpha": self.alpha, "q": self.q, "rejector": self.rejector,
                "calib_n": int(n), "rank": rank}

    def _in_set(self, p_toxic: float, cls: int) -> bool:
        score = 1.0 - p_toxic if cls == 1 else p_toxic
        return score <= self.q + 1e-12

    def choose(self, env: PPOEnv, masks: torch.Tensor) -> torch.Tensor:
        m = masks.numpy()
        actions = np.zeros(env.n, dtype=np.int64)
        k = env.K
        for i in range(env.n):
            if not bool(m[i].any()):
                continue
            stage = int(env.stage[i])
            if stage == 2:
                order = sorted(env.stage2_idx,
                               key=lambda j: env.stage2_rank.get(j, 99))
                actions[i] = _first_legal(m[i], [k + 2, k + 3] + order)
                continue
            if not bool(env.used[i, self.rejector]):
                actions[i] = _first_legal(m[i], [self.rejector])
                continue
            p = float(env.state[i, self.rejector])
            in1, in0 = self._in_set(p, 1), self._in_set(p, 0)
            if in1 and in0:
                actions[i] = _first_legal(m[i], [k])            # 集合不唯一 -> 升级
            elif in0:
                actions[i] = _first_legal(m[i], [k + 1, k])      # 唯一判无毒
            else:
                actions[i] = _first_legal(m[i], [k])             # 阶段一不能判有毒
        return torch.as_tensor(actions)


# ===================== 3) AutoMix（POMDP + 子集 DP） =====================

class AutoMixPOMDP:
    """用 ppo_train 学两个“判对概率”模型，再做子集动态规划。

    原文：AutoMix: Automatically Mixing Language Models (NeurIPS 2024)。
    原文用 LLM 自验证当观测，这里用分类器概率当观测，属于 adapted。
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.exit_model = None
        self.dec_model = None
        self._ctx: Optional[CascadeCtx] = None

    # ---- 特征 ----
    @staticmethod
    def _exit_feat(row, used, ctx: CascadeCtx) -> Optional[List[float]]:
        ps = [float(row[j]) for j in ctx.stage1 if j in used]
        if not ps:
            return None
        return [float(np.mean(ps)), float(np.min(ps)), float(np.max(ps)),
                float(np.std(ps)), len(ps) / max(len(ctx.stage1), 1)]

    @staticmethod
    def _dec_feat(row, used, ctx: CascadeCtx) -> Optional[List[float]]:
        rank = {j: r for r, j in enumerate(ctx.priority)}
        primary = ctx.primary
        if primary not in used:
            return None
        pred = stage2_prediction(row, used, ctx.stage2, rank, ctx.threshold)
        if pred is None:
            return None
        p1 = float(row[primary])
        refs = sorted([j for j in used if j != primary and j in rank],
                      key=lambda j: rank[j])
        p2 = float(row[refs[0]]) if refs else -1.0
        ps = [float(row[j]) for j in used]
        # 末尾带上“判定方向”，否则 P(正确) 会把判有毒和判无毒混在一起估。
        return [p1, p2, 1.0 if refs else 0.0,
                float(np.mean(ps)), float(np.min(ps)), float(np.max(ps)),
                1.0 if pred == 1 else 0.0]

    def _stage1_states(self, ctx: CascadeCtx):
        for r in range(1, len(ctx.stage1) + 1):
            for combo in itertools.combinations(ctx.stage1, r):
                yield tuple(combo)

    def _stage2_states(self, ctx: CascadeCtx):
        base = list(self._stage1_states(ctx))
        subsets = [()]
        for r in range(1, len(ctx.stage2) + 1):
            subsets += list(itertools.combinations(ctx.stage2, r))
        for b in base:
            for s in subsets:
                yield tuple(b) + tuple(s)

    def fit(self, probs, labels, ctx: CascadeCtx) -> Dict[str, float]:
        self._ctx = ctx
        # 出口模型：P(判无毒正确 | 阶段一已调模型的概率统计)
        fx, fy = [], []
        for i in range(len(labels)):
            for used in self._stage1_states(ctx):
                feat = self._exit_feat(probs[i], used, ctx)
                if feat is None:
                    continue
                fx.append(feat)
                fy.append(1 if int(labels[i]) == 0 else 0)
        # 升级判定模型：P(阶段二判决正确 | 主模型与参考模型概率)
        dx, dy = [], []
        rank = {j: r for r, j in enumerate(ctx.priority)}
        for i in range(len(labels)):
            for used in self._stage2_states(ctx):
                feat = self._dec_feat(probs[i], used, ctx)
                if feat is None:
                    continue
                pred = stage2_prediction(probs[i], used, ctx.stage2, rank, ctx.threshold)
                if pred is None:
                    continue
                dx.append(feat)
                dy.append(1 if pred == int(labels[i]) else 0)
        self.exit_model = _fit_binary(fx, fy)
        self.dec_model = _fit_binary(dx, dy)
        return {"exit_samples": int(len(fy)), "dec_samples": int(len(dy))}

    def _q_exit(self, feat) -> float:
        return float(self.exit_model.predict_proba([feat])[0, 1])

    def _q_dec(self, feat) -> float:
        return float(self.dec_model.predict_proba([feat])[0, 1])

    def choose(self, env: PPOEnv, masks: torch.Tensor) -> torch.Tensor:
        m = masks.numpy()
        actions = np.zeros(env.n, dtype=np.int64)
        k = env.K
        for i in range(env.n):
            if not bool(m[i].any()):
                continue
            row = env.cur_probs[i].numpy()
            used = tuple(j for j in range(env.K) if bool(env.used[i, j]))
            stage = int(env.stage[i])
            memo: Dict[Tuple[Tuple[int, ...], int], Tuple[float, Optional[int]]] = {}
            _, action = self._value(row, used, stage, env, memo)
            if action is None or not bool(m[i, action]):
                action = _first_legal(m[i], list(range(env.K)))
            actions[i] = int(action)
        return torch.as_tensor(actions)

    def _value(self, row, used, stage, env, memo):
        key = (tuple(sorted(used)), stage)
        if key in memo:
            return memo[key]
        elapsed = float(sum(float(env.latencies[j]) for j in used))
        best_val, best_act = -1e18, None
        for a in legal_actions(row, used, stage, self._ctx):
            if a >= env.K:
                if stage == 1 and a == env.K:
                    val = self._value(row, used, 2, env, memo)[0]
                elif stage == 1:
                    feat = self._exit_feat(row, used, self._ctx)
                    val = expected_reward(self._q_exit(feat), elapsed, self._ctx.cfg)
                else:
                    feat = self._dec_feat(row, used, self._ctx)
                    val = expected_reward(self._q_dec(feat), elapsed, self._ctx.cfg)
            else:
                val = self._value(row, tuple(sorted(used + (a,))), stage, env, memo)[0]
            if val > best_val + 1e-12:
                best_val, best_act = val, a
        memo[key] = (best_val, best_act)
        return best_val, best_act


# ===================== 4) RouteLLM（监督行为克隆） =====================

class RouteLLMSequential:
    """用“奖励最优动作”做标签训练路由器，推理时逐步选择动作。

    原文：RouteLLM: Learning to Route LLMs with Preference Data (ICLR 2025)。
    原文是弱/强模型二选一；这里扩展到多步级联，属于 adapted。
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.clf = None
        self.const_action: Optional[int] = None

    # 用真实标签做 oracle DP，得到每个状态的“奖励最优动作”
    def _oracle_action(self, row, label, used, stage, env, ctx: CascadeCtx, memo):
        key = (tuple(sorted(used)), stage)
        if key in memo:
            return memo[key]
        elapsed = float(sum(float(ctx.latencies[j]) for j in used))
        best_val, best_act = -1e18, None
        for a in legal_actions(row, used, stage, ctx):
            if a >= env.K:
                if stage == 1 and a == env.K:
                    val = self._oracle_action(row, label, used, 2, env, ctx, memo)[0]
                elif stage == 1:
                    val = terminal_reward(0, label, elapsed, ctx.cfg)
                else:
                    pred = stage2_prediction(
                        row, used, ctx.stage2,
                        {j: r for r, j in enumerate(ctx.priority)}, ctx.threshold)
                    val = terminal_reward(int(pred), label, elapsed, ctx.cfg)
            else:
                val = self._oracle_action(row, label, tuple(sorted(used + (a,))),
                                          stage, env, ctx, memo)[0]
            if val > best_val + 1e-12:
                best_val, best_act = val, a
        memo[key] = (best_val, best_act)
        return best_val, best_act

    def fit(self, probs, labels, ctx: CascadeCtx, max_per_state: int = 40000):
        env = ctx.env(probs, labels)
        xs, ys = [], []
        for i in range(len(labels)):
            row = probs[i].astype(np.float64)
            memo = {}
            self._oracle_action(row, int(labels[i]), (), 1, env, ctx, memo)
            for (used, stage), (_, act) in memo.items():
                if act is None:
                    continue
                xs.append(state_obs(row, used, stage))
                ys.append(int(act))
                if len(xs) >= max_per_state:
                    break
            if len(xs) >= max_per_state:
                break
        self.clf = None
        self.const_action: Optional[int] = None
        if len(np.unique(ys)) < 2:
            self.const_action = int(ys[0]) if ys else 0
            return {"train_states": int(len(xs)), "const_action": self.const_action}
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"))
        self.clf.fit(np.asarray(xs), np.asarray(ys))
        return {"train_states": int(len(xs)), "classes": [int(c) for c in self.clf.classes_]}

    def choose(self, env: PPOEnv, masks: torch.Tensor) -> torch.Tensor:
        m = masks.numpy()
        actions = np.zeros(env.n, dtype=np.int64)
        if self.clf is None:                     # 训练集只有单一动作时的兜底
            prefer = [self.const_action] if self.const_action is not None else []
            for i in range(env.n):
                if bool(m[i].any()):
                    actions[i] = _first_legal(m[i], prefer)
            return torch.as_tensor(actions)
        obs = env.obs().numpy()
        proba = self.clf.predict_proba(obs)
        classes = [int(c) for c in self.clf.classes_]
        actions = np.zeros(env.n, dtype=np.int64)
        for i in range(env.n):
            if not bool(m[i].any()):
                continue
            order = np.argsort(-proba[i])
            picked = None
            for idx in order:
                a = classes[int(idx)]
                if a < len(m[i]) and bool(m[i, a]):
                    picked = a
                    break
            actions[i] = int(picked) if picked is not None else _first_legal(m[i], [])
        return torch.as_tensor(actions)


class RouteLLMBinary:
    """严格二选一的可行近似：先调最小模型，再决定“就地判定 / 改调 roberta_plain”。

    真正的二选一路由需要文本特征，而 scores/*.npz 里只有概率，因此这里退化为
    “看过最小模型输出后再二选一”，属于 adapted；论文里应标注这一点。
    """

    def __init__(self, threshold: float = 0.5, seed: int = 42):
        self.threshold = threshold
        self.seed = seed
        self.clf = None
        self.tiny: Optional[int] = None
        self.strong: Optional[int] = None

    @staticmethod
    def _feat(p: float) -> List[float]:
        p = float(p)
        ent = -(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9))
        return [p, 1.0 - p, float(ent), abs(p - 0.5)]

    def fit(self, probs, labels, probs_dev, labels_dev, ctx: CascadeCtx):
        self.tiny = int(min(ctx.stage1, key=lambda j: ctx.latencies[j]))
        self.strong = int(ctx.primary)
        p_tiny = probs[:, self.tiny]
        tiny_correct = ((p_tiny >= 0.5).astype(int) == labels.astype(int))
        p_strong = probs[:, self.strong]
        strong_correct = ((p_strong >= 0.5).astype(int) == labels.astype(int))
        # 只在“小模型错、大模型对”时才值得改调大模型
        y = ((~tiny_correct) & strong_correct).astype(int)
        x = np.asarray([self._feat(p) for p in p_tiny])
        self.clf = _fit_binary(x, y)

        best_thr, best_reward, history = self.threshold, -1e18, {}
        for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
            self.threshold = float(thr)
            env = ctx.env(probs_dev, labels_dev)
            out = rollout(env, self.choose)
            mean_reward = float(out["rewards"].mean())
            history[f"{thr:.2f}"] = mean_reward
            if mean_reward > best_reward:
                best_reward, best_thr = mean_reward, float(thr)
        self.threshold = best_thr
        return {"threshold": best_thr, "dev_reward": best_reward,
                "tiny": self.tiny, "strong": self.strong, "sweep": history}

    def choose(self, env: PPOEnv, masks: torch.Tensor) -> torch.Tensor:
        m = masks.numpy()
        actions = np.zeros(env.n, dtype=np.int64)
        k = env.K
        for i in range(env.n):
            if not bool(m[i].any()):
                continue
            if int(env.stage[i]) == 2:
                order = sorted(env.stage2_idx,
                               key=lambda j: env.stage2_rank.get(j, 99))
                actions[i] = _first_legal(m[i], [k + 2, k + 3] + order)
                continue
            if not bool(env.used[i, self.tiny]):
                actions[i] = _first_legal(m[i], [self.tiny])
                continue
            p = float(env.state[i, self.tiny])
            goto_strong = float(self.clf.predict_proba([self._feat(p)])[0, 1]) >= self.threshold
            if goto_strong:
                actions[i] = _first_legal(m[i], [k])
            else:
                actions[i] = _first_legal(m[i], [k + 1, k])
        return torch.as_tensor(actions)
