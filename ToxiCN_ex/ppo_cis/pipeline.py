"""Frozen-classifier scoring, the two-stage PPO-CIS environment and static baselines.

PPO-CIS works on the outputs of the *frozen* classifiers, so we pre-compute one
toxicity probability per (sample, classifier) and cache it.  The RL part then
simulates the cascade without re-running the heavy transformer encoders.

Environment (two stages, following the paper's MDP):
    state   : beta vector of length K; called classifiers hold p(toxic),
              uncalled classifiers hold -1.
    actions : 0..K-1 = call classifier k, K = "toxic", K+1 = "nontoxic".
    stage 1 : only a high-confidence "nontoxic" exits.  "toxic" (or exhausting
              all stage-1 classifiers) escalates the sample to stage 2.
    stage 2 : "toxic" / "nontoxic" terminate the episode.
    reward  : time reward Eq.(2) on the episode's total latency, classification
              reward r / -iota*r, plus a batch throughput bonus Eq.(6).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CLASSIFIER_DIR, ClassifierSpec, RewardConfig, SCORE_DIR, StaticConfig
from .models import build_classifier
from .data import EncodedDataset, encode_split, get_tokenizer
from .utils import compute_metrics, forward_classifier


# ---------------------------------------------------------------- scoring -----

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

def _metrics(preds: np.ndarray, labels: np.ndarray, latencies: np.ndarray,
             calls: np.ndarray) -> Dict[str, float]:
    metrics = compute_metrics(preds, labels)
    metrics["avg_classifiers"] = float(np.mean(calls))
    metrics["avg_latency_ms"] = float(np.mean(latencies) * 1000)
    metrics["throughput"] = float(len(labels) / max(np.sum(latencies), 1e-9))
    return metrics


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
    """Eq. (2): reward for the episode's total processing time ``t``.

    ``phi + theta * log2(u - t)`` when the episode finishes inside the budget,
    and a monotonically decreasing penalty when it exceeds ``u``.  The paper's
    second branch ``phi + theta * log2(t - u)`` grows with ``t`` (a slower
    episode would be rewarded more), so the sign is flipped here to keep the
    reward monotone; everything else follows Eq. (2).
    """
    u = cfg.time_budget
    if total_latency_sec < u:
        return cfg.time_phi + cfg.time_theta * math.log2(
            max(u - total_latency_sec, cfg.eps))
    return cfg.time_phi - cfg.time_theta * math.log2(
        max(total_latency_sec - u, 0.0) + 1.0)


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
        # beta vector: p(toxic) for called classifiers, -1 for the others
        return self.state.clone()

    def _stage_columns(self, i: int) -> List[int]:
        return self.stage1_idx if int(self.stage[i]) == 1 else self.stage2_idx

    def _stage_exhausted(self, i: int) -> bool:
        cols = self._stage_columns(i)
        return all(bool(self.used[i, j]) for j in cols)

    def action_mask(self) -> torch.Tensor:
        """Legal actions for every sample.

        Stage 1: a high-confidence nontoxic exits; a high-confidence toxic (or
        exhausting stage 1) escalates.  Stage 2: a high-confidence toxic /
        nontoxic terminates; exhausting stage 2 forces a decision.
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
            used_probs = torch.stack([self.state[i, k] for k in used_cols])
            best_toxic = float(used_probs.max())
            best_nontoxic = 1.0 - float(used_probs.min())

            if int(self.stage[i]) == 1:
                if best_nontoxic >= self.threshold:
                    mask[i, self.K + 1] = True
                if best_toxic >= self.threshold or self._stage_exhausted(i):
                    mask[i, self.K] = True
            else:
                if best_toxic >= self.threshold or self._stage_exhausted(i):
                    mask[i, self.K] = True
                if best_nontoxic >= self.threshold or self._stage_exhausted(i):
                    mask[i, self.K + 1] = True

            if not mask[i].any():      # safety net: never leave a sample stuck
                mask[i, self.K] = True
                mask[i, self.K + 1] = True
        return mask

    # ------------------------------------------------------------- rewards ----
    def _terminal_reward(self, i: int, pred: int) -> float:
        label = int(self.cur_labels[i])
        r = self.cfg.correct_reward
        reward = r if pred == label else -self.cfg.penalty_factor * r
        reward += time_reward(float(self.elapsed[i]), self.cfg)
        return reward

    def _terminate(self, i: int, pred: int, rewards: torch.Tensor) -> None:
        if self.done[i]:
            return
        rewards[i] += self._terminal_reward(i, pred)
        self.pred[i] = pred
        self.done[i] = True

    def _force_decision(self, i: int, rewards: torch.Tensor) -> None:
        cols = [k for k in self._stage_columns(i) if bool(self.used[i, k])]
        if not cols:
            cols = [k for k in range(self.K) if bool(self.used[i, k])]
        mean_p = float(self.cur_probs[i, cols].mean()) if cols else 0.5
        self._terminate(i, int(mean_p >= 0.5), rewards)

    # ---------------------------------------------------------------- step ----
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

            if self._stage_exhausted(i):
                if int(self.stage[i]) == 1:
                    self.stage[i] = 2              # no stage-1 model left -> escalate
                else:
                    self._force_decision(i, rewards)
        return rewards, self.done.clone()

    def sample_latency(self) -> torch.Tensor:
        return self.elapsed.clone()

    def finalize(self) -> None:
        """Force a decision for any sample left undecided after ``max_steps``."""
        rewards = torch.zeros(self.n)
        for i in range(self.n):
            if not self.done[i]:
                self._force_decision(i, rewards)
