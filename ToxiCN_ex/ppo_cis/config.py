"""PPO-CIS configuration: two-stage cascade over five ToxiCN classifiers.

Reference paper:
    Bodaghi, Fung & Schmitt, "PPO-CIS: A Deep Reinforcement Learning Framework
    for Real-Time Toxicity Detection in Social Media", Knowledge-Based Systems, 2026.

Layout (as requested):
    * Stage 1 (high throughput): two small variants, BERT-Tiny and DistilRoBERTa.
    * Stage 2 (high accuracy): the three base classifiers already in this folder.

Stage 1 only performs the "high-confidence nontoxic -> exit" decision; anything
else escalates to Stage 2, which makes the final toxic / nontoxic decision.
This mirrors the paper's cascade and keeps the maximum number of calls at
``len(stage1) + len(stage2)``.

Classifiers:
    tiny_bert        <- BertForSequenceClassification on ckiplab/bert-tiny-chinese
    distil_roberta   <- AutoModelForSequenceClassification on a distilled Chinese RoBERTa
    bert             <- bert-safe-chat/train.py (BertForSequenceClassification)
    roberta_fusion   <- CHSD/model.py :: RoBertFusion (RoBERTa + BiGRU + TextCNN)
    tke_bert         <- src/BERT.py + src/Models.py (BERT + toxic/stance knowledge
                        embeddings + two-layer FFNN head)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
TOXICN_EX_DIR = PACKAGE_DIR.parent
DATA_DIR = TOXICN_EX_DIR / "ToxiCN" / "data"
LEXICON_DIR = TOXICN_EX_DIR / "ToxiCN" / "lexicon"

ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
CLASSIFIER_DIR = ARTIFACT_DIR / "classifiers"
SCORE_DIR = ARTIFACT_DIR / "scores"
PPO_DIR = ARTIFACT_DIR / "ppo"
CACHE_DIR = ARTIFACT_DIR / "cache"
REPORT_DIR = ARTIFACT_DIR / "reports"

# Local model directories.  Replace MODEL_ROOT with the folder holding the four
# downloaded checkpoints.  Each model must sit in a sub-folder named like the
# last part of its HuggingFace id:
#
#   <MODEL_ROOT>/bert-base-chinese/          (bert + tke_bert share this one)
#   <MODEL_ROOT>/chinese-roberta-wwm-ext/    (roberta_fusion)
#   <MODEL_ROOT>/bert-tiny-chinese/          (tiny_bert, stage 1)
#   <MODEL_ROOT>/chinese_roberta_L-4_H-256/  (distil_roberta, stage 1)
#
# The five classifiers need only four pretrained models: ``tke_bert`` adds the
# toxic/stance knowledge embeddings and the FFNN head on top of bert-base-chinese.
#
# The placeholder below is deliberately fake.  Set the environment variable
# PPO_CIS_MODEL_ROOT (or edit this line) before running, or pass explicit paths
# through --bert_model / --roberta_model / --tiny_bert_model / --distil_roberta_model.
MODEL_ROOT = Path(os.environ.get("PPO_CIS_MODEL_ROOT", "D:/models/ppo_cis_models"))

BERT_MODEL_NAME = os.environ.get(
    "PPO_CIS_BERT_MODEL", str(MODEL_ROOT / "bert-base-chinese"))
ROBERTA_MODEL_NAME = os.environ.get(
    "PPO_CIS_ROBERTA_MODEL", str(MODEL_ROOT / "chinese-roberta-wwm-ext"))
TINY_BERT_MODEL_NAME = os.environ.get(
    "PPO_CIS_TINY_BERT_MODEL", str(MODEL_ROOT / "bert-tiny-chinese"))
DISTIL_ROBERTA_MODEL_NAME = os.environ.get(
    "PPO_CIS_DISTIL_ROBERTA_MODEL", str(MODEL_ROOT / "chinese_roberta_L-4_H-256"))

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "tf_model.h5",
                 "flax_model.msgpack")


def resolve_model_name(name: str) -> str:
    """Map a model name to a local directory, never to the HuggingFace hub.

    ``name`` must be an existing directory or a bare ``hfl/chinese-roberta-wwm-ext``
    (resolved to ``MODEL_ROOT/chinese-roberta-wwm-ext``).  Downloading is
    disabled: a missing or incomplete directory raises ``FileNotFoundError``.
    """
    path = Path(name).expanduser()
    candidate = path if path.is_dir() else MODEL_ROOT / path.name
    if not candidate.is_dir():
        tried = [str(path)] if str(path) != str(candidate) else []
        tried.append(str(candidate))
        raise FileNotFoundError(
            f"Local model directory not found for '{name}'.\n"
            + "".join(f"  tried: {p}\n" for p in tried)
            + "  Pass --bert_model / --roberta_model / --tiny_bert_model / "
            "--distil_roberta_model, or set PPO_CIS_MODEL_ROOT.\n"
            "  Downloading models is disabled."
        )
    if not (candidate / "config.json").is_file():
        raise FileNotFoundError(
            f"{candidate} is not a HuggingFace model directory: config.json missing")
    if not any((candidate / f).is_file() for f in _WEIGHT_FILES):
        raise FileNotFoundError(
            f"{candidate} has no model weights "
            f"({' / '.join(_WEIGHT_FILES)}); downloading models is disabled")
    return str(candidate)


@dataclass
class DataConfig:
    train_json: Path = DATA_DIR / "train.json"
    test_json: Path = DATA_DIR / "test.json"
    max_len: int = 96
    val_ratio: float = 0.10        # split off from train.json for classifier validation
    ppo_ratio: float = 0.10        # split off from train.json for PPO training
    seed: int = 42
    min_chars: int = 2


@dataclass
class ClassifierSpec:
    name: str
    kind: str                      # bert | roberta_fusion | tke_bert
    stage: int = 2                 # 1 = high-throughput, 2 = high-accuracy
    model_name: str = ""           # transformer backbone name or local path
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    dropout: float = 0.3
    freeze_layers: int = 0
    patience: int = 1
    use_knowledge: bool = False    # TKE-BERT: feed toxic_ids / stance_ids


def default_classifier_specs(bert_model: Optional[str] = None,
                             roberta_model: Optional[str] = None,
                             tiny_bert_model: Optional[str] = None,
                             distil_roberta_model: Optional[str] = None
                             ) -> List[ClassifierSpec]:
    """The five classifiers used by the two-stage cascade."""
    bert = bert_model or BERT_MODEL_NAME
    roberta = roberta_model or ROBERTA_MODEL_NAME
    tiny = tiny_bert_model or TINY_BERT_MODEL_NAME
    distil = distil_roberta_model or DISTIL_ROBERTA_MODEL_NAME
    return [
        # ---------------------------------------------------------- stage 1 (throughput)
        ClassifierSpec(
            name="tiny_bert", kind="bert", stage=1, model_name=tiny,
            epochs=4, batch_size=64, lr=3e-5, weight_decay=0.01,
            dropout=0.1, freeze_layers=0, patience=1,
        ),
        ClassifierSpec(
            name="distil_roberta", kind="bert", stage=1, model_name=distil,
            epochs=4, batch_size=64, lr=3e-5, weight_decay=0.01,
            dropout=0.1, freeze_layers=0, patience=1,
        ),
        # ---------------------------------------------------------- stage 2 (accuracy)
        ClassifierSpec(
            name="bert", kind="bert", stage=2, model_name=bert,
            epochs=3, batch_size=32, lr=2e-5, weight_decay=0.01,
            dropout=0.1, freeze_layers=0, patience=1,
        ),
        ClassifierSpec(
            name="roberta_fusion", kind="roberta_fusion", stage=2, model_name=roberta,
            epochs=3, batch_size=32, lr=1e-5, weight_decay=0.001,
            dropout=0.5, freeze_layers=8, patience=1,
        ),
        ClassifierSpec(
            name="tke_bert", kind="tke_bert", stage=2, model_name=bert,
            epochs=4, batch_size=64, lr=1e-5, weight_decay=0.01,
            dropout=0.5, freeze_layers=0, patience=1, use_knowledge=True,
        ),
    ]


def stage_indices(specs: Sequence[ClassifierSpec]) -> Tuple[List[int], List[int]]:
    """Return (stage-1 indices, stage-2 indices) for the environment."""
    stage1 = [i for i, spec in enumerate(specs) if spec.stage == 1]
    stage2 = [i for i, spec in enumerate(specs) if spec.stage == 2]
    if not stage1 or not stage2:
        raise ValueError("Both stages need at least one classifier")
    return stage1, stage2


def validate_classifier_models(specs: Sequence[ClassifierSpec]) -> None:
    """Fail before training starts, listing every model that is missing.

    Keeps the pipeline from training the first classifiers and only then
    discovering that a later checkpoint is unavailable.
    """
    problems = []
    for spec in specs:
        try:
            resolve_model_name(spec.model_name)
        except FileNotFoundError as exc:
            problems.append(f"[{spec.name}] {exc}")
    if problems:
        raise FileNotFoundError(
            "Missing pretrained models:\n\n" + "\n\n".join(problems))


@dataclass
class PPOConfig:
    """Values follow Table 5 of the PPO-CIS paper where applicable."""
    clip: float = 0.2              # epsilon in the clipped surrogate objective
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    aux_coef: float = 0.10         # weight of the auxiliary cross-entropy head
    explore_decide: float = 0.25   # probability of forcing an early decision in rollouts
    max_grad_norm: float = 0.5
    batch_size: int = 256          # samples per rollout batch
    epochs: int = 10               # full passes over ppo_train
    update_epochs: int = 4         # PPO re-use epochs per rollout batch
    hidden_dim: int = 64
    seed: int = 42
    max_steps: int = 8             # 2 stage-1 calls + escalate + 3 stage-2 calls + decision


@dataclass
class RewardConfig:
    """Reward terms from Eq. (1)-(6) of the paper.

    ``r`` and ``iota``: classification reward, ``u``/``phi``/``theta``: time
    reward Eq. (2), ``throughput_scale``: weight of the Eq. (6) batch bonus.
    """
    correct_reward: float = 10.0      # r
    penalty_factor: float = 2.0       # iota
    time_budget: float = 1.0          # u, seconds (paper Table 5)
    time_phi: float = 1.0             # phi
    time_theta: float = 10.0          # theta_1
    throughput_scale: float = 1.0     # Eq.(6) bonus weight
    eps: float = 1e-6


@dataclass
class StaticConfig:
    confidence_threshold: float = 0.70   # xi for the fixed-threshold cascade baseline
