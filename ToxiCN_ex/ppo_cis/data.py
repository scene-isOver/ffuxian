"""ToxiCN data loading, cleaning, splitting, knowledge ids and tokenisation cache.

Dataset (already inside the repository):
    ToxiCN_ex/ToxiCN/data/train.json  (32,157 binary-labelled comments)
    ToxiCN_ex/ToxiCN/data/test.json   (5,323 binary-labelled comments)

Cleaning:
    * drop empty / too short comments
    * drop exact duplicates inside each split
    * drop test comments that also occur in train (avoids leakage)
    * keep only the binary ``toxic`` label, which is what PPO-CIS predicts

Knowledge ids for TKE-BERT:
    * ``toxic_ids``  : token positions covered by the five ToxiCN lexicons
                       (general / LGBT / region / sexism / racism), encoded 1..5
    * ``stance_ids`` : token positions of an attack stance.  The repository does
                       not ship ``attack_stance.json``, so this is all zeros
                       (neutral stance) unless the file is added later.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import CACHE_DIR, LEXICON_DIR, DataConfig, resolve_model_name

# ===================== 词典与清洗 =====================
# 五份词典文件对应“通用 / LGBT / 地域 / 性别歧视 / 种族歧视”五个毒性子类，
# 文件里存的是 token id 序列而不是字符串，所以匹配直接对 input_ids 做。
LEXICON_FILES = ("general.json", "LGBT.json", "region.json", "sexism.json", "racism.json")


def _read_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 单条记录清洗：去空白 / 去太短 / 去完全重复 / 丢非法标签，只保留 0-1 二分类。
def _clean(records: List[dict], min_chars: int) -> List[dict]:
    cleaned: List[dict] = []
    seen = set()
    for row in records:
        text = str(row.get("content", "")).strip()
        if len(text) < min_chars or text in seen:
            continue
        label = row.get("toxic", None)
        if label is None:
            continue
        try:
            label = int(label)
        except (TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue
        seen.add(text)
        cleaned.append({"text": text, "label": label})
    return cleaned


# ===================== 数据切分 =====================
# 先把 test 里与 train 重复的文本剔除（防泄漏），再从 train 按分层抽样切出
# clf_dev（验证集，用于早停）和 ppo_train（策略训练集），剩下的作为 clf_train。
def build_splits(cfg: DataConfig) -> Dict[str, list]:
    """Return clf_train / clf_dev / ppo_train / test (already cleaned)."""
    from sklearn.model_selection import train_test_split

    train_rows = _clean(_read_json(cfg.train_json), cfg.min_chars)
    test_rows = _clean(_read_json(cfg.test_json), cfg.min_chars)

    train_texts = {row["text"] for row in train_rows}
    test_rows = [row for row in test_rows if row["text"] not in train_texts]

    labels = [row["label"] for row in train_rows]
    temp_size = cfg.val_ratio + cfg.ppo_ratio
    train_part, temp_part = train_test_split(
        train_rows, test_size=temp_size, stratify=labels, random_state=cfg.seed
    )
    temp_labels = [row["label"] for row in temp_part]
    dev_part, ppo_part = train_test_split(
        temp_part, test_size=0.5, stratify=temp_labels, random_state=cfg.seed
    )

    def label_stats(rows):
        pos = sum(row["label"] for row in rows)
        return {"n": len(rows), "toxic": pos, "nontoxic": len(rows) - pos}

    return {
        "clf_train": train_part,
        "clf_dev": dev_part,
        "ppo_train": ppo_part,
        "test": test_rows,
        "stats": {
            "clf_train": label_stats(train_part),
            "clf_dev": label_stats(dev_part),
            "ppo_train": label_stats(ppo_part),
            "test": label_stats(test_rows),
        },
    }


# 打印四个划分的样本量与有毒/无毒分布，确认切分没有失衡。
def print_split_stats(splits: Dict[str, list]) -> None:
    print("Dataset split (cleaned):")
    for name in ("clf_train", "clf_dev", "ppo_train", "test"):
        s = splits["stats"][name]
        print("  {:<10s} n={:<6d} toxic={:<6d} nontoxic={:<6d}".format(
            name, s["n"], s["toxic"], s["nontoxic"]))


# ------------------------------------------------------------ knowledge ids ----

# ===================== TKE-BERT 的知识 id =====================
# ToxicLexicon 把五份词典加载成“首 token -> 候选词”的索引，
# 再对一句话扫描出每个 token 命中哪个毒性子类（0 表示未命中）。
class ToxicLexicon:
    """Fast matcher from BERT token ids to the five ToxiCN lexicon categories.

    The lexicon JSON files store *token ids* (not characters), e.g.
    ``{"傻逼": [1004, 6873]}``, so matching is done directly on ``input_ids``.
    """

    def __init__(self, lexicon_dir: Path = LEXICON_DIR):
        self.by_first: Dict[int, List[tuple]] = {}
        for level, filename in enumerate(LEXICON_FILES, start=1):
            path = Path(lexicon_dir) / filename
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for token_ids in data.values():
                ids = tuple(int(t) for t in token_ids)
                if not ids:
                    continue
                self.by_first.setdefault(ids[0], []).append((ids, level))

    def ids_for(self, input_ids: List[int]) -> List[int]:
        n = len(input_ids)
        out = [0] * n
        for start, token in enumerate(input_ids):
            if token == 0:                      # padding
                break
            for word, level in self.by_first.get(token, ()):
                end = start + len(word)
                if end <= n and tuple(input_ids[start:end]) == word:
                    for pos in range(start, end):
                        out[pos] = level
        return out


def build_toxic_ids(input_ids: List[int], lexicon: ToxicLexicon) -> List[int]:
    return lexicon.ids_for(input_ids)


def build_stance_ids(input_ids: List[int]) -> List[int]:
    """Stance ids are all 0 (neutral): attack_stance.json is not in the repo."""
    return [0] * len(input_ids)


# --------------------------------------------------------------- tokenising ---

# ===================== 分词与缓存 =====================
def get_tokenizer(model_name: str):
    # ``use_fast=True`` is much faster for the Chinese BERT/RoBERTa tokenizers.
    # ``resolve_model_name`` maps HF ids / checkpoint names onto MODEL_ROOT and
    # raises when the directory is missing (no network fallback).
    return AutoTokenizer.from_pretrained(
        resolve_model_name(model_name), use_fast=True, local_files_only=True)


# 分词结果缓存键：tokenizer 名 + max_len + 是否要知识 id，三者相同才复用。
def _cache_path(split_name: str, tokenizer_name: str, max_len: int,
                knowledge: bool) -> Path:
    key = hashlib.md5(
        f"{tokenizer_name}|{max_len}|{int(knowledge)}".encode("utf-8")
    ).hexdigest()[:10]
    return CACHE_DIR / f"{split_name}_{key}.pt"


# 把一个 split 整体分词成张量并缓存到 artifacts/cache；
# knowledge=True 时额外生成 toxic_ids / stance_ids 两路知识 id。
def encode_split(records: List[dict], tokenizer, split_name: str,
                 max_len: int, use_cache: bool = True,
                 knowledge: bool = False) -> Dict[str, torch.Tensor]:
    """Tokenise a split once and cache the tensors on disk.

    ``knowledge=True`` additionally produces ``toxic_ids`` / ``stance_ids`` for
    TKE-BERT.
    """
    tokenizer_name = getattr(tokenizer, "name_or_path", "unknown")
    path = _cache_path(split_name, tokenizer_name, max_len, knowledge)
    if use_cache and path.exists():
        return torch.load(path)

    encoded = tokenizer(
        [row["text"] for row in records],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    tensors = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor([row["label"] for row in records], dtype=torch.long),
        "lengths": encoded["attention_mask"].sum(dim=1),
    }
    if knowledge:
        lexicon = ToxicLexicon()
        toxic_ids, stance_ids = [], []
        for ids in encoded["input_ids"].tolist():
            toxic_ids.append(build_toxic_ids(ids, lexicon))
            stance_ids.append(build_stance_ids(ids))
        tensors["toxic_ids"] = torch.tensor(toxic_ids, dtype=torch.long)
        tensors["stance_ids"] = torch.tensor(stance_ids, dtype=torch.long)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, path)
    return tensors


# 训练用的 Dataset 包装：按索引返回 input_ids / attention_mask / labels，
# 有知识 id 时一并带上（DataLoader 负责组 batch）。
class EncodedDataset(Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors["labels"].size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.tensors["input_ids"][idx],
            "attention_mask": self.tensors["attention_mask"][idx],
            "labels": self.tensors["labels"][idx],
        }
        if "toxic_ids" in self.tensors:
            item["toxic_ids"] = self.tensors["toxic_ids"][idx]
        if "stance_ids" in self.tensors:
            item["stance_ids"] = self.tensors["stance_ids"][idx]
        return item
