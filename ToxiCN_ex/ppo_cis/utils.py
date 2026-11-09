"""Small shared helpers (metrics, JSON, AMP, latency)."""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(preds, labels) -> Dict[str, float]:
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "fnr": float(((preds == 0) & (labels == 1)).sum() / max((labels == 1).sum(), 1)),
        "fpr": float(((preds == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)),
    }


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def forward_classifier(model, batch, device) -> torch.Tensor:
    """Run a classifier on one batch, forwarding knowledge ids when present."""
    kwargs = {}
    if "toxic_ids" in batch:
        kwargs["toxic_ids"] = batch["toxic_ids"].to(device)
    if "stance_ids" in batch:
        kwargs["stance_ids"] = batch["stance_ids"].to(device)
    return model(batch["input_ids"].to(device),
                 batch["attention_mask"].to(device), **kwargs)


def get_device(name: Optional[str] = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def grad_scaler(device: torch.device):
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):     # torch < 2.3
        return torch.cuda.amp.GradScaler()


def measure_latency(model, loader, device: torch.device,
                    max_batches: int = 20, warmup_batches: int = 3) -> Dict[str, float]:
    """Measure single-sample latency and throughput with the real batch size."""
    model.eval()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            with amp_context(device):
                forward_classifier(model, batch, device)
            if i + 1 >= warmup_batches:
                break
        sync()
        start = time.perf_counter()
        n_samples = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            with amp_context(device):
                forward_classifier(model, batch, device)
            n_samples += batch["input_ids"].size(0)
        sync()
        elapsed = time.perf_counter() - start

    latency = elapsed / max(n_samples, 1)
    return {
        "latency_sec": float(latency),
        "latency_ms": float(latency * 1000),
        "throughput": float(1.0 / max(latency, 1e-12)),
        "measured_samples": int(n_samples),
    }
