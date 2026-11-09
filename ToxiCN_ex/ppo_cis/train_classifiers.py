"""Train the five PPO-CIS classifiers on ToxiCN (two throughput + three accuracy).

Stage 1 (high throughput): ``tiny_bert`` and ``distil_roberta``.
Stage 2 (high accuracy):   ``bert``, ``roberta_fusion`` and ``tke_bert``.

All five classifiers use the standard cross-entropy loss ``nn.CrossEntropyLoss``
over two logits (the binary cross-entropy used by the original ToxiCN_ex
``BCEWithLogitsLoss`` on a 2-dim target is the same objective in one-hot form).

Usage (from the ToxiCN_ex directory):
    python -m ppo_cis.train_classifiers
    python -m ppo_cis.train_classifiers --only tiny_bert distil_roberta --device cuda
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import CLASSIFIER_DIR, DataConfig, default_classifier_specs
from .data import (
    EncodedDataset, build_splits, encode_split, get_tokenizer, print_split_stats,
)
from .models import build_classifier
from .utils import (
    amp_context, compute_metrics, forward_classifier, grad_scaler, get_device,
    load_json, measure_latency, save_json,
)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    losses, preds, labels = [], [], []
    for batch in loader:
        target = batch["labels"].to(device)
        with amp_context(device):
            logits = forward_classifier(model, batch, device)
            loss = criterion(logits, target)
        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        labels.extend(target.cpu().numpy().tolist())
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


def train_one(spec, splits, tokenizer, device, data_cfg: DataConfig,
              force: bool = False) -> Dict[str, float]:
    ckpt_path = CLASSIFIER_DIR / f"{spec.name}.pt"
    if ckpt_path.exists() and not force:
        print(f"[skip] {spec.name}: {ckpt_path} already exists (use --force to retrain)")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("test_metrics", {})

    max_len = data_cfg.max_len
    knowledge = spec.use_knowledge
    train_t = encode_split(splits["clf_train"], tokenizer, "clf_train", max_len,
                           knowledge=knowledge)
    dev_t = encode_split(splits["clf_dev"], tokenizer, "clf_dev", max_len,
                         knowledge=knowledge)
    test_t = encode_split(splits["test"], tokenizer, "test", max_len,
                          knowledge=knowledge)

    train_loader = DataLoader(EncodedDataset(train_t), batch_size=spec.batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    dev_loader = DataLoader(EncodedDataset(dev_t), batch_size=spec.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(EncodedDataset(test_t), batch_size=spec.batch_size,
                             shuffle=False, num_workers=0)

    model = build_classifier(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr,
                                  weight_decay=spec.weight_decay)
    total_steps = max(len(train_loader) * spec.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = grad_scaler(device)

    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, -1, 0
    for epoch in range(spec.epochs):
        model.train()
        running_loss = 0.0
        seen_batches = 0
        pbar = tqdm(train_loader, desc=f"{spec.name} epoch {epoch + 1}/{spec.epochs}",
                    leave=False)
        for batch in pbar:
            target = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = forward_classifier(model, batch, device)
                loss = criterion(logits, target)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            seen_batches += 1
            pbar.set_postfix(loss=f"{running_loss / seen_batches:.4f}")

        dev_metrics = evaluate(model, dev_loader, device, criterion)
        print(f"[{spec.name}] epoch {epoch + 1}: "
              f"dev_acc={dev_metrics['accuracy']:.4f} dev_f1={dev_metrics['f1']:.4f} "
              f"dev_loss={dev_metrics['loss']:.4f}")

        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > spec.patience:
                print(f"[{spec.name}] early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, criterion)
    latency = measure_latency(model, test_loader, device)
    test_metrics.update(latency)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["n_train"] = len(train_t["labels"])
    test_metrics["n_test"] = len(test_t["labels"])
    test_metrics["model_name"] = spec.model_name

    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "spec": asdict(spec),
        "state_dict": model.state_dict(),
        "tokenizer_name": tokenizer.name_or_path,
        "max_len": max_len,
        "use_knowledge": knowledge,
        "test_metrics": test_metrics,
    }, ckpt_path)

    metrics_file = CLASSIFIER_DIR / "metrics.json"
    all_metrics = load_json(metrics_file) if metrics_file.exists() else {}
    all_metrics[spec.name] = test_metrics
    all_metrics["_meta"] = {
        "loss": "CrossEntropyLoss",
        "layout": "two-stage: tiny_bert + distil_roberta -> bert + roberta_fusion + tke_bert",
        "max_len": max_len,
    }
    save_json(all_metrics, metrics_file)

    print(f"[{spec.name}] TEST acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
          f"latency={test_metrics['latency_ms']:.2f}ms "
          f"throughput={test_metrics['throughput']:.1f}/s -> {ckpt_path}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the five PPO-CIS classifiers")
    parser.add_argument("--bert_model", default=None, help="BERT model name or local path")
    parser.add_argument("--roberta_model", default=None, help="RoBERTa model name or local path")
    parser.add_argument("--tiny_bert_model", default=None,
                        help="stage-1 BERT-Tiny model name or local path")
    parser.add_argument("--distil_roberta_model", default=None,
                        help="stage-1 distilled RoBERTa model name or local path")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--max_len", default=96, type=int)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    data_cfg = DataConfig(max_len=args.max_len)
    splits = build_splits(data_cfg)
    print_split_stats(splits)

    specs = default_classifier_specs(args.bert_model, args.roberta_model,
                                     args.tiny_bert_model, args.distil_roberta_model)
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]

    tokenizers = {}
    for spec in specs:
        tokenizer_name = spec.model_name
        if tokenizer_name not in tokenizers:
            tokenizers[tokenizer_name] = get_tokenizer(tokenizer_name)
        train_one(spec, splits, tokenizers[tokenizer_name], device, data_cfg, args.force)


if __name__ == "__main__":
    main()
