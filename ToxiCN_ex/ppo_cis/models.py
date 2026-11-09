"""Classifier architectures used by PPO-CIS.

Every classifier shares the same call signature:
    forward(input_ids, attention_mask, toxic_ids=None, stance_ids=None) -> logits (B, 2)

Source mapping:
    tiny_bert       <- TransformerClassifier on BERT-Tiny (stage 1, throughput)
    distil_roberta  <- TransformerClassifier on a distilled Chinese RoBERTa (stage 1)
    bert            <- bert-safe-chat/train.py :: BertForSequenceClassification
    roberta_fusion  <- CHSD/model.py :: RoBertFusion (RoBERTa + BiGRU + TextCNN)
    tke_bert        <- src/BERT.py (knowledge embeddings) + src/Models.py (FFNN head)

The TKE-BERT implementation below reproduces the original architecture with the
modern ``transformers`` BertModel: the toxic/stance knowledge embeddings are added
to the word embeddings *before* the position/token-type/LayerNorm step, which is
mathematically identical to the original ``BertEmbeddings`` code.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification

from .config import resolve_model_name


def _base_encoder(model: nn.Module) -> nn.Module:
    """Return the transformer backbone of an HF model (BertModel / RoBERTaModel)."""
    for attr in ("bert", "roberta", "deberta", "electra", "distilbert"):
        base = getattr(model, attr, None)
        if base is not None:
            return base
    return model


def _freeze_layers(model: nn.Module, freeze_layers: int) -> None:
    """Freeze the first ``freeze_layers`` transformer blocks (0 = train all)."""
    if freeze_layers <= 0:
        return
    base = _base_encoder(model)
    layers = getattr(getattr(base, "encoder", None), "layer", None)
    if layers is None:
        return
    for layer in layers[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False


class TransformerClassifier(nn.Module):
    """HF ``AutoModelForSequenceClassification``, as used by bert-safe-chat/train.py."""

    def __init__(self, model_name: str, num_classes: int = 2, dropout: float = 0.1,
                 freeze_layers: int = 0):
        super().__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes, hidden_dropout_prob=dropout)
        _freeze_layers(self.encoder, freeze_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                toxic_ids: Optional[torch.Tensor] = None,
                stance_ids: Optional[torch.Tensor] = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


class RoBertFusionClassifier(nn.Module):
    """RoBERTa + BiGRU + TextCNN, following CHSD/model.py :: RoBertFusion."""

    def __init__(self, model_name: str, num_classes: int = 2, gru_hidden: int = 64,
                 num_filters: int = 50, kernel_sizes: Sequence[int] = (3, 4, 5),
                 dropout: float = 0.5, freeze_layers: int = 8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, gru_hidden, bidirectional=True, batch_first=True)
        self.textcnn = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, hidden)) for k in kernel_sizes
        ])
        combined_size = gru_hidden * 2 + num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(combined_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        _freeze_layers(self.encoder, freeze_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                toxic_ids: Optional[torch.Tensor] = None,
                stance_ids: Optional[torch.Tensor] = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs[0]
        last_hidden = self.dropout(last_hidden)

        gru_out, _ = self.gru(last_hidden)
        gru_feat = self.dropout(gru_out[:, -1, :])

        x = last_hidden.unsqueeze(1)                    # (B, 1, L, H)
        cnn_feats = []
        for conv in self.textcnn:
            conv_out = torch.relu(conv(x)).squeeze(3)   # (B, filters, L')
            cnn_feats.append(torch.max(conv_out, dim=2).values)
        cnn_feat = self.dropout(torch.cat(cnn_feats, dim=1))

        features = torch.cat((gru_feat, cnn_feat), dim=1)
        hidden = torch.relu(self.fc1(features))
        return self.fc2(self.dropout(hidden))


class TKEBertClassifier(nn.Module):
    """BERT + toxic/stance knowledge embeddings + two-layer FFNN (ToxiCN_ex).

    Reproduces ``src/BERT.py :: BertEmbeddings`` + ``src/Models.py :: TwoLayerFFNNLayer``
    without depending on the old transformers-3.1 modelling file.
    """

    def __init__(self, model_name: str, num_classes: int = 2,
                 num_toxic_types: int = 6, dropout: float = 0.5,
                 freeze_layers: int = 0, fc_hidden_dim: int = 256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        # index 0 = no knowledge, 1..5 = the five ToxiCN lexicon categories
        self.toxic_embeddings = nn.Embedding(num_toxic_types, hidden)
        self.stance_embeddings = nn.Embedding(2, hidden)
        nn.init.normal_(self.toxic_embeddings.weight, std=0.02)
        nn.init.normal_(self.stance_embeddings.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.embedding_projection = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, fc_hidden_dim),
            nn.Tanh(),
            nn.Linear(fc_hidden_dim, num_classes),
        )
        _freeze_layers(self.bert, freeze_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                toxic_ids: Optional[torch.Tensor] = None,
                stance_ids: Optional[torch.Tensor] = None):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # word embedding + knowledge embedding (same as src/BERT.py Eq. embeddings)
        embeds = self.bert.embeddings.word_embeddings(input_ids)
        if toxic_ids is not None:
            embeds = embeds + self.toxic_embeddings(toxic_ids)
        if stance_ids is not None:
            embeds = embeds + self.stance_embeddings(stance_ids)

        # position + token-type + LayerNorm (+ dropout)
        embeds = self.bert.embeddings(inputs_embeds=embeds, token_type_ids=token_type_ids)
        # Build the additive attention mask directly: transformers 5.x removed
        # ModuleUtilsMixin.get_extended_attention_mask().
        extended_mask = (1.0 - attention_mask[:, None, None, :].to(embeds.dtype)) * \
            torch.finfo(embeds.dtype).min
        sequence_output = self.bert.encoder(embeds, attention_mask=extended_mask)[0]
        pooled = self.bert.pooler(sequence_output)

        hidden = torch.tanh(self.embedding_projection(self.dropout(pooled)))
        return self.classifier(self.dropout(hidden))


def build_classifier(spec, vocab_size: int = 0, pad_id: int = 0) -> nn.Module:
    """Instantiate a classifier from its ``ClassifierSpec``.

    ``spec.model_name`` is passed through ``resolve_model_name`` so that both
    ``MODEL_ROOT`` paths and old checkpoints storing plain HuggingFace ids load
    from the local model folders.
    """
    model_name = resolve_model_name(spec.model_name)
    if spec.kind == "bert":
        return TransformerClassifier(model_name, dropout=spec.dropout,
                                     freeze_layers=spec.freeze_layers)
    if spec.kind == "roberta_fusion":
        return RoBertFusionClassifier(model_name, dropout=spec.dropout,
                                      freeze_layers=spec.freeze_layers)
    if spec.kind == "tke_bert":
        return TKEBertClassifier(model_name, dropout=spec.dropout,
                                 freeze_layers=spec.freeze_layers)
    raise ValueError(f"Unknown classifier kind: {spec.kind}")
