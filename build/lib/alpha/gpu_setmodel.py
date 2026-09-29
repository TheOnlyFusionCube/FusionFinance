"""Permutation-equivariant cross-sectional verifier models without ticker identity."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SetModelConfig:
    n_features: int
    horizons: int = 4
    d_model: int = 64
    dropout: float = 0.1
    max_names: int = 534
    attention_heads: int = 4
    strict_gpu: bool = False

    def __post_init__(self) -> None:
        if self.n_features <= 0 or self.horizons <= 0 or self.d_model <= 0:
            raise ValueError("set model dimensions must be positive")
        if self.max_names <= 0:
            raise ValueError("max_names must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _validated_mask(values, mask):
    import torch

    if values.ndim != 3 or mask.ndim != 2 or values.shape[:2] != mask.shape:
        raise ValueError("values must be (batch,names,...) with mask (batch,names)")
    boolean = mask.to(device=values.device, dtype=torch.bool)
    if not bool(boolean.any(dim=1).all()):
        raise ValueError("every cross section must contain at least one valid name")
    return boolean


def masked_center(values, mask):
    """Center each horizon across valid names and zero padded rows."""
    boolean = _validated_mask(values, mask)
    weights = boolean.unsqueeze(-1).to(dtype=values.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values.float() * weights.float()).sum(dim=1, keepdim=True) / count.float()
    return ((values.float() - mean) * weights.float()).to(values.dtype)


def pad_sets(sequences, *, max_names: int = 534):
    """Pad a list of (names,features) tensors and return a boolean validity mask."""
    import torch

    if not sequences:
        raise ValueError("sequences cannot be empty")
    if any(sequence.ndim != 2 for sequence in sequences):
        raise ValueError("each sequence must have shape (names, features)")
    feature_count = sequences[0].shape[1]
    if any(sequence.shape[1] != feature_count for sequence in sequences):
        raise ValueError("all sequences must share feature width")
    if any(len(sequence) > max_names for sequence in sequences):
        raise ValueError(f"cross section exceeds max_names={max_names}")
    output = sequences[0].new_zeros((len(sequences), max_names, feature_count))
    mask = torch.zeros(len(sequences), max_names, dtype=torch.bool, device=output.device)
    for batch, sequence in enumerate(sequences):
        output[batch, :len(sequence)] = sequence
        mask[batch, :len(sequence)] = True
    return output, mask


def _finalize_heads(raw, mask, horizons: int):
    import torch
    import torch.nn.functional as functional

    batch, names, _ = raw.shape
    heads = raw.float().reshape(batch, names, horizons, 3)
    sign_logit, mu, raw_scale = heads.unbind(dim=-1)
    valid = mask.unsqueeze(-1)
    sign_logit = sign_logit.masked_fill(~valid, 0.0)
    mu = masked_center(mu, mask)
    scale = (functional.softplus(raw_scale) + 1e-4).masked_fill(~valid, 0.0)
    probability = torch.sigmoid(sign_logit).masked_fill(~valid, 0.0)
    return {
        "sign_logit": sign_logit,
        "p_adverse": probability,
        "mu": mu,
        "scale": scale,
    }


def _assert_inputs(module, values, mask, cfg: SetModelConfig) -> None:
    import torch

    if values.ndim != 3 or values.shape[-1] != cfg.n_features:
        raise ValueError("features must have shape (batch,names,n_features)")
    if values.shape[1] > cfg.max_names:
        raise ValueError(f"cross section exceeds max_names={cfg.max_names}")
    if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
        raise ValueError("set model features must be finite float32 values")
    _validated_mask(values, mask)
    if cfg.strict_gpu:
        from alpha.gpu_ml import assert_rocm_stage
        assert_rocm_stage(
            "cross-sectional-verifier", values,
            params=list(module.named_parameters()), strict=True,
        )


class DeepSetsVerifier:
    """Contextual Deep Sets: invariant pools feed an equivariant per-name head."""

    def __new__(cls, cfg: SetModelConfig):
        import torch.nn as nn

        class _DeepSets(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = cfg
                d = cfg.d_model
                self.phi = nn.Sequential(
                    nn.Linear(cfg.n_features, d), nn.GELU(), nn.Dropout(cfg.dropout),
                    nn.Linear(d, d), nn.GELU(),
                )
                self.attention = nn.Linear(d, 1)
                self.head = nn.Sequential(
                    nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(cfg.dropout),
                    nn.Linear(2 * d, cfg.horizons * 3),
                )

            def forward(self, values, mask):
                import torch
                _assert_inputs(self, values, mask, self.cfg)
                boolean = mask.to(device=values.device, dtype=torch.bool)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=values.is_cuda
                ):
                    tokens = self.phi(values)
                    valid = boolean.unsqueeze(-1)
                    weights = valid.to(tokens.dtype)
                    count = weights.sum(dim=1).clamp_min(1.0)
                    mean = (tokens * weights).sum(dim=1) / count
                    variance = ((tokens - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
                    std = variance.clamp_min(0.0).sqrt()
                    logits = self.attention(tokens).squeeze(-1).masked_fill(~boolean, -torch.inf)
                    attention = torch.softmax(logits.float(), dim=1).to(tokens.dtype)
                    pooled = (tokens * attention.unsqueeze(-1)).sum(dim=1)
                    context = torch.cat((mean, std, pooled), dim=-1)
                    joined = torch.cat(
                        (tokens, context.unsqueeze(1).expand(-1, tokens.shape[1], -1)), dim=-1
                    )
                    raw = self.head(joined)
                return _finalize_heads(raw, boolean, self.cfg.horizons)

        return _DeepSets()


class _AttentionBlock:
    def __new__(cls, d_model: int, heads: int, dropout: float):
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(d_model)
                self.attention = nn.MultiheadAttention(
                    d_model, heads, dropout=dropout, batch_first=True
                )
                self.norm2 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
                )

            def forward(self, values, mask):
                normalized = self.norm1(values.float()).to(values.dtype)
                attended, _ = self.attention(
                    normalized, normalized, normalized, key_padding_mask=~mask,
                    need_weights=False,
                )
                values = (values + attended).masked_fill(~mask.unsqueeze(-1), 0.0)
                normalized = self.norm2(values.float()).to(values.dtype)
                return (values + self.ffn(normalized)).masked_fill(~mask.unsqueeze(-1), 0.0)

        return _Block()


class SetTransformerVerifier:
    """Two-block pre-norm set transformer with no positional/ticker embeddings."""

    def __new__(cls, cfg: SetModelConfig):
        import torch.nn as nn

        class _SetTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = cfg
                self.project = nn.Linear(cfg.n_features, cfg.d_model)
                self.blocks = nn.ModuleList([
                    _AttentionBlock(cfg.d_model, cfg.attention_heads, cfg.dropout)
                    for _ in range(2)
                ])
                self.norm = nn.LayerNorm(cfg.d_model)
                self.head = nn.Linear(cfg.d_model, cfg.horizons * 3)

            def forward(self, values, mask):
                import torch
                _assert_inputs(self, values, mask, self.cfg)
                boolean = mask.to(device=values.device, dtype=torch.bool)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=values.is_cuda
                ):
                    hidden = self.project(values).masked_fill(~boolean.unsqueeze(-1), 0.0)
                    for block in self.blocks:
                        hidden = block(hidden, boolean)
                    raw = self.head(self.norm(hidden.float()).to(hidden.dtype))
                return _finalize_heads(raw, boolean, self.cfg.horizons)

        return _SetTransformer()
