"""Vectorized uncertainty-aware verifier policy that stays on the torch device."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


REJECT = -1
ABSTAIN = 0
APPROVE = 1
_RESERVED_CONTEXT_KEYS = frozenset({
    "schema_version", "p_adverse", "epistemic_mi", "novelty",
})


@dataclass(frozen=True)
class TensorPolicyThresholds:
    approve_max_adverse: float = 0.35
    reject_min_adverse: float = 0.65
    max_epistemic_mi: float = 0.10
    max_novelty: float = 0.50

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not 0.0 <= float(value) <= 1.0 for value in values.values()):
            raise ValueError("tensor policy thresholds must be in [0, 1]")
        if self.approve_max_adverse >= self.reject_min_adverse:
            raise ValueError("approve threshold must be below reject threshold")


@dataclass(frozen=True)
class TensorPolicyResult:
    decisions: Any
    decision_hash: str


def _validate_inputs(p_adverse: Any, epistemic_mi: Any, novelty: Any) -> tuple[Any, Any, Any]:
    import torch

    if not all(torch.is_tensor(value) for value in (p_adverse, epistemic_mi, novelty)):
        raise TypeError("tensor policy inputs must be torch tensors")
    try:
        adverse, epistemic, novel = torch.broadcast_tensors(
            p_adverse, epistemic_mi, novelty
        )
    except RuntimeError as exc:
        raise ValueError("tensor policy inputs are not broadcast-compatible") from exc
    if not all(value.device == adverse.device for value in (epistemic, novel)):
        raise ValueError("tensor policy inputs must share one device")
    for name, value in (("p_adverse", adverse), ("epistemic_mi", epistemic),
                        ("novelty", novel)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        if not bool(((value >= 0.0) & (value <= 1.0)).all()):
            raise ValueError(f"{name} must be in [0, 1]")
    return adverse.float(), epistemic.float(), novel.float()


def hash_decisions(decisions: Any, thresholds: TensorPolicyThresholds,
                   context: Mapping[str, Any] | None = None) -> str:
    import torch

    if not torch.is_tensor(decisions) or decisions.dtype != torch.int8:
        raise TypeError("decisions must be an int8 torch tensor")
    payload = {
        "decisions": decisions.detach().cpu().contiguous().tolist(),
        "shape": list(decisions.shape),
        "thresholds": asdict(thresholds),
        "context": dict(context or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_policy(p_adverse: Any, epistemic_mi: Any, novelty: Any, *,
                  thresholds: TensorPolicyThresholds = TensorPolicyThresholds(),
                  context: Mapping[str, Any] | None = None) -> TensorPolicyResult:
    """Return int8 decisions on the input device, then hash the produced tensor."""
    import torch

    supplied_context = dict(context or {})
    reserved = _RESERVED_CONTEXT_KEYS.intersection(supplied_context)
    if reserved:
        raise ValueError(f"policy context uses reserved key(s): {sorted(reserved)}")
    adverse, epistemic, novel = _validate_inputs(p_adverse, epistemic_mi, novelty)
    uncertain = ((epistemic > thresholds.max_epistemic_mi) |
                 (novel > thresholds.max_novelty))
    decisions = torch.full(adverse.shape, ABSTAIN, dtype=torch.int8, device=adverse.device)
    decisions = torch.where(
        (~uncertain) & (adverse <= thresholds.approve_max_adverse),
        torch.tensor(APPROVE, dtype=torch.int8, device=adverse.device), decisions,
    )
    decisions = torch.where(
        (~uncertain) & (adverse >= thresholds.reject_min_adverse),
        torch.tensor(REJECT, dtype=torch.int8, device=adverse.device), decisions,
    )
    hash_context = {
        "schema_version": 1,
        "p_adverse": adverse.detach().cpu().tolist(),
        "epistemic_mi": epistemic.detach().cpu().tolist(),
        "novelty": novel.detach().cpu().tolist(),
        **supplied_context,
    }
    return TensorPolicyResult(
        decisions, hash_decisions(decisions, thresholds, hash_context)
    )
