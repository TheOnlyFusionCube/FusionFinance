"""GPU-native ML core for the fusion stack under ROCm.

Device-agnostic torch: the math is validated on CPU, while strict production runs
pin ``cuda:0`` and fail closed unless they are executing through a ROCm/HIP build.

Design reviewed with an external ML-systems pass (GPT-5.6 "Sol"); the corrections
below are why the semantics differ from a naive ensemble:

1. ``DeepEnsemble`` — M MLPs held as batched 3-D weight tensors, one ``baddbmm``
   chain over the member axis. Each member emits THREE heads per row:
     * ``sign_logit`` -> P(residual < 0), the verifier's adverse probability
       (a DIRECT classifier, not a Normal-CDF of the ensemble spread — the spread
       is EPISTEMIC disagreement, not conditional return vol, so using it as
       P(adverse) is mis-specified);
     * ``mu``         -> conditional location of the residual return;
     * ``scale``      -> conditional ALEATORIC scale (softplus).
   Diversity comes from independent init + per-member DATE-BLOCK weights (stock-date
   rows are not independent, so we resample whole trading days, not rows). predict()
   returns calibrated P(adverse) plus an epistemic/aleatoric variance decomposition
   and the epistemic mutual information — the signals the meta-policy actually needs.

2. ``gpu_bootstrap`` — circular block-bootstrap of a return stream, chunked, with a
   Monte-Carlo standard error. Highest scientific value per VRAM byte.

3. ``fit_whitener`` / ``TorchOOD`` — Mahalanobis novelty via shrinkage + eigh (NO
   explicit inverse; RDNA3-safe), fit on train only.

4. ``assert_rocm_stage`` / ``device_report`` / ``MemoryProbe`` — fail-closed proof
   that each stage ran on the AMD GPU, plus the peak-VRAM receipts.
"""
from __future__ import annotations

import math
import resource
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# device + memory receipts / fail-closed GPU assertion
# --------------------------------------------------------------------------- #
def device_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return {"backend": "none", "error": repr(exc)}
    cuda = torch.cuda.is_available()
    hip = getattr(getattr(torch, "version", None), "hip", None)
    props = torch.cuda.get_device_properties(0) if cuda else None
    return {
        "backend": "torch-cuda" if cuda else "torch-cpu",
        "torch": str(torch.__version__),
        "rocm_hip": str(hip) if hip else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "device_count": torch.cuda.device_count() if cuda else 0,
        "vram_total_bytes": int(props.total_memory) if props else 0,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def assert_rocm_stage(stage: str, *tensors, expect_device_substr: str | None = None,
                      params=None, strict: bool = True) -> None:
    """Fail closed: refuse to run a stage on anything but the AMD ROCm GPU.
    A negative CI test that hides the GPU must make this raise at startup rather
    than let the stage fall through to CPU."""
    import torch
    if not strict:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"{stage}: no CUDA/HIP device (GPU unavailable)")
    if getattr(getattr(torch, "version", None), "hip", None) is None:
        raise RuntimeError(f"{stage}: torch is not a ROCm/HIP build")
    name = torch.cuda.get_device_name(0)
    if expect_device_substr and expect_device_substr not in name:
        raise RuntimeError(f"{stage}: unexpected device {name!r} (want {expect_device_substr})")
    for i, t in enumerate(tensors):
        if torch.is_tensor(t):
            if not t.is_cuda:
                raise RuntimeError(f"{stage}: CPU tensor arg {i}")
            if (t.device.index if t.device.index is not None else torch.cuda.current_device()) != 0:
                raise RuntimeError(f"{stage}: tensor arg {i} is not on verified cuda:0")
    for nm, p in (params or []):
        if not p.is_cuda:
            raise RuntimeError(f"{stage}: CPU parameter {nm}")
        if (p.device.index if p.device.index is not None else torch.cuda.current_device()) != 0:
            raise RuntimeError(f"{stage}: parameter {nm} is not on verified cuda:0")


class MemoryProbe:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        try:
            import torch
            self._torch = torch
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            self._torch = None

    def snapshot(self) -> dict[str, float]:
        peak_a = peak_r = 0
        t = self._torch
        if t is not None and t.cuda.is_available() and self.device.startswith("cuda"):
            t.cuda.synchronize()
            peak_a = int(t.cuda.max_memory_allocated())
            peak_r = int(t.cuda.max_memory_reserved())
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_b = rss * 1024 if sys.platform != "darwin" else rss
        return {"peak_vram_alloc_gb": round(peak_a / 2**30, 3),
                "peak_vram_reserved_gb": round(peak_r / 2**30, 3),
                "peak_host_rss_gb": round(rss_b / 2**30, 3)}


def _pick_device(device: str) -> str:
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    return device


# --------------------------------------------------------------------------- #
# probabilistic deep ensemble (sign + location + aleatoric-scale heads)
# --------------------------------------------------------------------------- #
@dataclass
class EnsembleConfig:
    members: int = 32                 # ablate M in {4,8,16,32,64,96}; keep smallest that ties
    hidden: tuple[int, ...] = (256, 256, 128)
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.10
    lambda_return: float = 0.25       # weight on the Gaussian-NLL head vs the sign BCE
    block_len: int = 20               # date-block bootstrap expected length (trading days)
    seed0: int = 7
    device: str = "cuda:0"
    strict: bool = False              # True => never fall back to CPU (production/host)

    def __post_init__(self) -> None:
        if self.members <= 0 or self.epochs <= 0 or self.block_len <= 0:
            raise ValueError("members, epochs, and block_len must be positive")
        if not self.hidden or any(width <= 0 for width in self.hidden):
            raise ValueError("hidden widths must be non-empty and positive")
        if self.lr <= 0.0 or self.weight_decay < 0.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid optimizer/dropout configuration")
        if self.lambda_return < 0.0:
            raise ValueError("lambda_return cannot be negative")

    def resolved_device(self) -> str:
        dev = _pick_device(self.device)
        if self.strict and not dev.startswith("cuda"):
            raise RuntimeError("EnsembleConfig.strict: ROCm cuda:0 required")
        return dev


class DeepEnsemble:
    _HEADS = 3  # sign_logit, mu, raw_scale

    def __init__(self, cfg: EnsembleConfig):
        self.cfg = cfg
        self.device = cfg.resolved_device()
        self._W: list = []
        self._b: list = []
        self.n_in: int | None = None
        self.y_center = 0.0
        self.y_scale = 1.0
        self.trained = False

    def _init_params(self, n_in: int):
        import torch
        g = torch.Generator(device="cpu").manual_seed(self.cfg.seed0)
        dims = [n_in, *self.cfg.hidden, self._HEADS]
        M = self.cfg.members
        self._W, self._b = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            w = (torch.randn(M, a, b, generator=g) * math.sqrt(2.0 / a)).to(self.device).requires_grad_(True)
            bias = torch.zeros(M, 1, b, device=self.device, requires_grad=True)
            self._W.append(w)
            self._b.append(bias)
        self.n_in = n_in

    def _forward(self, Xb, *, train: bool):
        import torch
        import torch.nn.functional as F
        H = Xb
        last = len(self._W) - 1
        for i, (W, b) in enumerate(zip(self._W, self._b)):
            H = torch.baddbmm(b, H, W)          # (M,N,out)
            if i != last:
                H = F.gelu(H)
                if train and self.cfg.dropout > 0:
                    H = F.dropout(H, p=self.cfg.dropout, training=True)
        return H  # (M,N,3)

    def _block_counts(self, n_dates: int) -> "Any":
        """Member-specific circular date-block bootstrap weights: (M, D)."""
        import torch
        M, L = self.cfg.members, max(1, self.cfg.block_len)
        n_blocks = math.ceil(n_dates / L)
        g = torch.Generator(device="cpu").manual_seed(self.cfg.seed0 + 101)
        starts = torch.randint(0, n_dates, (M, n_blocks), generator=g)
        offs = torch.arange(L)
        idx = (starts.unsqueeze(-1) + offs).reshape(M, -1)[:, :n_dates] % n_dates  # (M, D)
        counts = torch.zeros(M, n_dates)
        counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=counts.dtype))
        return counts.to(self.device)          # weight per (member, date)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> dict[str, Any]:
        """X:(N,in) y:(N,) residual returns; groups:(N,) date id per row (for date-block
        bagging). If groups is None, each row is its own date (row bootstrap fallback)."""
        import torch
        import torch.nn.functional as F
        cfg = self.cfg
        X = np.ascontiguousarray(X, np.float32)
        y = np.ascontiguousarray(y, np.float32).reshape(-1)
        if X.ndim != 2 or not len(X) or len(X) != len(y):
            raise ValueError("X and y must be non-empty aligned rank-2/rank-1 arrays")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("ensemble training arrays must be finite")
        N, n_in = X.shape
        if groups is None:
            groups = np.arange(N, dtype=np.int64)
        group_values = np.asarray(groups).reshape(-1)
        if len(group_values) != N:
            raise ValueError("groups must align with training rows")
        gcodes, ginv = np.unique(group_values, return_inverse=True)
        D = len(gcodes)
        self._init_params(n_in)

        # unit-scale the target so low-precision NLL is well conditioned
        self.y_center = float(np.median(y))
        self.y_scale = float(np.std(y) + 1e-8)
        ys = (y - self.y_center) / self.y_scale

        Xt = torch.from_numpy(X).to(self.device)
        yt = torch.from_numpy(ys).to(self.device)
        y_neg = (torch.from_numpy(y).to(self.device) < 0).float()          # adverse label
        row_date = torch.from_numpy(ginv.astype(np.int64)).to(self.device)  # (N,)
        counts = self._block_counts(D)                                      # (M,D)
        w = counts[:, row_date]                                             # (M,N) member date weights
        w = w / w.mean().clamp_min(1e-6)

        Xb = Xt.unsqueeze(0).expand(cfg.members, -1, -1)                    # (M,N,in)
        params = [*self._W, *self._b]
        assert_rocm_stage(
            "deep-ensemble-fit", Xt, yt, params=[(str(index), value) for index, value in enumerate(params)],
            strict=cfg.strict,
        )
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        t0 = time.time()
        last = float("nan")
        for _ in range(cfg.epochs):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=self.device.startswith("cuda"),
            ):
                out = self._forward(Xb, train=True)                        # (M,N,3)
            out = out.float()
            sign_logit, mu, raw_scale = out.unbind(-1)
            scale = F.softplus(raw_scale) + 1e-4
            bce = F.binary_cross_entropy_with_logits(
                sign_logit, y_neg.unsqueeze(0).expand_as(sign_logit), reduction="none")
            nll = torch.log(scale) + 0.5 * ((yt.unsqueeze(0) - mu) / scale) ** 2
            loss_elem = bce + cfg.lambda_return * nll                      # (M,N)
            loss = (loss_elem * w).sum() / w.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            last = float(loss.detach())
        self.trained = True
        return {"final_loss": round(last, 6), "seconds": round(time.time() - t0, 3),
                "members": cfg.members, "rows": N, "dates": D,
                "params_per_member": int(sum(W.shape[1] * W.shape[2] + b.shape[2]
                                             for W, b in zip(self._W, self._b)))}

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        import torch
        import torch.nn.functional as F
        X = np.ascontiguousarray(X, np.float32)
        if not self.trained or X.ndim != 2 or X.shape[1] != self.n_in or not np.isfinite(X).all():
            raise ValueError("prediction requires a fitted model and finite aligned features")
        Xt = torch.from_numpy(X).to(self.device)
        assert_rocm_stage(
            "deep-ensemble-predict", Xt,
            params=[(str(index), value) for index, value in enumerate([*self._W, *self._b])],
            strict=self.cfg.strict,
        )
        Xb = Xt.unsqueeze(0).expand(self.cfg.members, -1, -1)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=self.device.startswith("cuda"),
            ):
                out = self._forward(Xb, train=False)
            out = out.float()
            sign_logit, mu, raw_scale = out.unbind(-1)
            scale = F.softplus(raw_scale) + 1e-4
            p_member = torch.sigmoid(sign_logit)                # (M,N)
            p_adv = p_member.mean(0)                            # (N,)
            mu_mean = mu.mean(0) * self.y_scale + self.y_center
            epi = (mu.var(0, unbiased=False) * self.y_scale ** 2)
            ale = ((scale * self.y_scale) ** 2).mean(0)
            def H(p):
                p = p.clamp(1e-6, 1 - 1e-6)
                return -(p * p.log() + (1 - p) * (1 - p).log())
            mi = (H(p_adv) - H(p_member).mean(0)).clamp_min(0.0)  # epistemic MI
        def n(t):
            return t.detach().cpu().numpy()
        return {"p_adverse": n(p_adv), "mu": n(mu_mean),
                "epistemic_var": n(epi), "aleatoric_var": n(ale), "epistemic_mi": n(mi)}


# --------------------------------------------------------------------------- #
# inverse-free Mahalanobis OOD (shrinkage + eigh), fit on train only
# --------------------------------------------------------------------------- #
def fit_whitener(X: np.ndarray, device: str = "cuda:0", shrinkage: float = 0.05,
                 strict: bool = False):
    import torch
    dev = _pick_device(device)
    if strict and dev != "cuda:0":
        raise RuntimeError("strict whitener requires device='cuda:0'")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    values = np.asarray(X, np.float32)
    if values.ndim != 2 or not len(values) or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("whitener input must be a non-empty finite matrix")
    x = torch.tensor(values, device=dev)
    assert_rocm_stage("fit-whitener", x, strict=strict)
    mean = x.mean(0)
    xc = x - mean
    n = max(x.shape[0] - 1, 1)
    cov = (xc.T @ xc) / n
    d = cov.shape[0]
    avg = cov.diag().mean().clamp_min(1e-12)
    cov = (1 - shrinkage) * cov + shrinkage * avg * torch.eye(
        d, device=dev, dtype=torch.float32
    )
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(avg * 1e-8)
    whitener = evecs * evals.rsqrt()             # no explicit inverse
    return mean, whitener, d


class TorchOOD:
    def __init__(self, X: np.ndarray, device: str = "cuda:0", shrinkage: float = 0.05,
                 strict: bool = False):
        self.device = _pick_device(device)
        self.strict = strict
        self.mean, self.whitener, self.d = fit_whitener(
            X, self.device, shrinkage, strict=strict
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch
        values = np.asarray(X, np.float32)
        if values.ndim != 2 or values.shape[1] != self.d or not np.isfinite(values).all():
            raise ValueError("OOD query must be a finite aligned matrix")
        q = torch.tensor(values, device=self.device)
        assert_rocm_stage(
            "torch-ood-score", q, self.mean, self.whitener,
            strict=self.strict,
        )
        z = (q - self.mean) @ self.whitener
        m = z.square().sum(-1).clamp_min(0.0)
        score = 1.0 - torch.exp(-m / (2.0 * self.d))
        return score.clamp(max=1.0 - torch.finfo(score.dtype).eps).cpu().numpy()


# --------------------------------------------------------------------------- #
# chunked GPU block-bootstrap with Monte-Carlo standard error
# --------------------------------------------------------------------------- #
def gpu_bootstrap(daily_returns: np.ndarray, *, n_boot: int = 10000, block: int = 20,
                  periods: int = 252, device: str = "cuda:0", seed: int = 7,
                  chunk: int = 2000, strict: bool = False) -> dict[str, float]:
    import torch
    dev = _pick_device(device)
    if strict and dev != "cuda:0":
        raise RuntimeError("strict bootstrap requires device='cuda:0'")
    values = np.asarray(daily_returns, np.float32).reshape(-1)
    if n_boot <= 0 or block <= 0 or periods <= 0 or chunk <= 0:
        raise ValueError("bootstrap sizes and periods must be positive")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap returns must be non-empty and finite")
    r = torch.tensor(values, device=dev)
    assert_rocm_stage("gpu-bootstrap", r, strict=strict)
    N = r.shape[0]
    if N < block + 2:
        return {"sharpe": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_gt0": float("nan")}
    n_blocks = math.ceil(N / block)
    offs = torch.arange(block, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    sharpes = []
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        starts = torch.randint(0, N, (b, n_blocks), generator=g).to(dev)
        idx = (starts.unsqueeze(-1) + offs).reshape(b, -1)[:, :N] % N
        samp = r[idx]
        mu = samp.mean(1)
        sd = samp.std(1, unbiased=True).clamp_min(1e-12)
        sharpes.append((mu / sd) * math.sqrt(periods))
        done += b
    s = torch.cat(sharpes)
    q = torch.quantile(s, torch.tensor([0.025, 0.975], device=dev))
    point = float((r.mean() / r.std(unbiased=True).clamp_min(1e-12)) * math.sqrt(periods))
    mc_se = float(s.std(unbiased=True) / math.sqrt(s.numel()))
    return {"sharpe": round(point, 4), "lo": round(float(q[0]), 4), "hi": round(float(q[1]), 4),
            "p_gt0": round(float((s > 0).float().mean()), 4), "mc_se": round(mc_se, 5),
            "n_boot": int(s.numel()), "block": block}
