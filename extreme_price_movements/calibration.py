from __future__ import annotations

import numpy as np


def safe_clip_proba(p, eps: float = 1e-6):
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def logit(p, eps: float = 1e-6):
    p = safe_clip_proba(p, eps=eps)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def compute_prevalences(y, w=None):
    y = np.asarray(y, dtype=np.float64)
    p_unweighted = float(np.mean(y)) if y.size else 0.5
    if w is None:
        return p_unweighted, p_unweighted
    w = np.asarray(w, dtype=np.float64)
    den = float(np.sum(w))
    if den <= 1e-12:
        return p_unweighted, p_unweighted
    p_weighted = float(np.sum(w * y) / den)
    return p_unweighted, p_weighted


def compute_logit_shift(p_unweighted: float, p_weighted: float, eps: float = 1e-6):
    return float(logit(p_unweighted, eps=eps) - logit(p_weighted, eps=eps))


def apply_logit_shift(p_raw, delta_logit: float, eps: float = 1e-6):
    p_raw = safe_clip_proba(p_raw, eps=eps)
    z = logit(p_raw, eps=eps) + float(delta_logit)
    return safe_clip_proba(sigmoid(z), eps=eps)
