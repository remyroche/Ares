"""Frozen post-selection OOD features for the S52 shared meta champion.

The retained shared meta contract uses two OOD aggregates.  They are derived
from the selected pre-OOD inputs and therefore must use the exact train-fitted
reference at inference rather than an ad hoc live-window normalization.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


OOD_FEATURE_NAMES = (
    "meta_sel_ood_abs_z_mean",
    "meta_sel_ood_abs_z_max",
    "meta_sel_ood_abs_z_p95",
    "meta_sel_ood_iqr_exceed_frac",
    "meta_sel_ood_missing_frac",
    "meta_sel_ood_centroid_l2",
)


def fit_s52_meta_ood_reference(
    frame: pd.DataFrame, feature_names: Sequence[str]
) -> dict[str, Any]:
    """Fit the train-only reference used by the shared S52 meta contract."""
    columns = [str(col) for col in feature_names if str(col) in frame.columns]
    if len(columns) < 3:
        return {"enabled": False, "reason": "insufficient_input_features"}
    values = (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, copy=True)
    )
    finite = np.isfinite(values)
    safe = np.where(finite, values, np.nan).astype(np.float32, copy=False)
    mean = np.nanmean(safe, axis=0).astype(np.float32)
    std = np.nanstd(safe, axis=0).astype(np.float32)
    q25 = np.nanquantile(safe, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(safe, 0.75, axis=0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    q25 = np.nan_to_num(q25, nan=mean).astype(np.float32)
    q75 = np.nan_to_num(q75, nan=mean).astype(np.float32)
    return {
        "enabled": True,
        "schema_version": "s52_meta_post_selection_ood_v1",
        "feature_names": columns,
        "mean": mean,
        "std": std,
        "q25": q25,
        "q75": q75,
        "fit_rows": int(len(frame)),
    }


def append_s52_meta_ood_features(
    frame: pd.DataFrame,
    reference: Mapping[str, Any] | None,
    *,
    output_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Materialize requested OOD aggregates using a frozen train reference."""
    out = frame.copy()
    requested = set(output_features or OOD_FEATURE_NAMES)
    if not reference or not bool(reference.get("enabled", False)):
        for name in requested.intersection(OOD_FEATURE_NAMES):
            out[name] = np.float32(0.0)
        return out
    columns = [str(col) for col in reference.get("feature_names", []) if str(col)]
    if not columns:
        return out
    values = (
        out.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, copy=True)
    )
    finite = np.isfinite(values)
    mean = np.asarray(reference.get("mean", []), dtype=np.float32)
    std = np.asarray(reference.get("std", []), dtype=np.float32)
    q25 = np.asarray(reference.get("q25", []), dtype=np.float32)
    q75 = np.asarray(reference.get("q75", []), dtype=np.float32)
    if any(len(values.shape) != 2 for values in (values,)) or len(mean) != values.shape[1]:
        raise ValueError("S52 meta OOD reference does not match its feature contract")
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    q25 = np.where(np.isfinite(q25), q25, mean).astype(np.float32)
    q75 = np.where(np.isfinite(q75), q75, mean).astype(np.float32)
    filled = np.where(finite, values, mean[None, :]).astype(np.float32, copy=False)
    z = (filled - mean[None, :]) / std[None, :]
    abs_z = np.abs(z).astype(np.float32, copy=False)
    iqr = np.maximum(q75 - q25, 1e-6).astype(np.float32)
    exceed = ((filled < (q25 - 1.5 * iqr)) | (filled > (q75 + 1.5 * iqr))) & finite
    metrics = {
        "meta_sel_ood_abs_z_mean": np.mean(abs_z, axis=1).astype(np.float32),
        "meta_sel_ood_abs_z_max": np.max(abs_z, axis=1).astype(np.float32),
        "meta_sel_ood_abs_z_p95": np.quantile(abs_z, 0.95, axis=1).astype(np.float32),
        "meta_sel_ood_iqr_exceed_frac": np.mean(exceed, axis=1).astype(np.float32),
        "meta_sel_ood_missing_frac": np.mean(~finite, axis=1).astype(np.float32),
        "meta_sel_ood_centroid_l2": np.sqrt(np.mean(z * z, axis=1)).astype(np.float32),
    }
    for name in requested.intersection(OOD_FEATURE_NAMES):
        out[name] = metrics[name]
    return out
