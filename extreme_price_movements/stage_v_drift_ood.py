"""Stage-V grouped-MDA drift and OOD context.

This is deliberately a *context* builder, not an admission rule.  It turns
the groups retained by Stage-I grouped MDA into compact, row-local features
that describe whether economically relevant inputs are jointly active and how
far that activation is from a training-only reference distribution.

The state is side/layer scoped and serialisable.  ``fit`` is used for an OOS
fold or live model (the reference is training rows only); ``prequential`` is
available for research ledgers and never lets a row see its own or later
timestamp.  Neither function reads outcomes, ranks, or any future path data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .model_drift_features import fit_model_drift_state, transform_model_drift_features


STAGE_V_SCHEMA = "stage_v_grouped_mda_drift_ood_v1"
STAGE_V_FEATURE_COLUMNS: tuple[str, ...] = (
    "stage_v_reference_ready",
    "stage_v_mda_abs_z_mean",
    "stage_v_mda_abs_z_max",
    "stage_v_mda_tail_share",
    "stage_v_group_activation_mean",
    "stage_v_group_activation_max",
    "stage_v_group_coactivation_mean",
    "stage_v_group_coactivation_max",
    "stage_v_group_pattern_ood",
    "stage_v_group_drift_mean",
    "stage_v_group_drift_max",
    "stage_v_model_drift",
    "stage_v_ood_score",
)


@dataclass(frozen=True)
class StageVContract:
    """Prevent mixing a long/base reference into another model cell."""

    side: str
    layer: str

    def __post_init__(self) -> None:
        if str(self.side).lower() not in {"long", "short"}:
            raise ValueError("Stage V side must be 'long' or 'short'")
        if str(self.layer).lower() not in {"base", "meta"}:
            raise ValueError("Stage V layer must be 'base' or 'meta'")

    def normalized(self) -> "StageVContract":
        return StageVContract(str(self.side).lower(), str(self.layer).lower())


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        if col in frame.columns:
            out[str(col)] = pd.to_numeric(frame[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        for key in ("group_audit", "mda_group_audit", "groups"):
            if key in value:
                return _as_frame(value[key])
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


def _members(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return list(dict.fromkeys(str(v) for v in value if str(v).strip()))
    return list(dict.fromkeys(v for v in str(value or "").split("|") if v.strip()))


def resolve_stage_v_mda_groups(
    mda_audit: Any,
    *,
    available_columns: Sequence[str],
    max_groups: int = 24,
) -> list[dict[str, Any]]:
    """Extract positive grouped-MDA groups from the existing audit artifact.

    The lower bound is used when present; this means a group which only looked
    good due to MDA noise cannot become a Stage-V context feature.  Singleton
    groups are intentionally excluded: Stage V is about co-activation, while
    the base model already has the individual selected inputs.
    """
    audit = _as_frame(mda_audit)
    if audit.empty:
        return []
    available = {str(c) for c in available_columns}
    rows: list[dict[str, Any]] = []
    for pos, (_, row) in enumerate(audit.iterrows()):
        kind = str(row.get("group_kind", "correlation")).lower()
        if kind not in {"correlation", "group", "mda_group"}:
            continue
        lower = pd.to_numeric(pd.Series([row.get("group_mda_lower_95", row.get("mda_lower_95", np.nan))]), errors="coerce").iloc[0]
        mean = pd.to_numeric(pd.Series([row.get("group_mda_mean", row.get("mda_mean", 0.0))]), errors="coerce").iloc[0]
        if not np.isfinite(lower):
            lower = mean
        if not np.isfinite(lower) or float(lower) <= 0.0:
            continue
        members = [c for c in _members(row.get("features", row.get("members", ""))) if c in available]
        if len(members) < 2:
            continue
        rows.append(
            {
                "group_id": str(row.get("group_id", f"mda_group_{pos:03d}")),
                "members": members,
                "importance": float(max(float(lower), 1e-8)),
            }
        )
    # Multiple folds may repeat the same id/members.  Keep the best conservative
    # lower bound, and avoid a feature belonging to several near-duplicate groups.
    dedup: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(sorted(row["members"]))
        if key not in dedup or row["importance"] > dedup[key]["importance"]:
            dedup[key] = row
    result = sorted(dedup.values(), key=lambda r: (-r["importance"], r["group_id"]))
    return result[: max(0, int(max_groups))]


def _robust_stats(x: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    median = x.median(axis=0, skipna=True).fillna(0.0)
    q25 = x.quantile(0.25).fillna(median)
    q75 = x.quantile(0.75).fillna(median)
    scale = (q75 - q25).abs().replace(0.0, np.nan)
    scale = scale.fillna(x.std(axis=0, skipna=True)).replace(0.0, np.nan).fillna(1.0)
    tail = ((x.fillna(median) - median).abs() / scale).quantile(0.80).fillna(1.0)
    return median.astype(float), scale.astype(float), tail.astype(float)


def _derive_coactivation_groups(
    matrix: pd.DataFrame,
    mda_groups: Sequence[Mapping[str, Any]],
    median: pd.Series,
    scale: pd.Series,
    *,
    threshold: float = 0.70,
) -> list[dict[str, Any]]:
    """Merge MDA groups whose *training-only activation* fires together.

    Correlation groups protect MDA from redundant features.  This second, much
    smaller grouping is different: it uses only the soft activation of those
    retained MDA groups and identifies context patterns that occur together.
    It is fitted on training rows, never on the OOS/live batch.
    """
    if not mda_groups:
        return []
    activation, _, _ = _group_matrix(matrix, mda_groups, median, scale)
    n_groups = activation.shape[1]
    parent = np.arange(n_groups, dtype=np.int32)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return int(i)

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    if n_groups > 1 and len(matrix) >= 8:
        corr = np.corrcoef(activation, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        rows, cols = np.where(np.triu(corr >= float(threshold), k=1))
        for a, b in zip(rows, cols):
            union(int(a), int(b))
    components: dict[int, list[int]] = {}
    for i in range(n_groups):
        components.setdefault(find(i), []).append(i)
    result: list[dict[str, Any]] = []
    for order, members in enumerate(components.values()):
        source = [mda_groups[i] for i in members]
        features = list(dict.fromkeys(c for group in source for c in group["members"]))
        result.append(
            {
                "group_id": f"coactive_{order:03d}",
                "members": features,
                "importance": float(sum(float(group["importance"]) for group in source)),
                "source_mda_group_ids": [str(group["group_id"]) for group in source],
            }
        )
    return sorted(result, key=lambda row: (-row["importance"], row["group_id"]))


def _group_matrix(
    matrix: pd.DataFrame,
    groups: Sequence[Mapping[str, Any]],
    median: pd.Series,
    scale: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return activation, soft all-members-fire coactivation and disagreement."""
    n = len(matrix)
    if not groups:
        empty = np.zeros((n, 0), dtype=np.float32)
        return empty, empty, empty
    activation: list[np.ndarray] = []
    coactivation: list[np.ndarray] = []
    disagreement: list[np.ndarray] = []
    for group in groups:
        members = [c for c in group["members"] if c in matrix.columns]
        values = matrix.reindex(columns=members).fillna(median.reindex(members).fillna(0.0))
        z = ((values - median.reindex(members)) / scale.reindex(members).replace(0.0, 1.0)).to_numpy(dtype=np.float32)
        # A differentiable gate: 0 near the training centre, 1 when a member
        # is materially active.  ``min`` is a conservative soft "all fire"
        # summary and cannot be accidentally used as a hard router.
        active = 1.0 / (1.0 + np.exp(-((np.abs(z) - 1.0) / 0.5)))
        activation.append(np.mean(active, axis=1))
        coactivation.append(np.min(active, axis=1))
        disagreement.append(np.std(active, axis=1))
    return (
        np.column_stack(activation).astype(np.float32),
        np.column_stack(coactivation).astype(np.float32),
        np.column_stack(disagreement).astype(np.float32),
    )


def fit_stage_v_drift_ood_state(
    reference: pd.DataFrame,
    *,
    contract: StageVContract,
    mda_audit: Any,
    feature_columns: Sequence[str] | None = None,
    max_groups: int = 24,
) -> dict[str, Any]:
    """Fit a side/layer-specific state on training rows only.

    ``reference`` is intentionally the sole source of distribution parameters.
    The result records that fact so an OOS writer can audit it without trusting
    call-site conventions.
    """
    normalized = contract.normalized()
    candidates = [str(c) for c in (feature_columns or reference.columns)]
    matrix = _numeric(reference, candidates)
    finite = matrix.notna().mean(axis=0)
    variance = matrix.var(axis=0, skipna=True).fillna(0.0)
    usable = [c for c in matrix.columns if float(finite[c]) >= 0.5 and float(variance[c]) > 1e-12]
    mda_groups = resolve_stage_v_mda_groups(mda_audit, available_columns=usable, max_groups=max_groups)
    used = list(dict.fromkeys(c for group in mda_groups for c in group["members"]))
    if len(matrix) < 8 or not used or not mda_groups:
        return {
            "enabled": False,
            "schema": STAGE_V_SCHEMA,
            "contract": asdict(normalized),
            "reference_role": "train_only",
            "reference_rows": int(len(matrix)),
            "reason": "insufficient_mda_groups_or_reference_rows",
        }
    matrix = matrix.reindex(columns=used)
    median, scale, tail = _robust_stats(matrix)
    groups = _derive_coactivation_groups(matrix, mda_groups, median, scale)
    activation, coactivation, disagreement = _group_matrix(matrix, groups, median, scale)
    group_reference = {
        "activation_median": np.median(activation, axis=0).astype(float).tolist(),
        "activation_scale": np.maximum(np.subtract(*np.quantile(activation, [0.75, 0.25], axis=0)), 1e-4).astype(float).tolist(),
        "coactivation_median": np.median(coactivation, axis=0).astype(float).tolist(),
        "coactivation_scale": np.maximum(np.subtract(*np.quantile(coactivation, [0.75, 0.25], axis=0)), 1e-4).astype(float).tolist(),
        "disagreement_median": np.median(disagreement, axis=0).astype(float).tolist(),
        "disagreement_scale": np.maximum(np.subtract(*np.quantile(disagreement, [0.75, 0.25], axis=0)), 1e-4).astype(float).tolist(),
    }
    return {
        "enabled": True,
        "schema": STAGE_V_SCHEMA,
        "contract": asdict(normalized),
        "reference_role": "train_only",
        "reference_rows": int(len(matrix)),
        "feature_columns": used,
        "median": median.to_dict(),
        "scale": scale.to_dict(),
        "tail_z80": tail.to_dict(),
        "groups": groups,
        "mda_groups": mda_groups,
        "coactivation_fit": {
            "source": "training_only_mda_group_activation",
            "correlation_threshold": 0.70,
        },
        "group_reference": group_reference,
        # Reuse the established row-local diagnostic, fitted on precisely the
        # same training features.  No batch covariance is used at transform.
        "model_drift_state": fit_model_drift_state(matrix, feature_columns=used),
        "soft_context_only": True,
    }


def _empty_features(index: pd.Index, ready: float = 0.0) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=index, columns=STAGE_V_FEATURE_COLUMNS, dtype=np.float32)
    result["stage_v_reference_ready"] = np.float32(ready)
    return result


def transform_stage_v_drift_ood_features(
    frame: pd.DataFrame,
    state: Mapping[str, Any] | None,
    *,
    contract: StageVContract,
) -> pd.DataFrame:
    """Build soft row-local Stage-V context from a frozen training state."""
    normalized = contract.normalized()
    if not isinstance(state, Mapping) or not bool(state.get("enabled", False)):
        return _empty_features(frame.index)
    state_contract = state.get("contract", {})
    if not isinstance(state_contract, Mapping) or {
        "side": str(state_contract.get("side", "")).lower(),
        "layer": str(state_contract.get("layer", "")).lower(),
    } != asdict(normalized):
        raise ValueError("Stage V state cannot be used across side/layer contracts")
    columns = [str(c) for c in state.get("feature_columns", [])]
    groups = state.get("groups", []) or []
    if not columns or not groups:
        return _empty_features(frame.index)
    matrix = _numeric(frame, columns).reindex(columns=columns)
    median = pd.Series(state.get("median", {}), dtype=float).reindex(columns).fillna(0.0)
    scale = pd.Series(state.get("scale", {}), dtype=float).reindex(columns).replace(0.0, np.nan).fillna(1.0)
    tail = pd.Series(state.get("tail_z80", {}), dtype=float).reindex(columns).fillna(1.0)
    z = ((matrix.fillna(median) - median) / scale).to_numpy(dtype=np.float32)
    abs_z = np.abs(z)
    activation, coactivation, disagreement = _group_matrix(matrix, groups, median, scale)
    ref = state.get("group_reference", {})
    def _ref(name: str, width: int, default: float) -> np.ndarray:
        arr = np.asarray(ref.get(name, [default] * width), dtype=np.float32)
        return arr if len(arr) == width else np.full(width, default, dtype=np.float32)
    a_med, a_scale = _ref("activation_median", activation.shape[1], 0.0), _ref("activation_scale", activation.shape[1], 1.0)
    c_med, c_scale = _ref("coactivation_median", coactivation.shape[1], 0.0), _ref("coactivation_scale", coactivation.shape[1], 1.0)
    d_med, d_scale = _ref("disagreement_median", disagreement.shape[1], 0.0), _ref("disagreement_scale", disagreement.shape[1], 1.0)
    group_drift = 0.5 * np.abs((activation - a_med) / np.maximum(a_scale, 1e-4)) + 0.5 * np.abs((coactivation - c_med) / np.maximum(c_scale, 1e-4))
    pattern_ood = np.mean(np.abs((disagreement - d_med) / np.maximum(d_scale, 1e-4)), axis=1)
    raw_drift = transform_model_drift_features(matrix, state.get("model_drift_state"), index=frame.index)
    model_drift = raw_drift.get("row_drift_v1_inference_drift_score", pd.Series(0.0, index=frame.index)).to_numpy(dtype=np.float32)
    out = _empty_features(frame.index, ready=1.0)
    out["stage_v_mda_abs_z_mean"] = np.mean(abs_z, axis=1).astype(np.float32)
    out["stage_v_mda_abs_z_max"] = np.max(abs_z, axis=1).astype(np.float32)
    out["stage_v_mda_tail_share"] = np.mean(abs_z > tail.to_numpy(dtype=np.float32), axis=1).astype(np.float32)
    out["stage_v_group_activation_mean"] = np.mean(activation, axis=1).astype(np.float32)
    out["stage_v_group_activation_max"] = np.max(activation, axis=1).astype(np.float32)
    out["stage_v_group_coactivation_mean"] = np.mean(coactivation, axis=1).astype(np.float32)
    out["stage_v_group_coactivation_max"] = np.max(coactivation, axis=1).astype(np.float32)
    out["stage_v_group_pattern_ood"] = np.tanh(pattern_ood / 3.0).astype(np.float32)
    out["stage_v_group_drift_mean"] = np.tanh(np.mean(group_drift, axis=1) / 3.0).astype(np.float32)
    out["stage_v_group_drift_max"] = np.tanh(np.max(group_drift, axis=1) / 3.0).astype(np.float32)
    out["stage_v_model_drift"] = np.clip(model_drift, 0.0, 1.0)
    out["stage_v_ood_score"] = np.clip(
        0.35 * out["stage_v_mda_tail_share"].to_numpy()
        + 0.30 * out["stage_v_group_drift_max"].to_numpy()
        + 0.20 * out["stage_v_group_pattern_ood"].to_numpy()
        + 0.15 * out["stage_v_model_drift"].to_numpy(),
        0.0,
        1.0,
    ).astype(np.float32)
    return out.reindex(columns=STAGE_V_FEATURE_COLUMNS).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def attach_stage_v_context(
    ledger: pd.DataFrame,
    context: pd.DataFrame,
    *,
    candidate_ids: Sequence[Any],
) -> pd.DataFrame:
    """Append context to an OOF ledger without changing its score or ranking.

    Pooled-global top-k selection belongs to the experiment runner and must
    happen *after* a model consumes this context.  This helper deliberately
    contains no rank, threshold, side reweighting, or admission operation.
    """
    ids = np.asarray(candidate_ids, dtype=object).reshape(-1)
    if len(ledger) != len(context) or len(ids) != len(ledger):
        raise ValueError("Stage V ledger attachment requires row-aligned inputs")
    if len(pd.unique(ids)) != len(ids):
        raise ValueError("Stage V ledger attachment requires unique candidate_ids")
    if any(col not in context.columns for col in STAGE_V_FEATURE_COLUMNS):
        raise ValueError("Stage V ledger attachment requires the complete context contract")
    out = ledger.copy()
    if "candidate_id" in out.columns and not np.array_equal(out["candidate_id"].to_numpy(dtype=object), ids):
        raise ValueError("Stage V candidate_ids do not match the OOF ledger order")
    for col in STAGE_V_FEATURE_COLUMNS:
        out[col] = context[col].to_numpy(dtype=np.float32)
    return out


def prequential_stage_v_drift_ood_features(
    frame: pd.DataFrame,
    *,
    timestamps: Sequence[Any],
    contract: StageVContract,
    mda_audit: Any,
    feature_columns: Sequence[str] | None = None,
    initial_reference: pd.DataFrame | None = None,
    min_reference_rows: int = 64,
    refresh_every_timestamps: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prequential context with whole-timestamp, prior-only references.

    It returns features and a compact provenance table.  A refresh at timestamp
    ``t`` sees only rows whose timestamp is strictly less than ``t``.  Thus
    same-bar candidates cannot leak information to each other.  This path is
    intended for OOF research; live/OOS scoring should normally call ``fit`` on
    the model's frozen training partition then ``transform`` once.
    """
    ts = pd.to_datetime(pd.Series(timestamps, index=frame.index), utc=True, errors="coerce")
    if len(ts) != len(frame) or ts.isna().any():
        raise ValueError("Stage V prequential transform requires aligned finite UTC timestamps")
    ordered_index = ts.sort_values(kind="mergesort").index
    out = _empty_features(frame.index)
    audit_rows: list[dict[str, Any]] = []
    history = initial_reference.copy() if initial_reference is not None else pd.DataFrame(columns=frame.columns)
    current_state: dict[str, Any] | None = None
    unique_ts = pd.Index(ts.loc[ordered_index].unique())
    refresh = max(1, int(refresh_every_timestamps))
    for ordinal, stamp in enumerate(unique_ts):
        idx = ts.index[ts.eq(stamp)]
        if ordinal % refresh == 0 or current_state is None:
            if len(history) >= int(min_reference_rows):
                current_state = fit_stage_v_drift_ood_state(
                    history,
                    contract=contract,
                    mda_audit=mda_audit,
                    feature_columns=feature_columns,
                )
            else:
                current_state = None
        if current_state is not None:
            out.loc[idx] = transform_stage_v_drift_ood_features(frame.loc[idx], current_state, contract=contract)
        audit_rows.append({
            "timestamp": stamp,
            "side": contract.normalized().side,
            "layer": contract.normalized().layer,
            "reference_rows": int(len(history)),
            "reference_max_ts": (pd.to_datetime(history["__stage_v_ts__"], utc=True).max() if "__stage_v_ts__" in history and len(history) else pd.NaT),
            "strictly_prior_reference": True,
            "state_enabled": bool(current_state and current_state.get("enabled", False)),
        })
        append = frame.loc[idx].copy()
        append["__stage_v_ts__"] = stamp
        # Avoid concatenating against an all-NA schema-only frame (and keep
        # the first reference batch's dtypes intact for the numeric adapter).
        history = (
            append.reset_index(drop=True)
            if history.empty
            else pd.concat([history, append], axis=0, ignore_index=True)
        )
    return out.reindex(columns=STAGE_V_FEATURE_COLUMNS).astype(np.float32), pd.DataFrame(audit_rows)
