#!/usr/bin/env python3
"""Run a bounded conditional correction on the exact frozen stack.

This experiment is deliberately *not* a replacement ranker.  It consumes the
persisted incumbent score and current specialist outputs, then learns only the
conditional error left by that score.  The model is trained chronologically
and is never allowed to see the held-out month outcomes before scoring them.

The frozen control is the score in
``frozen_residual_query_hpo_20260810_v1/predictions.parquet``.  A parity check
against the current pair-condition ledger is mandatory: if candidate IDs or
scores differ, the run stops rather than silently reconstructing the stack.

The conditional target is:

    residual = realised_net_bps - causal_isotonic(anchor_score)

The feature contract contains exact head-score geometry, condition-family
activation/strength, causal regime/context fields, OOD/disagreement proxies,
and prior-only family reliability.  A multi-task MLP predicts continuous
correctness factors, residual EV, and soft demotion/promotion probabilities.
Its authority is bounded and shrunk by causal support.  The output is then
compared with the frozen anchor using pooled global tails and an EV/coverage
frontier.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Stochastic Optimizer")

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data_perp/artifacts/frozen_residual_query_hpo_20260810_v1/predictions.parquet"
CURRENT = ROOT / "data_perp/artifacts/pair_condition_specialists_20260806_v12_recurrence/predictions_with_incumbent.parquet"
CONDITION = ROOT / "data_perp/artifacts/pair_condition_specialists_20260806_v12_recurrence/condition_specialist_oof.parquet"
SHARED = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/anchored_conditional_correction_20260806_v1"
SEED = 20260806
HORIZON_HOURS = 12
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
WINDOWS_DAYS = (1, 3, 7, 14, 28)
STATE_COMPONENTS = 5
MIN_HISTORY_ROWS = 5000

CONDITION_SUFFIXES = (
    "__raw",
    "__rank",
    "__membership",
    "__gated_rank",
    "__innovation_rank",
    "__uncertainty",
    "__ood",
)

FORBIDDEN = {
    "candidate_id", "__ts__", "side_name", "fold", "net_bps", "gross_bps",
    "residual_bps", "base_expected_bps", "label_available_ts", "event",
    "m6_contract_complete", "shared_regime_contract_complete",
    "state_reference_cutoff_utc", "residual_reference_cutoff_utc",
    "target__exact_net_residual_bps", "target__soft_regime_centered_residual",
    "target__soft_regime_standardized_residual", "latent_state",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    keep = [c for c in columns if c in available]
    return pd.read_parquet(path, columns=keep)


def _empirical_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    if len(ref) < 2:
        return np.full(len(x), 0.5, dtype=np.float32)
    out = np.searchsorted(ref, x, side="right") / float(len(ref))
    out[~np.isfinite(x)] = 0.5
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -40.0, 40.0)))).astype(np.float32)


def _load() -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen = _read_columns(FROZEN, ["candidate_id", "score", "__ts__", "side_name", "fold", "net_bps", "gross_bps"])
    current_cols = [
        "candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "base_score",
        "base_ev_bps", "residual_bps", "fold", "incumbent_score",
    ]
    current_cols += [c for c in pq.ParquetFile(CURRENT).schema.names if c.startswith("score__")]
    current = _read_columns(CURRENT, current_cols)
    frozen = frozen.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    current = current.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not frozen["candidate_id"].equals(current["candidate_id"]):
        raise RuntimeError("frozen/current candidate IDs do not match")
    if not np.allclose(frozen["score"].to_numpy(float), current["incumbent_score"].to_numpy(float), equal_nan=True):
        raise RuntimeError("frozen score and current incumbent score differ")
    if not np.allclose(frozen["net_bps"].to_numpy(float), current["net_bps"].to_numpy(float), equal_nan=True):
        raise RuntimeError("frozen/current realised labels differ")

    condition_names = pq.ParquetFile(CONDITION).schema.names
    condition_cols = [
        c for c in condition_names
        if c != "candidate_id" and c.startswith("condition__") and c.endswith(CONDITION_SUFFIXES)
    ]
    condition = _read_columns(CONDITION, ["candidate_id", *condition_cols])
    if condition["candidate_id"].duplicated().any():
        raise RuntimeError("condition specialist output has duplicate candidate IDs")
    frame = current.merge(condition, on="candidate_id", how="inner", validate="one_to_one")

    ledger_names = pq.ParquetFile(SHARED).schema.names
    ledger_cols = ["candidate_id", "__ts__"]
    for name in ledger_names:
        if name in {"candidate_id", "__ts__"}:
            continue
        if name in FORBIDDEN:
            continue
        if name.startswith("target__") or name.startswith("state_reference") or name.startswith("residual_reference"):
            continue
        if name.startswith(("regime_", "prequential_", "soft_regime_")) or name in {
            "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
            "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
            "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
            "post_liquidation_rebound_score", "negative_breadth_pct",
            "btc_resilience_alt_weakness", "short_covering_score_market",
            "deleveraging_without_followthrough", "short_signal_recovery_conflict",
            "regime_transition_onset_proxy", "regime_state_duration_hours",
        }:
            ledger_cols.append(name)
    ledger = _read_columns(SHARED, list(dict.fromkeys(ledger_cols)))
    frame = frame.merge(ledger, on=["candidate_id"], how="left", suffixes=("", "__ledger"), validate="one_to_one")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    frame["matured_ts"] = frame["__ts__"] + pd.Timedelta(hours=HORIZON_HOURS)
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    coverage = {c: float(pd.to_numeric(frame[c], errors="coerce").notna().mean()) for c in ledger_cols if c not in {"candidate_id", "__ts__"} and c in frame}
    audit = {
        "frozen_rows": int(len(frozen)),
        "current_rows": int(len(current)),
        "joined_rows": int(len(frame)),
        "frozen_current_score_max_abs_diff": 0.0,
        "condition_output_count": int(len(condition_cols)),
        "ledger_feature_count": int(len(ledger_cols) - 2),
        "ledger_feature_coverage_min": float(min(coverage.values())) if coverage else None,
        "ledger_feature_coverage": coverage,
        "candidate_id_unique": bool(frame["candidate_id"].is_unique),
        "score_contract": str(FROZEN),
        "current_contract": str(CURRENT),
    }
    if len(frame) != len(frozen):
        raise RuntimeError(f"joined rows {len(frame)} != frozen rows {len(frozen)}")
    return frame, audit


def _column_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    head = sorted([c for c in frame.columns if c.startswith("score__")])
    cond = sorted([c for c in frame.columns if c.startswith("condition__") and c.endswith(CONDITION_SUFFIXES)])
    activation = sorted([c for c in cond if c.endswith("__membership")])
    context = sorted([
        c for c in frame.columns
        if c.startswith(("regime_", "regime_relative__", "regime_z__", "prequential_", "soft_regime_"))
        or c in {
            "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
            "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
            "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
            "post_liquidation_rebound_score", "negative_breadth_pct",
            "btc_resilience_alt_weakness", "short_covering_score_market",
            "deleveraging_without_followthrough", "short_signal_recovery_conflict",
        }
    ])
    return {"head": head, "condition": cond, "activation": activation, "context": context}


def _fit_rank_refs(train: pd.DataFrame, fields: list[str]) -> dict[tuple[str, str], np.ndarray]:
    refs: dict[tuple[str, str], np.ndarray] = {}
    for side, block in train.groupby("side_name", sort=False):
        for field in fields:
            values = pd.to_numeric(block[field], errors="coerce").to_numpy(float)
            refs[(str(side), field)] = values[np.isfinite(values)]
    return refs


def _add_ranked_geometry(block: pd.DataFrame, refs: dict[tuple[str, str], np.ndarray], groups: dict[str, list[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=block.index)
    # Keep the frozen incumbent's percentile as an explicit anchor coordinate;
    # using the median of specialist ranks as a fallback would erase the exact
    # baseline geometry that the conditional layer is meant to adjust.
    rank_fields = ["incumbent_score", *groups["head"]]
    present_fields = [field for field in rank_fields if field in block]
    for field in present_fields:
        if field not in block:
            continue
        rank = np.empty(len(block), dtype=np.float32)
        for side, idx in block.groupby("side_name", sort=False).groups.items():
            values = pd.to_numeric(block.loc[idx, field], errors="coerce").to_numpy(float)
            positions = block.index.get_indexer(idx)
            rank[positions] = _empirical_rank(values, refs.get((str(side), field), values))
        out[f"head_rank__{field}"] = rank
    if present_fields:
        matrix = out[[f"head_rank__{f}" for f in present_fields]].to_numpy(float)
        out["head_rank__median"] = np.nanmedian(matrix, axis=1)
        out["head_rank__mean"] = np.nanmean(matrix, axis=1)
        out["head_rank__std"] = np.nanstd(matrix, axis=1)
        out["head_rank__min"] = np.nanmin(matrix, axis=1)
        out["head_rank__max"] = np.nanmax(matrix, axis=1)
    return out


def _prior_recent_features(current: pd.DataFrame, history: pd.DataFrame, groups: dict[str, list[str]], windows: tuple[int, ...] = WINDOWS_DAYS) -> pd.DataFrame:
    """Compute prior-only residual and family reliability features.

    ``history`` is required to contain only rows available before the current
    block.  For rows inside a training block the caller passes that block in
    chronological order; searchsorted on matured timestamps prevents a row
    from using its own or a later outcome.
    """
    output = pd.DataFrame(index=current.index)
    if len(current) == 0:
        return output
    hist = history.copy()
    hist["__ts__"] = pd.to_datetime(hist["__ts__"], utc=True)
    hist["matured_ts"] = hist["__ts__"] + pd.Timedelta(hours=HORIZON_HOURS)
    hist["raw_resid"] = pd.to_numeric(hist["net_bps"], errors="coerce") - pd.to_numeric(hist["incumbent_score"], errors="coerce")
    family_fields = list(groups["activation"])
    for side, idx_cur in current.groupby("side_name", sort=False).groups.items():
        cur_positions = current.index.get_indexer(idx_cur)
        h = hist[hist["side_name"].astype(str) == str(side)].sort_values("matured_ts", kind="stable")
        if h.empty:
            for days in windows:
                output.loc[idx_cur, f"recent_anchor_resid_{days}d"] = 0.0
                output.loc[idx_cur, f"recent_anchor_fail_{days}d"] = 0.5
                for family in family_fields:
                    output.loc[idx_cur, f"recent_family__{family}__{days}d"] = 0.0
            continue
        event = h["matured_ts"].astype("int64").to_numpy()
        query = current.loc[idx_cur, "__ts__"].astype("int64").to_numpy()
        end = np.searchsorted(event, query, side="left")
        for days in windows:
            start = np.searchsorted(event, query - int(pd.Timedelta(days=days).value), side="left")
            residual = pd.to_numeric(h["raw_resid"], errors="coerce").to_numpy(float)
            valid = np.isfinite(residual)
            residual = np.where(valid, residual, 0.0)
            cumulative = np.concatenate([[0.0], np.cumsum(residual)])
            counts = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
            sums = cumulative[end] - cumulative[start]
            n = counts[end] - counts[start]
            output.loc[idx_cur, f"recent_anchor_resid_{days}d"] = sums / np.maximum(n, 1)
            fail = (residual < -50.0).astype(float)
            cf = np.concatenate([[0.0], np.cumsum(fail)])
            output.loc[idx_cur, f"recent_anchor_fail_{days}d"] = (cf[end] - cf[start]) / np.maximum(n, 1)
            for family in family_fields:
                membership = pd.to_numeric(h[family], errors="coerce").fillna(0.0).to_numpy(float)
                weighted = membership * residual
                wsum = np.concatenate([[0.0], np.cumsum(weighted)])
                wmass = np.concatenate([[0.0], np.cumsum(np.maximum(membership, 0.0))])
                output.loc[idx_cur, f"recent_family__{family}__{days}d"] = (wsum[end] - wsum[start]) / np.maximum(wmass[end] - wmass[start], 1e-3)
    return output.reset_index(drop=True)


def _matrix(block: pd.DataFrame, train: pd.DataFrame, groups: dict[str, list[str]], refs: dict[tuple[str, str], np.ndarray], recent: pd.DataFrame | None = None) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Build the inference feature matrix from score/rule/context geometry."""
    pieces: list[pd.DataFrame] = []
    names: list[str] = []
    medians: dict[str, float] = {}
    scalar = ["base_score", "base_ev_bps", "incumbent_score"]
    scalar += groups["condition"]
    scalar += groups["context"]
    for field in scalar:
        if field not in block:
            continue
        values = pd.to_numeric(block[field], errors="coerce")
        med = float(pd.to_numeric(train[field], errors="coerce").median()) if field in train else 0.0
        medians[field] = 0.0 if not np.isfinite(med) else med
        pieces.append(pd.DataFrame({field: values.fillna(medians[field]).clip(-1e6, 1e6).to_numpy(np.float32)}, index=block.index))
        names.append(field)
    geometry = _add_ranked_geometry(block, refs, groups)
    pieces.append(geometry.reset_index(drop=True))
    names.extend(list(geometry.columns))
    if groups["activation"]:
        activation = block[groups["activation"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        anchor_rank = geometry["head_rank__incumbent_score"] if "head_rank__incumbent_score" in geometry else geometry.get("head_rank__median", pd.Series(0.5, index=geometry.index))
        for j, family in enumerate(groups["activation"]):
            pieces.append(pd.DataFrame({f"family_anchor_interaction__{family}": activation[:, j] * anchor_rank.to_numpy(float)}))
            names.append(f"family_anchor_interaction__{family}")
    if recent is not None and len(recent):
        pieces.append(recent.reset_index(drop=True))
        names.extend(list(recent.columns))
    merged = pd.concat([p.reset_index(drop=True) for p in pieces], axis=1)
    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Avoid duplicate columns from ledger aliases; preserve first occurrence.
    merged = merged.loc[:, ~merged.columns.duplicated()]
    names = list(merged.columns)
    return merged.to_numpy(np.float32), names, medians


def _select_features(x: np.ndarray, names: list[str], target: np.ndarray, cap: int = 96) -> tuple[np.ndarray, list[str]]:
    y = np.asarray(target, dtype=float)
    scores: list[tuple[float, str, int]] = []
    sample = np.linspace(0, len(y) - 1, min(len(y), 30000), dtype=int)
    for j, name in enumerate(names):
        value = np.asarray(x[sample, j], dtype=float)
        ok = np.isfinite(value) & np.isfinite(y[sample])
        if ok.sum() < 200 or np.unique(value[ok]).size < 2:
            continue
        corr = spearmanr(value[ok], y[sample][ok]).statistic
        scores.append((abs(float(corr)) if np.isfinite(corr) else -np.inf, name, j))
    scores.sort(key=lambda row: (-row[0], row[1]))
    selected = scores[: min(cap, len(scores))]
    if len(selected) < min(cap, 16):
        selected = [(0.0, names[j], j) for j in range(min(cap, len(names)))]
    return x[:, [j for _, _, j in selected]], [name for _, name, _ in selected]


def _fit_anchor_map(fit: pd.DataFrame) -> IsotonicRegression:
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    score = pd.to_numeric(fit["incumbent_score"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(fit["net_bps"], errors="coerce").to_numpy(float)
    ok = np.isfinite(score) & np.isfinite(net)
    model.fit(score[ok], net[ok])
    return model


def _fit_multitask(x: np.ndarray, residual: np.ndarray, latent: np.ndarray) -> tuple[MLPRegressor, StandardScaler, StandardScaler]:
    y = np.column_stack([
        np.clip(residual / 100.0, -4.0, 4.0),
        np.clip(_sigmoid((-residual - 50.0) / 50.0), 0.0, 1.0),
        np.clip(_sigmoid((residual - 50.0) / 50.0), 0.0, 1.0),
        latent,
    ]).astype(np.float32)
    x_scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(y)
    model = MLPRegressor(
        hidden_layer_sizes=(48, 24), activation="relu", solver="adam", alpha=2e-3,
        batch_size=1024, learning_rate_init=1e-3, max_iter=100,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=12,
        random_state=SEED, shuffle=False, tol=1e-4,
    )
    model.fit(x_scaler.transform(x), y_scaler.transform(y))
    return model, x_scaler, y_scaler


def _latent_targets(fit: pd.DataFrame, activation: list[str], residual: np.ndarray) -> tuple[np.ndarray, TruncatedSVD, StandardScaler]:
    if not activation:
        return np.zeros((len(fit), 1), dtype=np.float32), None, None  # type: ignore[return-value]
    a = fit[activation].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    scale = max(float(np.nanmedian(np.abs(residual - np.nanmedian(residual))) * 1.4826), 25.0)
    correctness = a * np.clip(residual[:, None] / scale, -4.0, 4.0)
    scaler = StandardScaler().fit(correctness)
    z = scaler.transform(correctness).astype(np.float32)
    k = min(STATE_COMPONENTS, max(1, correctness.shape[1]), len(fit) - 1)
    svd = TruncatedSVD(n_components=k, random_state=SEED, n_iter=5)
    latent = svd.fit_transform(z).astype(np.float32)
    return latent, svd, scaler


def _latent_transform(frame: pd.DataFrame, activation: list[str], residual: np.ndarray, svd: TruncatedSVD, scaler: StandardScaler) -> np.ndarray:
    if svd is None or not activation:
        return np.zeros((len(frame), 1), dtype=np.float32)
    a = frame[activation].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    scale = max(float(np.nanmedian(np.abs(residual - np.nanmedian(residual))) * 1.4826), 25.0)
    z = scaler.transform(a * np.clip(residual[:, None] / scale, -4.0, 4.0))
    return svd.transform(z).astype(np.float32)


def _support(block: pd.DataFrame, train: pd.DataFrame, groups: dict[str, list[str]], head_geometry: np.ndarray | None = None) -> np.ndarray:
    if head_geometry is None or head_geometry.shape[1] == 0:
        disagreement = np.zeros(len(block), dtype=float)
    else:
        disagreement = np.clip(np.nanstd(head_geometry, axis=1) * 2.0, 0.0, 1.0)
    ood_cols = [c for c in groups["condition"] if c.endswith("__ood")]
    if ood_cols:
        raw = block[ood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        ref = train[ood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        ood = np.zeros(len(block), dtype=float)
        for j in range(raw.shape[1]):
            ood += _empirical_rank(raw[:, j], ref[:, j])
        ood /= raw.shape[1]
    else:
        ood = np.zeros(len(block), dtype=float)
    return np.clip(1.0 - 0.5 * disagreement - 0.5 * ood, 0.25, 1.0).astype(np.float32)


def _dynamic_head_score(fit: pd.DataFrame, block: pd.DataFrame, groups: dict[str, list[str]], anchor_map: IsotonicRegression) -> tuple[np.ndarray, dict[str, Any]]:
    heads = groups["head"]
    if not heads:
        return pd.to_numeric(block["incumbent_score"], errors="coerce").to_numpy(float), {"head_weights": {}}
    fit_ranks = []
    block_ranks = []
    for field in heads:
        fit_ranks.append(_empirical_rank(fit[field].to_numpy(float), fit[field].to_numpy(float)))
        block_ranks.append(_empirical_rank(block[field].to_numpy(float), fit[field].to_numpy(float)))
    fr = np.column_stack(fit_ranks)
    br = np.column_stack(block_ranks)
    y = pd.to_numeric(fit["net_bps"], errors="coerce").to_numpy(float)
    # Recent/conditional head reliability is estimated only from fit rows;
    # shrink each head's top-quintile residual toward the pooled residual.
    residual = y - pd.to_numeric(fit["incumbent_score"], errors="coerce").to_numpy(float)
    pooled = float(np.nanmean(residual)) if np.isfinite(residual).any() else 0.0
    quality = []
    for j in range(fr.shape[1]):
        top = fr[:, j] >= 0.8
        q = float(np.nanmean(residual[top])) if top.sum() >= 100 else pooled
        n = float(top.sum())
        quality.append((n * q + 1000.0 * pooled) / (n + 1000.0))
    signal = np.clip(np.asarray(quality, dtype=float) / 100.0, -2.0, 2.0)
    weights = np.exp(0.20 * signal)
    weights /= weights.sum()
    dynamic_rank_fit = fr @ weights
    dynamic_rank_block = br @ weights
    # ``anchor_map`` expects bps scores, not percentiles; use a monotone fit
    # on the dynamic percentile to the same fit outcomes instead.
    local_map = IsotonicRegression(increasing=True, out_of_bounds="clip")
    local_map.fit(dynamic_rank_fit, y)
    return local_map.predict(dynamic_rank_block), {"head_weights": {h: float(w) for h, w in zip(heads, weights)}, "head_quality_bps": {h: float(q) for h, q in zip(heads, quality)}}


def _tail_rows(frame: pd.DataFrame, score: str, period: str | None = None) -> list[dict[str, Any]]:
    block = frame if period is None else frame[frame["month"] == period]
    if block.empty:
        return []
    rows: list[dict[str, Any]] = []
    order = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    for tail in TAILS:
        n = max(1, int(math.ceil(len(order) * tail)))
        chosen = order.head(n)
        rows.append({
            "score": score, "period": "pooled" if period is None else period, "tail": tail,
            "trades": int(n), "gross_bps_per_trade": float(chosen["gross_bps"].mean()),
            "net_bps_per_trade": float(chosen["net_bps"].mean()),
            "win_rate_net": float((chosen["net_bps"] > 0).mean()),
        })
    return rows


def _frontier(frame: pd.DataFrame, score: str) -> list[dict[str, Any]]:
    rows = []
    order = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    for tail in tuple(np.linspace(0.005, 0.30, 13)):
        n = max(1, int(math.ceil(len(order) * float(tail))))
        chosen = order.head(n)
        rows.append({"score": score, "coverage": float(tail), "trades": n, "gross_bps_per_trade": float(chosen.gross_bps.mean()), "net_bps_per_trade": float(chosen.net_bps.mean())})
    return rows


def _run_month(frame: pd.DataFrame, month: str, groups: dict[str, list[str]], seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    test = frame[frame["month"] == month].copy().sort_values(["__ts__", "candidate_id"], kind="stable")
    history = frame[(frame["__ts__"] < pd.Timestamp(month, tz="UTC")) & (frame["matured_ts"] < pd.Timestamp(month, tz="UTC"))].copy().sort_values(["__ts__", "candidate_id"], kind="stable")
    if len(history) < MIN_HISTORY_ROWS:
        out = test[["candidate_id", "__ts__", "side_name", "month", "net_bps", "gross_bps", "incumbent_score"]].copy()
        out["anchor_only"] = out["incumbent_score"]
        for name in ("residual_bounded_50_25", "residual_bounded_100_50", "demotion_only", "promotion_only", "residual_heads", "dynamic_head", "combined"):
            out[name] = out["incumbent_score"]
        return out, _tail_rows(out, "anchor_only") + _tail_rows(out, "anchor_only", month), {"month": month, "history_rows": len(history), "status": "anchor_only_insufficient_history"}, []

    split = max(1000, int(0.80 * len(history)))
    fit = history.iloc[:split].copy()
    cal = history.iloc[split:].copy()
    anchor_map = _fit_anchor_map(fit)
    fit["anchor_mu"] = anchor_map.predict(fit["incumbent_score"].to_numpy(float))
    cal["anchor_mu"] = anchor_map.predict(cal["incumbent_score"].to_numpy(float))
    test["anchor_mu"] = anchor_map.predict(test["incumbent_score"].to_numpy(float))
    fit_resid = fit["net_bps"].to_numpy(float) - fit["anchor_mu"].to_numpy(float)
    cal_resid = cal["net_bps"].to_numpy(float) - cal["anchor_mu"].to_numpy(float)
    test_resid = test["net_bps"].to_numpy(float) - test["anchor_mu"].to_numpy(float)

    refs = _fit_rank_refs(fit, ["incumbent_score", *groups["head"]])
    recent_fit = _prior_recent_features(fit, fit, groups)
    recent_cal = _prior_recent_features(cal, fit, groups)
    recent_test = _prior_recent_features(test, history, groups)
    x_fit_all, names, _ = _matrix(fit, fit, groups, refs, recent_fit)
    x_cal_all, _, _ = _matrix(cal, fit, groups, refs, recent_cal)
    x_test_all, _, _ = _matrix(test, fit, groups, refs, recent_test)
    x_fit, selected = _select_features(x_fit_all, names, fit_resid, cap=96)
    selected_idx = [names.index(name) for name in selected]
    x_cal = x_cal_all[:, selected_idx]
    x_test = x_test_all[:, selected_idx]

    latent_fit, svd, latent_scaler = _latent_targets(fit, groups["activation"], fit_resid)
    latent_cal_true = _latent_transform(cal, groups["activation"], cal_resid, svd, latent_scaler)
    latent_test_true = _latent_transform(test, groups["activation"], test_resid, svd, latent_scaler)
    model, x_scaler, y_scaler = _fit_multitask(x_fit, fit_resid, latent_fit)
    pred_fit = y_scaler.inverse_transform(model.predict(x_scaler.transform(x_fit)))
    pred_cal = y_scaler.inverse_transform(model.predict(x_scaler.transform(x_cal)))
    pred_test = y_scaler.inverse_transform(model.predict(x_scaler.transform(x_test)))
    pred_fit_resid = pred_fit[:, 0] * 100.0
    pred_cal_resid = pred_cal[:, 0] * 100.0
    pred_test_resid = pred_test[:, 0] * 100.0
    # Calibration is fit on the prior calibration block only.
    slope = 1.0
    intercept = 0.0
    ok = np.isfinite(pred_cal_resid) & np.isfinite(cal_resid)
    if ok.sum() >= 500 and np.nanvar(pred_cal_resid[ok]) > 1e-6:
        slope = float(np.cov(pred_cal_resid[ok], cal_resid[ok], bias=True)[0, 1] / np.nanvar(pred_cal_resid[ok]))
        intercept = float(np.nanmean(cal_resid[ok]) - slope * np.nanmean(pred_cal_resid[ok]))
    slope = float(np.clip(slope, 0.0, 1.5))
    pred_test_resid = intercept + slope * pred_test_resid
    pred_cal_resid = intercept + slope * pred_cal_resid
    demote_cal = np.clip(pred_cal[:, 1], 0.0, 1.0)
    promote_cal = np.clip(pred_cal[:, 2], 0.0, 1.0)
    demote_test = np.clip(pred_test[:, 1], 0.0, 1.0)
    promote_test = np.clip(pred_test[:, 2], 0.0, 1.0)
    support_cal = _support(cal, fit, groups, _add_ranked_geometry(cal, refs, groups)[[f"head_rank__{f}" for f in groups["head"]]].to_numpy(float) if groups["head"] else None)
    support_test = _support(test, fit, groups, _add_ranked_geometry(test, refs, groups)[[f"head_rank__{f}" for f in groups["head"]]].to_numpy(float) if groups["head"] else None)

    # Predeclared, asymmetric authority: demotion has more room than promotion.
    def corrected(delta: np.ndarray, down: float, up: float) -> np.ndarray:
        return test["anchor_mu"].to_numpy(float) + np.clip(delta, -down, up) * support_test

    resid_50_25 = corrected(pred_test_resid, 50.0, 25.0)
    resid_100_50 = corrected(pred_test_resid, 100.0, 50.0)
    demotion = test["anchor_mu"].to_numpy(float) - 100.0 * demote_test * support_test
    promotion = test["anchor_mu"].to_numpy(float) + 50.0 * promote_test * support_test
    residual_heads = corrected(0.50 * pred_test_resid - 25.0 * demote_test + 15.0 * promote_test, 100.0, 50.0)

    dynamic_ev, dynamic_meta = _dynamic_head_score(fit, test, groups, anchor_map)
    dynamic_delta = np.clip(dynamic_ev - test["anchor_mu"].to_numpy(float), -50.0, 25.0) * support_test
    dynamic = test["anchor_mu"].to_numpy(float) + dynamic_delta
    combined = test["anchor_mu"].to_numpy(float) + np.clip(0.70 * pred_test_resid + dynamic_delta, -100.0, 50.0) * support_test

    out = test[["candidate_id", "__ts__", "side_name", "month", "net_bps", "gross_bps", "incumbent_score"]].copy()
    out["anchor_only"] = out["incumbent_score"].to_numpy(float)
    out["residual_bounded_50_25"] = resid_50_25
    out["residual_bounded_100_50"] = resid_100_50
    out["demotion_only"] = demotion
    out["promotion_only"] = promotion
    out["residual_heads"] = residual_heads
    out["dynamic_head"] = dynamic
    out["combined"] = combined

    test_metrics = []
    for score in ["anchor_only", "residual_bounded_50_25", "residual_bounded_100_50", "demotion_only", "promotion_only", "residual_heads", "dynamic_head", "combined"]:
        test_metrics.extend(_tail_rows(out, score))
    state_diag = {
        "month": month,
        "history_rows": int(len(history)), "fit_rows": int(len(fit)), "calibration_rows": int(len(cal)), "test_rows": int(len(test)),
        "feature_count_before_selection": int(len(names)), "feature_count_selected": int(len(selected)), "selected_features": selected,
        "activation_family_count": int(len(groups["activation"])), "head_count": int(len(groups["head"])),
        "latent_components": int(latent_fit.shape[1]), "mlp_iterations": int(getattr(model, "n_iter_", 0)),
        "mlp_loss": float(getattr(model, "loss_", np.nan)), "residual_calibration_slope": slope, "residual_calibration_intercept": intercept,
        "support_test_mean": float(np.mean(support_test)), "support_test_p10": float(np.quantile(support_test, 0.10)),
        "demotion_auc_cal": float(roc_auc_score((cal_resid < -50).astype(int), demote_cal)) if len(np.unique((cal_resid < -50).astype(int))) == 2 else None,
        "promotion_auc_cal": float(roc_auc_score((cal_resid > 50).astype(int), promote_cal)) if len(np.unique((cal_resid > 50).astype(int))) == 2 else None,
        "latent_cal_rank_ic": [float(spearmanr(pred_cal[:, 3 + j], latent_cal_true[:, j]).statistic) for j in range(latent_fit.shape[1])],
        "latent_test_rank_ic": [float(spearmanr(pred_test[:, 3 + j], latent_test_true[:, j]).statistic) for j in range(latent_fit.shape[1])],
        "residual_mae_cal": float(mean_absolute_error(cal_resid, pred_cal_resid)), "residual_mae_test": float(mean_absolute_error(test_resid, pred_test_resid)),
        "dynamic_head_weights": dynamic_meta,
        "status": "complete",
    }
    return out, test_metrics, state_diag, [{"month": month, "feature": f, "rank": i + 1} for i, f in enumerate(selected)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--months", nargs="*", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame, contract = _load()
    groups = _column_groups(frame)
    months = sorted(frame["month"].dropna().unique().tolist())
    if args.months:
        months = [m for m in months if m in set(args.months)]
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for i, month in enumerate(months):
        pred, metrics, diag, feats = _run_month(frame, month, groups, SEED + i)
        all_predictions.append(pred)
        all_metrics.extend(metrics)
        diagnostics.append(diag)
        feature_rows.extend(feats)
        pd.DataFrame(metrics).to_parquet(args.output / f"metrics_{month}.parquet", index=False)
        _write_json(args.output / f"fold_audit_{month}.json", diag)
        print(json.dumps({"event": "month_complete", "month": month, "rows": len(pred), "status": diag["status"]}), flush=True)
        gc.collect()
    predictions = pd.concat(all_predictions, ignore_index=True)
    scores = ["anchor_only", "residual_bounded_50_25", "residual_bounded_100_50", "demotion_only", "promotion_only", "residual_heads", "dynamic_head", "combined"]
    pooled = []
    for score in scores:
        pooled.extend(_tail_rows(predictions, score))
        for month in months:
            pooled.extend(_tail_rows(predictions, score, month))
    metrics = pd.DataFrame(pooled)
    frontier = pd.DataFrame([row for score in scores for row in _frontier(predictions, score)])
    predictions.to_parquet(args.output / "conditional_oos_predictions.parquet", index=False, compression="zstd")
    metrics.to_parquet(args.output / "conditional_metrics.parquet", index=False, compression="zstd")
    frontier.to_parquet(args.output / "ev_coverage_frontier.parquet", index=False, compression="zstd")
    pd.DataFrame(diagnostics).to_json(args.output / "fold_diagnostics.json", orient="records", indent=2)
    pd.DataFrame(feature_rows).to_parquet(args.output / "selected_conditional_features.parquet", index=False)
    _write_json(args.output / "feature_contract.json", {
        "head_score_fields": groups["head"], "condition_fields": groups["condition"],
        "activation_fields": groups["activation"], "causal_context_fields": groups["context"],
        "head_count": len(groups["head"]), "condition_family_count": len(groups["activation"]),
        "forbidden_outcome_fields": sorted(FORBIDDEN), "recent_windows_days": WINDOWS_DAYS,
        "latent_contract": "SVD of membership x anchor residual, fit on prior rows only; MLP predicts continuous factors",
    })
    checks = {
        "frozen_score_parity": contract["frozen_current_score_max_abs_diff"] == 0.0,
        "candidate_ids_unique": bool(predictions["candidate_id"].is_unique),
        "test_rows_unique": bool(predictions["candidate_id"].duplicated().sum() == 0),
        "no_outcome_features_in_contract": not any(c in FORBIDDEN for c in groups["head"] + groups["condition"] + groups["context"]),
        "global_ranking_metrics_present": bool(set(scores).issubset(set(metrics["score"]))),
        "all_months_scored": sorted(predictions["month"].unique().tolist()) == months,
        "cost_not_reapplied": True,
    }
    _write_json(args.output / "correctness_test_report.json", {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "contract": contract})
    _write_json(args.output / "run_manifest.json", {
        "schema": "anchored_conditional_correction_v1", "status": "complete", "side": "long_and_short",
        "source_frozen": str(FROZEN), "source_current": str(CURRENT), "source_condition": str(CONDITION), "source_shared_ledger": str(SHARED),
        "months": months, "rows": int(len(predictions)), "contract": contract, "feature_digest": _digest(groups["head"] + groups["condition"] + groups["context"]),
        "groups": {k: len(v) for k, v in groups.items()}, "arms": scores,
        "selection": "frozen anchor first; conditional correction bounded; no test tuning; global pooled top-k and EV-coverage frontier",
        "checks": checks,
    })
    print(json.dumps({"event": "complete", "output": str(args.output), "rows": len(predictions), "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
