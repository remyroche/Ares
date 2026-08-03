#!/usr/bin/env python3
"""Stage-2 causal feature-information diagnosis for the root-cause roadmap.

This runner is deliberately diagnostic-only.  It joins the Stage-0 immutable
candidate ledger to the explicitly contracted *pre-entry* raw feature panel,
then reports information, transport, residual-probe, and drift evidence.  It
does not fit or alter a deployable base, residual, meta, auxiliary, or action
head; the small probe models are disposable diagnostic instruments.

The runner is intentionally conservative about provenance.  A feature may be
reported, but is eligible for causal probes only when it is in the frozen raw
feature contract, its row timestamp is no later than the Stage-0 cutoff, and
it has no direct target/future arithmetic signature.  The output retains
separate flags for source classes (OI/funding/order-book/model-derived) that
need a source-specific availability audit before any production reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "data_perp/artifacts"
DEFAULT_LEDGER = ART / "root_cause_diagnostic_substrate_20260731_v4/diagnostic_row_ledger.parquet"
DEFAULT_RAW_PANEL = ART / "long_exact_h12_raw_base_panel_20260730_v2/raw_base_panel.parquet"
DEFAULT_RAW_CONTRACT = ART / "long_exact_h12_raw_base_panel_20260730_v2/raw_feature_contract.json"
DEFAULT_OUTPUT = ART / "root_cause_feature_information_20260731_v4"
SCHEMA = "root_cause_feature_information_v2"

IDENTITY_COLUMNS = {
    "candidate_id", "symbol", "product", "side", "decision_ts", "feature_cutoff_ts",
    "executable_entry_ts", "label_end_ts", "label_available_ts", "policy_id",
    "cost_model_id", "path_source_id", "policy_archetype", "execution_geometry_key",
    "gross_h12_bps", "execution_adjusted_gross_h12_bps", "net_h12_bps", "fee_bps",
    "spread_bps", "slippage_bps", "total_cost_bps", "gross_h12_proxy_status",
}

DIRECT_TARGET_RE = re.compile(
    r"(?:^|_)(?:gross_h12|net_h12|execution_(?:gross|net|cost)|label|target|outcome|"
    r"future|mfe_12h|mae_12h|time_to.*mfe|exit_reason|action_value|continue_better)(?:_|$)",
    re.I,
)
TARGET_ARITHMETIC_RE = re.compile(
    r"(?:row_cost|realized_cost|realised_cost|known_row_cost|future_fill|action_exit|"
    r"exit_(?:price|fill|spread)|execution_cost|net_exit|net_continue|delta_continue)",
    re.I,
)
SOURCE_SENSITIVE_RE = re.compile(r"(?:^oi_|_oi_|open_interest|fund|orderbook|^ob_|liquidat|aggressor)", re.I)
UPSTREAM_RE = re.compile(r"(?:base.*(?:score|pred)|oof|prediction|leaf|residual_expected|mapped_ev)", re.I)


@dataclass(frozen=True)
class Fold:
    """A side-local chronological test window with purged resolved training."""

    side: str
    fold: str
    start: pd.Timestamp
    end: pd.Timestamp
    train_index: np.ndarray
    test_index: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, names: Iterable[str], context: str) -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")


def utc_column(frame: pd.DataFrame, name: str) -> pd.Series:
    values = pd.to_datetime(frame[name], utc=True, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} has invalid UTC timestamps")
    return values


def classify_mechanism(name: str) -> str:
    """Deterministic, non-economic grouping of an already admitted feature."""
    key = name.lower()
    if UPSTREAM_RE.search(key):
        return "upstream_model_predictions"
    if re.search(r"(?:^oi_|_oi_|open_interest|leverage|liquidat|unwind)", key):
        return "open_interest"
    if re.search(r"(?:fund|funding)", key):
        return "funding"
    if re.search(r"(?:orderbook|^ob_|spread|amihud|liquidity)", key):
        return "cost_liquidity"
    if re.search(r"(?:regime|transition|cluster|archetype|mahalanobis|reconstruction|changepoint)", key):
        return "regime_transition"
    # Principal-component and explicit breadth fields describe the joint
    # cross-section even when their suffix is a variance statistic.  They must
    # not be folded into the single-asset volatility bucket merely because of
    # the word "variance".
    if re.search(r"(?:market_pc|mkt_pc|cross_asset|pct_assets|breadth|dispersion|peer|universe|basket|symbol_minus)", key):
        return "cross_sectional_breadth"
    if re.search(r"(?:vol|(?:^|_)rv(?:_|$)|atr|range|drawdown|semivol|variance|chopp|squeeze)", key):
        return "volatility"
    if re.search(r"(?:volume|vwap|obv)", key):
        return "volume"
    if re.search(r"(?:cross_asset|market_|mkt_|pct_assets|breadth|dispersion|peer|universe|basket|symbol_minus)", key):
        return "cross_sectional_breadth"
    if re.search(r"(?:barrier|entry|side|mark_perp|mark_vs)", key):
        return "setup_geometry"
    return "price_trend"


def scan_target_proximity(name: str) -> dict[str, bool]:
    key = str(name).lower()
    direct = bool(DIRECT_TARGET_RE.search(key))
    arithmetic = bool(TARGET_ARITHMETIC_RE.search(key))
    return {
        "direct_target_name": direct,
        "target_arithmetic_overlap": arithmetic,
        "future_path_dependency_name": bool(re.search(r"(?:future|mfe_12h|mae_12h|exit_|timeout)", key)),
        "sensitive_source_name": bool(SOURCE_SENSITIVE_RE.search(key)),
        "upstream_model_dependency_name": bool(UPSTREAM_RE.search(key)),
        "hard_reject_name": bool(direct or arithmetic),
    }


def _metadata_for(name: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    item: Any = (metadata or {}).get(name, {})
    return dict(item) if isinstance(item, Mapping) else {}


def build_feature_inventory(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Create an explicit per-feature provenance and concentration ledger."""
    rows: list[dict[str, Any]] = []
    for name in feature_names:
        meta = _metadata_for(name, metadata)
        probe = scan_target_proximity(name)
        x = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = x.dropna()
        if len(finite):
            q01, q50, q99 = np.nanquantile(finite.to_numpy(float), [0.01, 0.50, 0.99])
            rounded = finite.round(8).value_counts(dropna=False)
            top_frequency = float(rounded.iloc[0] / len(finite)) if len(rounded) else np.nan
        else:
            q01 = q50 = q99 = top_frequency = np.nan
        contract_scoped = bool(meta.get("contract_scoped", True))
        declared_live = meta.get("live_reproducible")
        live_status = "VERIFIED" if declared_live is True else ("REJECTED" if declared_live is False else "NOT_VERIFIED")
        # An exact raw-panel/cutoff identity is adequate for a sealed research
        # probe.  It does not demonstrate that the live feature producer has
        # the same source, latency, or freshness at inference time.
        source_ts = str(meta.get("source_timestamp", "feature_cutoff_ts (sealed raw-panel contract)"))
        availability_ts = str(meta.get("availability_timestamp", "feature_cutoff_ts (sealed raw-panel contract)"))
        staleness = str(meta.get("staleness", "NOT_VERIFIED"))
        staleness_status = "VERIFIED" if "staleness" in meta else "NOT_VERIFIED"
        unresolved = bool(meta.get("unresolved_provenance", False))
        research_causal_eligible = bool(
            contract_scoped
            and not unresolved
            and not probe["hard_reject_name"]
        )
        production_live_eligible = bool(
            research_causal_eligible and live_status == "VERIFIED" and staleness_status == "VERIFIED"
        )
        rows.append({
            "feature_name": name,
            "mechanism_group": str(meta.get("mechanism_group", classify_mechanism(name))),
            "source_timestamp": source_ts,
            "availability_timestamp": availability_ts,
            "staleness": staleness,
            "staleness_status": staleness_status,
            "live_reproducible": declared_live if declared_live in (True, False) else np.nan,
            "live_reproducibility_status": live_status,
            "contract_scoped": contract_scoped,
            "unresolved_provenance": unresolved,
            "research_causal_probe_eligible": research_causal_eligible,
            # Backwards-compatible alias used by the diagnostic-only probes.
            "causal_probe_eligible": research_causal_eligible,
            "production_live_reuse_eligible": production_live_eligible,
            "research_causal_availability_status": "SEALED_RAW_PANEL_CUTOFF_VERIFIED" if research_causal_eligible else "REJECTED_OR_UNRESOLVED",
            "provenance_status": "REJECTED_TARGET_PROXIMITY" if probe["hard_reject_name"] else ("UNRESOLVED" if unresolved else "RAW_CONTRACT_CUTOFF_ONLY_LIVE_NOT_VERIFIED"),
            "missing_rate": float(x.isna().mean()),
            "finite_rows": int(len(finite)),
            "unique_values": int(finite.nunique()),
            "unique_fraction": float(finite.nunique() / max(len(finite), 1)),
            "largest_value_fraction": top_frequency,
            "p01": q01, "p50": q50, "p99": q99,
            **probe,
        })
    return pd.DataFrame(rows)


def make_chronological_folds(
    frame: pd.DataFrame,
    *,
    decision_col: str = "decision_ts",
    label_available_col: str = "label_available_ts",
    side_col: str = "side",
    min_train_rows: int = 2_500,
    max_folds: int | None = None,
) -> list[Fold]:
    """Monthly, side-local folds whose train labels resolved before test starts."""
    require_columns(frame, (decision_col, label_available_col, side_col), "diagnostic frame")
    decision = utc_column(frame, decision_col)
    available = utc_column(frame, label_available_col)
    if (available < decision).any():
        raise ValueError("label availability precedes the decision timestamp")
    result: list[Fold] = []
    months = pd.period_range(decision.min().to_period("M"), decision.max().to_period("M"), freq="M")
    for side in sorted(frame[side_col].dropna().astype(str).unique()):
        side_mask = frame[side_col].astype(str).eq(side).to_numpy()
        side_decision = decision.to_numpy()
        side_available = available.to_numpy()
        for month in months:
            start = pd.Timestamp(month.start_time, tz="UTC")
            end = start + pd.offsets.MonthBegin(1)
            test = np.flatnonzero(side_mask & (side_decision >= start) & (side_decision < end))
            # The availability test both purges overlapping H12 labels and
            # applies a naturally conservative embargo up to the test boundary.
            train = np.flatnonzero(side_mask & (side_available < start))
            if len(test) and len(train) >= min_train_rows:
                result.append(Fold(side=side, fold=str(month), start=start, end=end, train_index=train, test_index=test))
    if max_folds is not None and max_folds > 0:
        grouped: list[Fold] = []
        for side in sorted({f.side for f in result}):
            grouped.extend([f for f in result if f.side == side][-max_folds:])
        result = grouped
    if not result:
        raise ValueError("no chronological folds have the required resolved training support")
    return result


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def _univariate_metrics(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 20:
        return {"rows": int(len(values)), "spearman_ic": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "top_bottom_decile_spread_bps": np.nan, "top_decile_gross_bps": np.nan, "bottom_decile_gross_bps": np.nan}
    q = max(1, int(np.ceil(len(values) * 0.10)))
    ordered = values.sort_values("x", kind="stable")
    bottom = float(ordered.y.iloc[:q].mean())
    top = float(ordered.y.iloc[-q:].mean())
    binary = (values.y > 0.0).astype(int)
    if binary.nunique() == 2:
        auc = float(roc_auc_score(binary, values.x))
        pr = float(average_precision_score(binary, values.x))
    else:
        auc = pr = np.nan
    return {
        "rows": int(len(values)),
        "spearman_ic": _rank_ic(values.x.to_numpy(float), values.y.to_numpy(float)),
        "roc_auc": auc, "pr_auc": pr,
        "top_bottom_decile_spread_bps": top - bottom,
        "top_decile_gross_bps": top, "bottom_decile_gross_bps": bottom,
    }


def run_univariate_tests(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Side-local transported tests.  No pooled-period feature selection occurs."""
    require_columns(frame, (target_col,), "diagnostic frame")
    rows: list[dict[str, Any]] = []
    feature_names = inventory.feature_name.tolist()
    for fold in folds:
        test = frame.iloc[fold.test_index]
        target = test[target_col]
        for name in feature_names:
            metrics = _univariate_metrics(test[name], target)
            rows.append({"feature_name": name, "side": fold.side, "fold": fold.fold, "fold_start": fold.start, "fold_end": fold.end, "train_rows": len(fold.train_index), **metrics})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby(["feature_name", "side"], as_index=False).agg(
        folds=("fold", "nunique"),
        transported_ic_mean=("spearman_ic", "mean"), transported_ic_median=("spearman_ic", "median"),
        transported_ic_std=("spearman_ic", "std"),
        positive_ic_fraction=("spearman_ic", lambda x: float(np.mean(np.asarray(x, float) > 0.0))),
        auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"),
        top_bottom_decile_spread_mean_bps=("top_bottom_decile_spread_bps", "mean"),
        top_decile_gross_mean_bps=("top_decile_gross_bps", "mean"),
        bottom_decile_gross_mean_bps=("bottom_decile_gross_bps", "mean"),
        evaluated_rows=("rows", "sum"),
    )
    return detail, summary


def run_directional_alpha_diagnostics(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
) -> pd.DataFrame:
    """Report frozen OOF alpha information without relabelling it as EV.

    This namespace is deliberately separate from the economic-residual probe.
    The native soft-alpha target measures directional opportunity; it is not an
    execution-value target and must not be used as the residual target for the
    base/residual EV audit.
    """
    target_col = "__reconstructed_soft_alpha_12h__"
    score_specs = (
        ("directional_alpha_base", "score_base_alpha"),
        ("directional_alpha_residual_stack", "score_residual_alpha"),
    )
    rows: list[dict[str, Any]] = []
    fully_oof = "residual_is_oof" in frame.columns and frame.residual_is_oof.fillna(False).astype(bool).all()
    for namespace, score_col in score_specs:
        if target_col not in frame.columns or score_col not in frame.columns:
            rows.append({
                "namespace": namespace, "score_column": score_col,
                "target_column": target_col, "status": "NOT_RUN_MISSING_FROZEN_OOF_ALPHA_SCORE_OR_TARGET",
            })
            continue
        if not fully_oof:
            rows.append({
                "namespace": namespace, "score_column": score_col,
                "target_column": target_col, "status": "NOT_RUN_SCORE_NOT_PROVEN_FULLY_OOF",
            })
            continue
        for fold in folds:
            test = frame.iloc[fold.test_index]
            pair = test.loc[:, [score_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 20:
                rows.append({
                    "namespace": namespace, "score_column": score_col, "target_column": target_col,
                    "side": fold.side, "fold": fold.fold, "status": "NOT_RUN_INSUFFICIENT_SUPPORT", "rows": len(pair),
                })
                continue
            y = pair[target_col].to_numpy(float)
            pred = pair[score_col].to_numpy(float)
            binary = (y > 0.5).astype(int)
            auc = pr = brier = np.nan
            if np.unique(binary).size == 2:
                clipped = np.clip(pred, 1e-6, 1.0 - 1e-6)
                auc = float(roc_auc_score(binary, pred))
                pr = float(average_precision_score(binary, pred))
                brier = float(brier_score_loss(binary, clipped))
            rows.append({
                "namespace": namespace, "score_column": score_col, "target_column": target_col,
                "side": fold.side, "fold": fold.fold, "status": "OK", "rows": len(pair),
                "spearman_ic": _rank_ic(pred, y), "roc_auc": auc, "pr_auc": pr, "brier": brier,
                "alpha_residual_mae": float(np.mean(np.abs(y - pred))),
                "alpha_residual_bias": float(np.mean(y - pred)),
            })
    return pd.DataFrame(rows)


def materialize_fold_local_gross_maps(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create side-local chronological OOS maps from frozen alpha to gross H12.

    Each map is fit solely on rows whose labels were available before its test
    month.  It is intentionally a one-dimensional, deterministic calibration
    layer, not a new alpha/residual model.  The returned test prediction is in
    bps and may be used as the canonical economic residual baseline.
    """
    specs = (
        ("canonical_gross_base", "score_base_alpha"),
        ("canonical_gross_residual_stack", "score_residual_alpha"),
    )
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    fully_oof = "residual_is_oof" in frame.columns and frame.residual_is_oof.fillna(False).astype(bool).all()
    for namespace, score_col in specs:
        if score_col not in frame.columns:
            rows.append({"head": namespace, "score_column": score_col, "status": "NOT_RUN_MISSING_FROZEN_OOF_ALPHA_SCORE"})
            continue
        if not fully_oof:
            rows.append({"head": namespace, "score_column": score_col, "status": "NOT_RUN_SCORE_NOT_PROVEN_FULLY_OOF"})
            continue
        for fold in folds:
            train, test = frame.iloc[fold.train_index], frame.iloc[fold.test_index]
            train_pair = train.loc[:, [score_col, "gross_h12_bps"]].apply(pd.to_numeric, errors="coerce").dropna()
            test_pair = test.loc[:, [score_col, "gross_h12_bps"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(train_pair) < 100 or len(test_pair) < 20 or train_pair[score_col].nunique() < 2:
                rows.append({
                    "head": namespace, "score_column": score_col, "side": fold.side, "fold": fold.fold,
                    "status": "NOT_RUN_INSUFFICIENT_RESOLVED_MAPPING_SUPPORT", "train_rows": len(train_pair), "test_rows": len(test_pair),
                })
                continue
            mapper = IsotonicRegression(increasing=True, out_of_bounds="clip")
            mapper.fit(train_pair[score_col].to_numpy(float), train_pair.gross_h12_bps.to_numpy(float))
            prediction = mapper.predict(test_pair[score_col].to_numpy(float))
            residual = test_pair.gross_h12_bps.to_numpy(float) - prediction
            rows.append({
                "head": namespace, "score_column": score_col, "side": fold.side, "fold": fold.fold,
                "status": "OK", "train_rows": len(train_pair), "test_rows": len(test_pair),
                "train_label_available_before_test_start": True,
                "mapping_target": "gross_h12_bps", "prediction_unit": "bps",
                "mapping_oos_mae_bps": float(np.mean(np.abs(residual))),
                "mapping_oos_ic": _rank_ic(prediction, test_pair.gross_h12_bps.to_numpy(float)),
            })
            predictions.append(pd.DataFrame({
                "candidate_id": test.loc[test_pair.index, "candidate_id"].astype(str).to_numpy(),
                "head": namespace, "side": fold.side, "fold": fold.fold,
                "score_column": score_col, "gross_h12_bps": test_pair.gross_h12_bps.to_numpy(float),
                "gross_mapped_prediction_bps": prediction, "gross_mapping_residual_bps": residual,
            }))
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=[
        "candidate_id", "head", "side", "fold", "score_column", "gross_h12_bps", "gross_mapped_prediction_bps", "gross_mapping_residual_bps",
    ])
    if not pred.empty and pred.duplicated(["candidate_id", "head"]).any():
        raise ValueError("fold-local gross map has duplicate candidate/head OOS predictions")
    return pd.DataFrame(rows), pred


def run_current_netmap_diagnostics(frame: pd.DataFrame, folds: Sequence[Fold]) -> pd.DataFrame:
    """Describe the inherited OOF net maps without treating them as gross maps."""
    specs = (
        ("current_netmap_base", "current_netmap_base_prediction"),
        ("current_netmap_residual_stack", "current_netmap_residual_stack_prediction"),
    )
    rows: list[dict[str, Any]] = []
    for head, prediction_col in specs:
        if prediction_col not in frame.columns:
            rows.append({"head": head, "prediction_column": prediction_col, "target_column": "net_h12_bps", "status": "NOT_RUN_MISSING_FROZEN_OOF_NET_MAP"})
            continue
        for fold in folds:
            test = frame.iloc[fold.test_index]
            pair = test.loc[:, [prediction_col, "net_h12_bps"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 20:
                rows.append({"head": head, "prediction_column": prediction_col, "target_column": "net_h12_bps", "side": fold.side, "fold": fold.fold, "status": "NOT_RUN_INSUFFICIENT_SUPPORT", "rows": len(pair)})
                continue
            residual = pair.net_h12_bps.to_numpy(float) - pair[prediction_col].to_numpy(float)
            rows.append({
                "head": head, "prediction_column": prediction_col, "target_column": "net_h12_bps",
                "side": fold.side, "fold": fold.fold, "status": "OK", "rows": len(pair),
                "residual_definition": f"net_h12_bps - {prediction_col}",
                "netmap_oof_mae_bps": float(np.mean(np.abs(residual))),
                "netmap_oof_residual_bias_bps": float(np.mean(residual)),
                "netmap_oof_ic": _rank_ic(pair[prediction_col].to_numpy(float), pair.net_h12_bps.to_numpy(float)),
            })
    return pd.DataFrame(rows)


def _feature_matrix(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    chosen = [name for name in features if name in train.columns]
    if not chosen:
        return np.empty((len(train), 0)), np.empty((len(test), 0)), []
    tr = train.loc[:, chosen].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    te = test.loc[:, chosen].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = tr.notna().any(axis=0)
    chosen = [name for name in chosen if bool(keep.get(name, False))]
    tr, te = tr.loc[:, chosen], te.loc[:, chosen]
    medians = tr.median(axis=0).fillna(0.0)
    return tr.fillna(medians).to_numpy(np.float32), te.fillna(medians).to_numpy(np.float32), chosen


def _probe_model(seed: int) -> Pipeline:
    """Fixed low-capacity probe, intentionally not a model-selection arm."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("regression", Ridge(alpha=50.0)),
    ])


def _cap_probe_training(x: np.ndarray, y: np.ndarray, *, limit: int = 25_000) -> tuple[np.ndarray, np.ndarray, bool]:
    """Keep probe runtime bounded using a deterministic time-spread subset."""
    if len(x) <= limit:
        return x, y, False
    indices = np.linspace(0, len(x) - 1, limit, dtype=np.int64)
    return x[indices], y[indices], True


def run_mechanism_group_oof(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    target_col: str,
    seed: int = 71,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed, side-local chronological OOF probes on identical outer rows.

    The group set and model are predeclared; there is no test-period feature or
    HPO choice.  This is therefore an incremental information diagnostic, not
    a model-selection exercise.
    """
    admitted = inventory.loc[inventory.causal_probe_eligible, ["feature_name", "mechanism_group"]]
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for group in sorted(admitted.mechanism_group.unique()):
        features = admitted.loc[admitted.mechanism_group.eq(group), "feature_name"].tolist()
        for fold_i, fold in enumerate(folds):
            train, test = frame.iloc[fold.train_index], frame.iloc[fold.test_index]
            usable_train = train[target_col].notna()
            usable_test = test[target_col].notna()
            x_train, x_test, selected = _feature_matrix(train.loc[usable_train], test.loc[usable_test], features)
            if len(selected) == 0 or len(x_train) < 200 or len(x_test) < 20:
                rows.append({"mechanism_group": group, "side": fold.side, "fold": fold.fold, "status": "NOT_RUN_INSUFFICIENT_CAUSAL_FEATURE_OR_SUPPORT", "features": len(selected), "train_rows": int(len(x_train)), "test_rows": int(len(x_test))})
                continue
            model = _probe_model(seed + fold_i)
            y_train = train.loc[usable_train, target_col].to_numpy(float)
            y_test = test.loc[usable_test, target_col].to_numpy(float)
            x_fit, y_fit, capped = _cap_probe_training(x_train, y_train)
            model.fit(x_fit, y_fit)
            pred = model.predict(x_test)
            economics = _univariate_metrics(pd.Series(pred), pd.Series(y_test))
            rows.append({"mechanism_group": group, "side": fold.side, "fold": fold.fold, "status": "OK", "features": len(selected), "feature_names_json": json.dumps(selected), "train_rows": int(len(x_train)), "fit_rows": int(len(x_fit)), "train_rows_capped": capped, "test_rows": int(len(x_test)), "train_mae_bps": float(np.mean(np.abs(y_fit - model.predict(x_fit)))), "oof_mae_bps": float(np.mean(np.abs(y_test - pred))), **economics})
            predictions.append(pd.DataFrame({"candidate_id": test.loc[usable_test, "candidate_id"].astype(str).to_numpy(), "side": fold.side, "fold": fold.fold, "mechanism_group": group, "target_gross_h12_bps": y_test, "prediction": pred}))
    detail = pd.DataFrame(rows)
    pred_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=["candidate_id", "side", "fold", "mechanism_group", "target_gross_h12_bps", "prediction"])
    return detail, pred_frame


def _read_prediction_source(path: Path | None, *, prediction_col: str | None, name: str) -> pd.DataFrame | None:
    if path is None:
        return None
    data = pd.read_parquet(path)
    require_columns(data, ("candidate_id",), f"{name} prediction source")
    column = prediction_col or next((x for x in ("prediction", f"{name}_prediction", "base_prediction", "residual_prediction") if x in data.columns), None)
    if column is None:
        raise ValueError(f"{name} prediction source has no declared prediction column")
    out = data.loc[:, ["candidate_id", column]].rename(columns={column: f"{name}_prediction"}).copy()
    out["candidate_id"] = out.candidate_id.astype(str)
    if out.candidate_id.duplicated().any():
        raise ValueError(f"{name} prediction source candidate IDs are not unique")
    return out


def run_residual_probes(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    target_by_head: Mapping[str, str],
    seed: int = 701,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probe *economic* residuals of supplied OOF/frozen head predictions.

    ``target_by_head`` uses the output namespace as key and requires a
    corresponding ``<key>_prediction`` column in the same unit as the target.
    Each probe target is exactly ``realised gross H12 - frozen prediction``.
    Missing OOF inputs produce explicit ``NOT_RUN`` rows; this runner never
    substitutes a directional-alpha score or an in-sample prediction.
    """
    causal = inventory.loc[inventory.causal_probe_eligible]
    all_features = causal.feature_name.tolist()
    future_features = inventory.loc[inventory.future_path_dependency_name, "feature_name"].tolist()
    rows: list[dict[str, Any]] = []
    predicted: list[pd.DataFrame] = []
    for head, target_col in target_by_head.items():
        prediction_col = f"{head}_prediction"
        if prediction_col not in frame.columns:
            rows.append({"head": head, "probe_family": "all", "status": "NOT_RUN_MISSING_OOF_OR_FROZEN_PREDICTION", "target_column": target_col, "residual_definition": f"{target_col} - {prediction_col}"})
            continue
        if target_col not in frame.columns:
            rows.append({"head": head, "probe_family": "all", "status": "NOT_RUN_MISSING_TARGET", "target_column": target_col, "residual_definition": f"{target_col} - {prediction_col}"})
            continue
        valid = frame[[prediction_col, target_col]].notna().all(axis=1)
        if int(valid.sum()) == 0:
            rows.append({"head": head, "probe_family": "all", "status": "NOT_RUN_NO_PREDICTION_TARGET_OVERLAP", "target_column": target_col, "residual_definition": f"{target_col} - {prediction_col}"})
            continue
        residual = pd.to_numeric(frame[target_col], errors="coerce") - pd.to_numeric(frame[prediction_col], errors="coerce")
        probe_sets: dict[str, list[str]] = {"same_causal_features": all_features, "future_oracle_features": future_features}
        for group in sorted(causal.mechanism_group.unique()):
            group_features = causal.loc[causal.mechanism_group.eq(group), "feature_name"].tolist()
            probe_sets[f"mechanism_only__{group}"] = group_features
            probe_sets[f"excluded_causal_group__{group}"] = [name for name in all_features if name not in set(group_features)]
        for family, features in probe_sets.items():
            for fold_i, fold in enumerate(folds):
                train = frame.iloc[fold.train_index]
                test = frame.iloc[fold.test_index]
                train_valid = train[[prediction_col, target_col]].notna().all(axis=1)
                test_valid = test[[prediction_col, target_col]].notna().all(axis=1)
                x_train, x_test, selected = _feature_matrix(train.loc[train_valid], test.loc[test_valid], features)
                if len(selected) == 0 or len(x_train) < 200 or len(x_test) < 20:
                    rows.append({"head": head, "probe_family": family, "side": fold.side, "fold": fold.fold, "status": "NOT_RUN_INSUFFICIENT_FEATURE_OR_SUPPORT", "features": len(selected), "train_rows": len(x_train), "test_rows": len(x_test), "target_column": target_col, "residual_definition": f"{target_col} - {prediction_col}"})
                    continue
                y_train = residual.iloc[fold.train_index][train_valid].to_numpy(float)
                y_test = residual.iloc[fold.test_index][test_valid].to_numpy(float)
                model = _probe_model(seed + 31 * fold_i + len(family))
                x_fit, y_fit, capped = _cap_probe_training(x_train, y_train)
                model.fit(x_fit, y_fit)
                pred = model.predict(x_test)
                ic = _rank_ic(pred, y_test)
                rows.append({"head": head, "probe_family": family, "side": fold.side, "fold": fold.fold, "status": "OK", "features": len(selected), "feature_names_json": json.dumps(selected), "train_rows": len(x_train), "fit_rows": len(x_fit), "train_rows_capped": capped, "test_rows": len(x_test), "residual_probe_oof_mae_bps": float(np.mean(np.abs(y_test - pred))), "residual_probe_oof_ic": ic, "residual_std_bps": float(np.std(y_test)), "target_column": target_col, "residual_definition": f"{target_col} - {prediction_col}"})
                predicted.append(pd.DataFrame({"candidate_id": test.loc[test_valid, "candidate_id"].astype(str).to_numpy(), "head": head, "probe_family": family, "side": fold.side, "fold": fold.fold, "residual": y_test, "residual_probe_prediction": pred}))
    return pd.DataFrame(rows), (pd.concat(predicted, ignore_index=True) if predicted else pd.DataFrame(columns=["candidate_id", "head", "probe_family", "side", "fold", "residual", "residual_probe_prediction"]))


def _psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    ref = reference[np.isfinite(reference)]
    cmp = comparison[np.isfinite(comparison)]
    if len(ref) < 20 or len(cmp) < 20:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, bins=edges)[0].astype(float) / len(ref)
    b = np.histogram(cmp, bins=edges)[0].astype(float) / len(cmp)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def _js(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    ref = reference[np.isfinite(reference)]
    cmp = comparison[np.isfinite(comparison)]
    if len(ref) < 20 or len(cmp) < 20:
        return np.nan
    lo, hi = min(float(np.min(ref)), float(np.min(cmp))), max(float(np.max(ref)), float(np.max(cmp)))
    if not np.isfinite(lo + hi) or np.isclose(lo, hi):
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(ref, bins=edges)[0].astype(float); q = np.histogram(cmp, bins=edges)[0].astype(float)
    p, q = p / max(p.sum(), 1.0), q / max(q.sum(), 1.0); m = 0.5 * (p + q)
    mask_p, mask_q = p > 0, q > 0
    return float(0.5 * np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p])) + 0.5 * np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q])))


def _adversarial_auc(reference: pd.DataFrame, comparison: pd.DataFrame, features: Sequence[str], seed: int) -> float:
    """Fixed linear adversarial validation, kept deliberately cheap and diagnostic.

    The accompanying PSI/JS/Wasserstein tables capture univariate non-linear
    shifts.  This model measures multivariate separability without turning the
    drift audit into a repeated HPO/model-training job.
    """
    if len(reference) < 100 or len(comparison) < 100:
        return np.nan
    selected = list(features)[:64]
    x_ref, x_cmp, selected = _feature_matrix(reference, comparison, selected)
    if len(selected) == 0:
        return np.nan
    take_ref, take_cmp = min(len(x_ref), 3_000), min(len(x_cmp), 3_000)
    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(len(x_ref), take_ref, replace=False); cmp_idx = rng.choice(len(x_cmp), take_cmp, replace=False)
    x = np.vstack([x_ref[ref_idx], x_cmp[cmp_idx]]); y = np.r_[np.zeros(take_ref), np.ones(take_cmp)]
    order = rng.permutation(len(y)); cut = int(len(y) * 0.70)
    if cut < 50 or len(y) - cut < 50:
        return np.nan
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", RidgeClassifier(alpha=10.0)),
    ])
    model.fit(x[order[:cut]], y[order[:cut]])
    return float(roc_auc_score(y[order[cut:]], model.decision_function(x[order[cut:]])))


def run_drift_diagnostics(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    prediction_columns: Sequence[str] = (),
    target_col: str = "gross_h12_bps",
    target_by_prediction: Mapping[str, str] | None = None,
    max_features: int = 96,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Distribution, adversarial, prediction, and calibration transport drift."""
    causal = inventory.loc[inventory.causal_probe_eligible].sort_values(["missing_rate", "feature_name"]).feature_name.tolist()[:max_features]
    work = frame.copy()
    work["__month__"] = utc_column(work, "decision_ts").dt.to_period("M").astype(str)
    detailed: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for side in sorted(work.side.astype(str).unique()):
        side_frame = work.loc[work.side.astype(str).eq(side)].copy()
        months = sorted(side_frame.__month__.unique())
        prior: pd.DataFrame | None = None
        for ordinal, month in enumerate(months):
            current = side_frame.loc[side_frame.__month__.eq(month)]
            if prior is None or prior.empty:
                prior = current.copy()
                continue
            for name in causal:
                ref = pd.to_numeric(prior[name], errors="coerce").to_numpy(float)
                cmp = pd.to_numeric(current[name], errors="coerce").to_numpy(float)
                detailed.append({"scope_type": "side_month", "side": side, "scope_value": month, "feature_name": name, "reference_rows": len(prior), "comparison_rows": len(current), "psi": _psi(ref, cmp), "jensen_shannon": _js(ref, cmp), "wasserstein": float(wasserstein_distance(ref[np.isfinite(ref)], cmp[np.isfinite(cmp)])) if np.isfinite(ref).any() and np.isfinite(cmp).any() else np.nan, "missingness_delta": float(np.mean(~np.isfinite(cmp)) - np.mean(~np.isfinite(ref)))})
            auc = _adversarial_auc(prior, current, causal, seed=901 + ordinal)
            cohort_rows.append({"scope_type": "side_month", "side": side, "scope_value": month, "reference": "all_prior_months", "rows": len(current), "adversarial_auc": auc})
            for prediction_col in prediction_columns:
                if prediction_col not in current.columns:
                    continue
                calibration_target = (target_by_prediction or {}).get(prediction_col, target_col)
                if calibration_target not in current.columns:
                    continue
                pair = current[[prediction_col, calibration_target]].apply(pd.to_numeric, errors="coerce").dropna()
                slope = intercept = np.nan
                if len(pair) >= 30 and pair[prediction_col].nunique() > 1:
                    fit = LinearRegression().fit(pair[[prediction_col]], pair[calibration_target])
                    slope, intercept = float(fit.coef_[0]), float(fit.intercept_)
                prediction_rows.append({"scope_type": "side_month", "side": side, "scope_value": month, "prediction_column": prediction_col, "target_column": calibration_target, "rows": len(pair), "prediction_mean": float(pd.to_numeric(current[prediction_col], errors="coerce").mean()), "prediction_std": float(pd.to_numeric(current[prediction_col], errors="coerce").std()), "calibration_slope": slope, "calibration_intercept": intercept})
            prior = pd.concat([prior, current], ignore_index=True)
    # Cohort-level summaries use the broad, previously observed side distribution
    # as reference and therefore identify concentration without inventing a
    # separate selection/ranking policy.
    for group_col in ("symbol", "policy_archetype"):
        if group_col not in work.columns:
            continue
        for side in sorted(work.side.astype(str).unique()):
            local = work.loc[work.side.astype(str).eq(side)]
            for group, current in local.groupby(group_col, dropna=True, observed=True):
                if len(current) < 100:
                    continue
                reference = local.loc[~local.index.isin(current.index)]
                if len(reference) < 100:
                    continue
                values = []
                for name in causal[:32]:
                    values.append(_psi(pd.to_numeric(reference[name], errors="coerce").to_numpy(float), pd.to_numeric(current[name], errors="coerce").to_numpy(float)))
                cohort_rows.append({"scope_type": "symbol_cohort" if group_col == "symbol" else "regime", "side": side, "scope_value": str(group), "reference": f"other_{group_col}_same_side", "rows": len(current), "adversarial_auc": np.nan, "median_feature_psi": float(np.nanmedian(values)) if values else np.nan})
    return pd.DataFrame(detailed), pd.DataFrame(cohort_rows), pd.DataFrame(prediction_rows)


def _load_metadata(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("feature metadata must be a JSON mapping keyed by feature name")
    return payload


def _load_frame(ledger_path: Path, raw_panel_path: Path, raw_contract_path: Path, *, max_features: int | None) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    ledger = pd.read_parquet(ledger_path)
    require_columns(ledger, ("candidate_id", "side", "decision_ts", "feature_cutoff_ts", "label_available_ts", "gross_h12_bps"), "Stage-0 diagnostic ledger")
    ledger = ledger.copy(); ledger["candidate_id"] = ledger.candidate_id.astype(str)
    if ledger.candidate_id.duplicated().any():
        raise ValueError("Stage-0 diagnostic ledger candidate IDs are not unique")
    ledger["decision_ts"] = utc_column(ledger, "decision_ts")
    ledger["feature_cutoff_ts"] = utc_column(ledger, "feature_cutoff_ts")
    ledger["label_available_ts"] = utc_column(ledger, "label_available_ts")
    if not ledger.feature_cutoff_ts.le(ledger.decision_ts).all():
        raise ValueError("Stage-0 ledger feature cutoff follows a decision")
    contract = json.loads(raw_contract_path.read_text(encoding="utf-8"))
    features = [str(x) for x in contract["raw_feature_columns"]]
    if max_features is not None:
        features = features[:max_features]
    raw_columns = ["candidate_id", "__ts__", "__symbol__", "__reconstructed_soft_alpha_12h__", *features]
    schema = set(pd.read_parquet(raw_panel_path, columns=None).columns) if False else None
    # The contract is authoritative, but old artifacts may not have every
    # documented optional feature; read the intersecting schema only.
    import pyarrow.parquet as pq
    available = set(pq.read_schema(raw_panel_path).names)
    raw_columns = [name for name in raw_columns if name in available]
    raw = pd.read_parquet(raw_panel_path, columns=raw_columns)
    require_columns(raw, ("candidate_id",), "raw feature panel")
    raw["candidate_id"] = raw.candidate_id.astype(str)
    if raw.candidate_id.duplicated().any():
        raise ValueError("raw feature panel candidate IDs are not unique")
    actual_features = [name for name in features if name in raw.columns]
    keep = [name for name in ("candidate_id", "__ts__", "__symbol__", "__reconstructed_soft_alpha_12h__", *actual_features) if name in raw.columns]
    joined = ledger.merge(raw.loc[:, keep], on="candidate_id", how="left", validate="one_to_one", suffixes=("", "_raw"))
    if joined[actual_features].isna().all(axis=1).any():
        raise ValueError("raw feature coverage is incomplete for Stage-0 candidates")
    if "__ts__" in joined.columns:
        raw_ts = pd.to_datetime(joined["__ts__"], utc=True, errors="coerce")
        if raw_ts.notna().any() and not raw_ts.fillna(joined.feature_cutoff_ts).eq(joined.feature_cutoff_ts).all():
            raise ValueError("raw feature timestamp does not match Stage-0 feature cutoff")
    if "symbol" not in joined.columns and "__symbol__" in joined.columns:
        joined["symbol"] = joined["__symbol__"].astype(str)
    return joined, actual_features, {"ledger_sha256": sha256(ledger_path), "raw_panel_sha256": sha256(raw_panel_path), "raw_contract_sha256": sha256(raw_contract_path)}


def run(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    raw_panel_path: Path = DEFAULT_RAW_PANEL,
    raw_contract_path: Path = DEFAULT_RAW_CONTRACT,
    output: Path = DEFAULT_OUTPUT,
    feature_metadata_path: Path | None = None,
    base_predictions_path: Path | None = None,
    residual_predictions_path: Path | None = None,
    base_prediction_col: str | None = None,
    residual_prediction_col: str | None = None,
    max_features: int | None = None,
    max_folds: int | None = None,
    min_train_rows: int = 2_500,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    for path in (ledger_path, raw_panel_path, raw_contract_path):
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = _load_metadata(feature_metadata_path)
    frame, features, input_hashes = _load_frame(ledger_path, raw_panel_path, raw_contract_path, max_features=max_features)
    base = _read_prediction_source(base_predictions_path, prediction_col=base_prediction_col, name="base")
    residual = _read_prediction_source(residual_predictions_path, prediction_col=residual_prediction_col, name="residual")
    for extra in (base, residual):
        if extra is not None:
            frame = frame.merge(extra, on="candidate_id", how="left", validate="one_to_one")
    if "__reconstructed_soft_alpha_12h__" in frame.columns:
        frame["base_target"] = pd.to_numeric(frame["__reconstructed_soft_alpha_12h__"], errors="coerce")
    inventory = build_feature_inventory(frame, features, metadata=metadata)
    folds = make_chronological_folds(frame, min_train_rows=min_train_rows, max_folds=max_folds)
    univariate, univariate_summary = run_univariate_tests(frame, inventory, folds, target_col="gross_h12_bps")
    mechanisms, mechanism_predictions = run_mechanism_group_oof(frame, inventory, folds, target_col="gross_h12_bps")
    directional_alpha = run_directional_alpha_diagnostics(frame, folds)
    gross_mapping, gross_mapping_predictions = materialize_fold_local_gross_maps(frame, folds)
    if not gross_mapping_predictions.empty:
        mapped = gross_mapping_predictions.pivot(index="candidate_id", columns="head", values="gross_mapped_prediction_bps").reset_index()
        mapped.columns.name = None
        mapped = mapped.rename(columns={
            "canonical_gross_base": "canonical_gross_base_prediction",
            "canonical_gross_residual_stack": "canonical_gross_residual_stack_prediction",
        })
        frame = frame.merge(mapped, on="candidate_id", how="left", validate="one_to_one")

    # Directional alpha and economic residuals have different target spaces.
    # The former remains a labelled diagnostic above.  The canonical economic
    # residual uses the strict fold-local gross map just materialised.  The
    # historical expected-EV maps remain a distinct *net-map* diagnostic.
    economic_oof = False
    if "residual_is_oof" in frame.columns:
        oof = frame.residual_is_oof.fillna(False).astype(bool)
        if not oof.all():
            raise ValueError("Stage-0 expected-value score source is not fully OOF")
        economic_oof = {"score_base_expected_ev", "score_residual_expected_ev"}.issubset(frame.columns)
    if economic_oof:
        # These maps were trained on net EV by the historic stack.  They are
        # never represented as gross maps in the output.
        frame["current_netmap_base_prediction"] = pd.to_numeric(frame.score_base_expected_ev, errors="coerce") * 10_000.0
        frame["current_netmap_residual_stack_prediction"] = pd.to_numeric(frame.score_residual_expected_ev, errors="coerce") * 10_000.0
    else:
        # Optional external files are allowed only as explicitly supplied,
        # already-OOF *gross* predictions in bps.
        if "base_prediction" in frame.columns:
            frame["canonical_gross_base_prediction"] = pd.to_numeric(frame.base_prediction, errors="coerce")
        if "residual_prediction" in frame.columns:
            frame["canonical_gross_residual_stack_prediction"] = pd.to_numeric(frame.residual_prediction, errors="coerce")
    current_netmap_diagnostics = run_current_netmap_diagnostics(frame, folds)
    target_by_head: dict[str, str] = {}
    if "canonical_gross_base_prediction" in frame.columns:
        target_by_head["canonical_gross_base"] = "gross_h12_bps"
    if "canonical_gross_residual_stack_prediction" in frame.columns:
        target_by_head["canonical_gross_residual_stack"] = "gross_h12_bps"
    if not target_by_head:
        residual_probes = pd.DataFrame([
            {"head": "canonical_gross_base", "probe_family": "all", "status": "NOT_RUN_MISSING_FROZEN_OOF_ECONOMIC_PREDICTION", "target_column": "gross_h12_bps", "residual_definition": "gross_h12_bps - canonical_gross_base_prediction"},
            {"head": "canonical_gross_residual_stack", "probe_family": "all", "status": "NOT_RUN_MISSING_FROZEN_OOF_ECONOMIC_PREDICTION", "target_column": "gross_h12_bps", "residual_definition": "gross_h12_bps - canonical_gross_residual_stack_prediction"},
        ])
        residual_predictions = pd.DataFrame(columns=["candidate_id", "head", "probe_family", "side", "fold", "residual", "residual_probe_prediction"])
    else:
        residual_probes, residual_predictions = run_residual_probes(frame, inventory, folds, target_by_head=target_by_head)
    prediction_columns = [name for name in (
        "canonical_gross_base_prediction", "canonical_gross_residual_stack_prediction",
        "current_netmap_base_prediction", "current_netmap_residual_stack_prediction",
    ) if name in frame.columns]
    drift, cohort_drift, prediction_drift = run_drift_diagnostics(
        frame, inventory, prediction_columns=prediction_columns,
        target_by_prediction={
            "canonical_gross_base_prediction": "gross_h12_bps",
            "canonical_gross_residual_stack_prediction": "gross_h12_bps",
            "current_netmap_base_prediction": "net_h12_bps",
            "current_netmap_residual_stack_prediction": "net_h12_bps",
        },
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        inventory.to_parquet(stage / "feature_information_inventory.parquet", index=False, compression="zstd")
        univariate.to_parquet(stage / "feature_information_univariate.parquet", index=False, compression="zstd")
        univariate_summary.to_parquet(stage / "feature_information_results.parquet", index=False, compression="zstd")
        mechanisms.to_parquet(stage / "feature_information_mechanism_oof.parquet", index=False, compression="zstd")
        mechanism_predictions.to_parquet(stage / "feature_information_mechanism_predictions.parquet", index=False, compression="zstd")
        directional_alpha.to_parquet(stage / "feature_information_directional_alpha_diagnostics.parquet", index=False, compression="zstd")
        gross_mapping.to_parquet(stage / "feature_information_fold_local_gross_mapping.parquet", index=False, compression="zstd")
        gross_mapping_predictions.to_parquet(stage / "feature_information_fold_local_gross_mapping_predictions.parquet", index=False, compression="zstd")
        current_netmap_diagnostics.to_parquet(stage / "feature_information_current_netmap_diagnostics.parquet", index=False, compression="zstd")
        residual_probes.to_parquet(stage / "feature_information_residual_probes.parquet", index=False, compression="zstd")
        residual_predictions.to_parquet(stage / "feature_information_residual_probe_predictions.parquet", index=False, compression="zstd")
        drift.to_parquet(stage / "feature_information_drift.parquet", index=False, compression="zstd")
        cohort_drift.to_parquet(stage / "feature_information_cohort_drift.parquet", index=False, compression="zstd")
        prediction_drift.to_parquet(stage / "feature_information_prediction_calibration_drift.parquet", index=False, compression="zstd")
        summary = {
            "schema": SCHEMA,
            "status": "DIAGNOSTIC_ONLY_NO_MODEL_OR_POLICY_PROMOTION",
            "rows": len(frame), "features_requested": len(features), "features_contract_scoped": int(inventory.contract_scoped.sum()),
            "features_causal_probe_eligible": int(inventory.causal_probe_eligible.sum()),
            "features_production_live_reuse_verified": int(inventory.production_live_reuse_eligible.sum()),
            "features_rejected_target_proximity": int(inventory.hard_reject_name.sum()),
            "folds": [{"side": f.side, "fold": f.fold, "start": f.start, "end": f.end, "train_rows": len(f.train_index), "test_rows": len(f.test_index)} for f in folds],
            "target": "gross_h12_bps", "gross_proxy_caveat": sorted(set(frame.get("gross_h12_proxy_status", pd.Series(["UNSPECIFIED"])).astype(str))),
            "directional_alpha_namespace": "FROZEN_STAGE0_OOF_ALPHA_INFORMATION_ONLY",
            "canonical_gross_mapping_namespace": "STRICT_SIDE_LOCAL_CHRONOLOGICAL_OOS_ISOTONIC_ALPHA_TO_GROSS_BPS",
            "canonical_gross_residual_definition": "gross_h12_bps - fold_local_oos_gross_mapped_prediction_bps for each head",
            "current_netmap_namespace": "FROZEN_STAGE0_OOF_NET_EXPECTED_EV_REFERENCE_ONLY",
            "current_netmap_residual_definition": "net_h12_bps - frozen_net_expected_ev_prediction_bps for each head",
            "canonical_gross_base_prediction_status": "FOLD_LOCAL_OOS_GROSS_MAP" if "canonical_gross_base_prediction" in frame.columns else ("SUPPLIED_EXTERNAL_GROSS_BPS" if base is not None else "NOT_RUN_MISSING_FROZEN_OOF_ALPHA_SCORE"),
            "canonical_gross_residual_stack_prediction_status": "FOLD_LOCAL_OOS_GROSS_MAP" if "canonical_gross_residual_stack_prediction" in frame.columns else ("SUPPLIED_EXTERNAL_GROSS_BPS" if residual is not None else "NOT_RUN_MISSING_FROZEN_OOF_ALPHA_SCORE"),
            "current_netmap_base_prediction_status": "STAGE0_OOF_NET_EXPECTED_EV_BPS" if economic_oof else "NOT_RUN_MISSING_FROZEN_OOF_NET_MAP",
            "current_netmap_residual_stack_prediction_status": "STAGE0_OOF_NET_EXPECTED_EV_BPS" if economic_oof else "NOT_RUN_MISSING_FROZEN_OOF_NET_MAP",
            "feature_contract_interpretation": "raw contract plus exact raw-panel cutoff establishes sealed research causal availability only; live reproducibility and staleness are NOT_VERIFIED unless supplied as per-feature evidence. Raw contract excludes H12 outcomes/path labels/maps/OOF/action fields; sensitive source classes remain explicitly flagged for source-specific availability review",
            "inputs_sha256": input_hashes,
        }
        write_json(stage / "feature_information_summary.json", summary)
        outputs = {path.name: sha256(path) for path in sorted(stage.iterdir())}
        manifest = {**summary, "outputs_sha256": outputs, "code_sha256": sha256(Path(__file__).resolve())}
        write_json(stage / "run_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--raw-panel", type=Path, default=DEFAULT_RAW_PANEL)
    parser.add_argument("--raw-contract", type=Path, default=DEFAULT_RAW_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-metadata", type=Path)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--residual-predictions", type=Path)
    parser.add_argument("--base-prediction-col")
    parser.add_argument("--residual-prediction-col")
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--min-train-rows", type=int, default=2_500)
    args = parser.parse_args()
    print(json.dumps(json_safe(run(
        ledger_path=args.ledger, raw_panel_path=args.raw_panel, raw_contract_path=args.raw_contract,
        output=args.output, feature_metadata_path=args.feature_metadata,
        base_predictions_path=args.base_predictions, residual_predictions_path=args.residual_predictions,
        base_prediction_col=args.base_prediction_col, residual_prediction_col=args.residual_prediction_col,
        max_features=args.max_features, max_folds=args.max_folds, min_train_rows=args.min_train_rows,
    )), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
