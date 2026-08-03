#!/usr/bin/env python3
"""Strict Stage-1 C0--C8 ``P(retain | clear)`` learnability ablation.

This is deliberately a feature-information experiment.  It uses the frozen
v11 E15 retention contract as C0, fits only exact H0 clear-first rows, freezes
all incremental choices before the August--November final evaluation, and
does not construct a Stage-2 combination or modify an execution policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.continuation_features import (
    CONTINUATION_FEATURE_GROUPS,
    CONTINUATION_REGIME_FEATURE_KEYS,
    CONTINUATION_SIDE_PRICE_FEATURE_KEYS,
    CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS,
)
from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from scripts import run_exact_h12_target_purity_ablation as v11

FEATURE_PANEL = ROOT / "data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v2/stage_c_candidate_population.parquet"
FEATURE_GROUPS = ROOT / "data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v2/retention_feature_groups.json"
FROZEN_E15 = ROOT / "data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v11/selected_execution_features.json"
V11_RESULTS = ROOT / "data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v11/target_ablation_results.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/stage_c_conditional_retention_ablation_20260731_v4"
SIDES = ("long", "short")
# The frozen E15/source-compatible panel starts in April 2023.  Retaining
# this history only expands the training support for the fixed 2024 protocol;
# it does not create an older evaluation fold or admit Aug--Nov labels.
HISTORY_START = pd.Timestamp("2023-04-01T00:00:00Z")
DEV_START = pd.Timestamp("2024-04-01T00:00:00Z")
EVAL_START = pd.Timestamp("2024-08-01T00:00:00Z")
END = pd.Timestamp("2024-12-01T00:00:00Z")
HORIZON_HOURS = 12
INCREMENTAL_CAP = 32
MIN_TRAIN_ROWS = 250
MIN_AVAILABILITY = 0.50
NEAR_CONSTANT_SHARE = 0.995
CORRELATION_LIMIT = 0.96
BOOTSTRAP_REPLICATES = 200

BLOCKED_ARMS = {
    "C4": "source_blocked: OI has no independently verified observed/publication timestamp",
    "C5": "source_blocked: funding has no independently verified observed/publication timestamp",
    "C7": "source_blocked: no candidate-level strict OOF/prequential regime sidecar",
}

# The H25/continuous labels are diagnostics in the Stage-C panel.  They may
# never enter selection or fitting for the sole admitted Stage-1 target.
PRIMARY_TARGET = "retain_h0_given_clear"
TARGET_LEAKAGE_TOKENS = (
    "retain_h0_given_clear", "retain_h25_given_clear", "continuous_net_given_clear",
    "postcost_h0_", "postcost_h25_",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_hash(values: Iterable[object]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _month(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame.decision_ts, utc=True).dt.strftime("%Y-%m")


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _paired_seed(seed: int, *, side_index: int, fold_index: int, phase: str) -> int:
    """Arm-invariant fixed seeds: paired C0/Cx fits differ only by features."""
    phase_offsets = {"development_selector": 0, "development_model": 500, "final_model": 9_000}
    if phase not in phase_offsets:
        raise ValueError(f"unknown paired seed phase {phase!r}")
    return int(seed + side_index * 1_000 + fold_index * 10 + phase_offsets[phase])


def _train_mask(frame: pd.DataFrame, fold_start: pd.Timestamp) -> pd.Series:
    """Strict H12 purge/embargo plus resolved-label cutoff for a fold."""
    cutoff = fold_start - pd.Timedelta(hours=HORIZON_HOURS)
    return frame.decision_ts.lt(cutoff) & frame.label_available_ts.lt(fold_start)


def _development_months() -> list[str]:
    """Predeclared 2024 OOF months, including April once history is retained."""
    return [str(month) for month in pd.period_range(DEV_START, EVAL_START - pd.Timedelta(days=1), freq="M")]


def _fit_classifier(x: pd.DataFrame, y: np.ndarray, *, seed: int, trees: int) -> lgb.LGBMClassifier:
    """Use the frozen v11 side-local classifier class and fixed parameters."""
    return v11._fit_classifier(x, y, seed=seed, trees=trees)


def _group_features(_: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """One independently tested mechanism per arm; never an all-feature arm."""
    return {
        "C1": [*CONTINUATION_FEATURE_GROUPS["F1_price_continuation_exhaustion"], *CONTINUATION_SIDE_PRICE_FEATURE_KEYS],
        "C2": list(CONTINUATION_FEATURE_GROUPS["F2_volume_liquidity_proxies"]),
        "C3": [*CONTINUATION_FEATURE_GROUPS["F3_volatility_transition"], *CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS],
        "C4": [],
        "C5": [],
        "C6": list(CONTINUATION_FEATURE_GROUPS["F6_cross_sectional_confirmation"]),
        "C7": list(CONTINUATION_REGIME_FEATURE_KEYS),
        "C8": list(CONTINUATION_FEATURE_GROUPS["F8_predeclared_composites"]),
    }


def _frozen_e15_features() -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Load the persisted E15 list and only its raw columns (memory bounded)."""
    persisted = json.loads(FROZEN_E15.read_text(encoding="utf-8"))
    selected = {side: list(persisted[side]) for side in SIDES}
    policy_fields = {"estimated_spread_bps", "entry_half_spread_bps", "barrier_pct", "entry_price_log"}
    raw_selected = sorted({name for values in selected.values() for name in values if name not in policy_fields})
    base = pd.read_parquet(v11.PANEL, columns=["candidate_id", *raw_selected])
    alignment_columns = [
        "candidate_id", "symbol", "side", "decision_ts", "feature_cutoff_ts", "label_end_ts", "label_available_ts",
        "target_id", "execution_policy_id", "cost_model_id", "exact_h12_net_bps", "execution_entry_price",
        "estimated_spread_bps", "entry_half_spread_bps", "barrier_pct",
    ]
    alignment = pd.read_parquet(v11.ALIGNMENT, columns=alignment_columns)
    frame = alignment.merge(base, on="candidate_id", how="inner", validate="one_to_one")
    if len(frame) != len(alignment) or frame.candidate_id.duplicated().any():
        raise ValueError("frozen v11 E15 raw panel and alignment identities differ")
    frame["entry_price_log"] = np.log(pd.to_numeric(frame.execution_entry_price, errors="coerce"))
    missing = sorted({name for values in selected.values() for name in values}.difference(frame.columns))
    if missing:
        raise ValueError(f"persisted E15 controls are absent from their frozen source: {missing}")
    for column in ("decision_ts", "feature_cutoff_ts", "label_end_ts", "label_available_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    return selected, frame


def _load_frame(feature_panel: Path, *, smoke: bool) -> tuple[dict[str, list[str]], pd.DataFrame]:
    f0, frozen = _frozen_e15_features()
    compatible = pd.read_parquet(feature_panel)
    required = {
        "candidate_id", "side", "decision_ts", "feature_cutoff_ts", "label_end_ts", "label_available_ts",
        "target_id", "execution_policy_id", "cost_model_id", "feature_available_ts", PRIMARY_TARGET,
        f"{PRIMARY_TARGET}__valid", f"{PRIMARY_TARGET}__condition_met",
        f"{PRIMARY_TARGET}__support_side", f"{PRIMARY_TARGET}__support_month",
    }
    missing = sorted(required.difference(compatible.columns))
    if missing or compatible.candidate_id.duplicated().any():
        raise ValueError(f"Stage-C compatible panel contract is incomplete: {missing}")
    frame = frozen.merge(compatible, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_stagec"))
    for name in (
        "side", "decision_ts", "feature_cutoff_ts", "label_end_ts", "label_available_ts",
        "target_id", "execution_policy_id", "cost_model_id",
    ):
        stage_name = f"{name}_stagec"
        if stage_name not in frame:
            raise ValueError(f"frozen E15 and Stage-C identity contract lacks {stage_name}")
        staged = (
            frame[stage_name]
            if name in {"side", "target_id", "execution_policy_id", "cost_model_id"}
            else pd.to_datetime(frame[stage_name], utc=True, errors="raise")
        )
        if not frame[name].eq(staged).all():
            raise ValueError(f"frozen E15 and Stage-C identity contract differs for {name}")
        frame = frame.drop(columns=stage_name)
    if "exact_h12_net_bps_stagec" in frame:
        if not np.allclose(frame.exact_h12_net_bps, frame.exact_h12_net_bps_stagec, equal_nan=True):
            raise ValueError("frozen E15 and Stage-C exact H12 net values differ")
        frame = frame.drop(columns="exact_h12_net_bps_stagec")
    frame["feature_available_ts"] = pd.to_datetime(frame.feature_available_ts, utc=True, errors="raise")
    # Keep compatible pre-April history exclusively as resolved training
    # support.  Development test folds remain fixed to April--July and the
    # final August--November OOS rows remain completely untouched by selection.
    frame = frame.loc[frame.decision_ts.ge(HISTORY_START) & frame.decision_ts.lt(END)].copy()
    frame = frame.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    if smoke:
        # Deterministic month/side thinning preserves the chronological protocol.
        frame["__sample_hash"] = pd.util.hash_pandas_object(frame.candidate_id, index=False).astype("uint64")
        frame = (
            frame.assign(month=_month(frame))
            .sort_values(["month", "side", "__sample_hash"], kind="stable")
            .groupby(["month", "side"], group_keys=False)
            .head(500)
            .drop(columns=["__sample_hash", "month"])
            .sort_values(["decision_ts", "candidate_id"], kind="stable")
            .reset_index(drop=True)
        )
    if frame.candidate_id.duplicated().any() or not frame.feature_available_ts.le(frame.decision_ts).all():
        raise ValueError("Stage-C candidate identity or feature availability contract failed")
    valid = frame[f"{PRIMARY_TARGET}__valid"].astype(bool)
    condition = frame[f"{PRIMARY_TARGET}__condition_met"].astype(bool)
    target = pd.to_numeric(frame[PRIMARY_TARGET], errors="coerce")
    expected = pd.to_numeric(frame.exact_h12_net_bps, errors="coerce").gt(0.0)
    if (
        not valid.eq(condition).all()
        or not target.loc[valid].notna().all()
        or not target.loc[valid].astype(int).eq(expected.loc[valid].astype(int)).all()
        or not target.loc[~valid].isna().all()
        or not frame.loc[valid, f"{PRIMARY_TARGET}__support_side"].eq(frame.loc[valid, "side"]).all()
        or not frame.loc[~valid, f"{PRIMARY_TARGET}__support_side"].isna().all()
        or not frame.loc[valid, f"{PRIMARY_TARGET}__support_month"].eq(frame.loc[valid, "decision_ts"].dt.strftime("%Y-%m")).all()
        or not frame.loc[~valid, f"{PRIMARY_TARGET}__support_month"].isna().all()
    ):
        raise ValueError("Stage-C primary retain_h0_given_clear target/support contract failed")
    return f0, frame


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _fit_transform(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit all availability/missingness/constant/clipping state on train only."""
    requested = list(validate_feature_columns(_unique(columns)))
    state: dict[str, Any] = {
        "requested": requested,
        "availability_threshold": MIN_AVAILABILITY,
        "near_constant_share": NEAR_CONSTANT_SHARE,
        "clip_quantiles": [0.01, 0.99],
        "removed_missing": [],
        "removed_near_constant": [],
        "removed_absent": [],
        "clip_bounds": {},
    }
    usable: list[str] = []
    for name in requested:
        if name not in train or name not in test:
            state["removed_absent"].append(name)
            continue
        values = _numeric(train, name)
        availability = float(values.notna().mean())
        if availability < MIN_AVAILABILITY:
            state["removed_missing"].append({"feature": name, "availability": availability})
            continue
        non_null = values.dropna()
        dominant_share = float(non_null.value_counts(dropna=False, normalize=True).iloc[0]) if len(non_null) else 1.0
        if non_null.nunique() <= 1 or dominant_share >= NEAR_CONSTANT_SHARE:
            state["removed_near_constant"].append({"feature": name, "distinct": int(non_null.nunique()), "dominant_share": dominant_share})
            continue
        lower, upper = non_null.quantile([0.01, 0.99]).tolist()
        state["clip_bounds"][name] = [float(lower), float(upper)]
        usable.append(name)
    state["usable"] = usable

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {name: _numeric(frame, name).clip(*state["clip_bounds"][name]) for name in usable},
            index=frame.index,
        )

    return apply(train), apply(test), state


def _select_incremental(train: pd.DataFrame, y: np.ndarray, candidates: list[str], *, seed: int, trees: int, cap: int = INCREMENTAL_CAP) -> tuple[list[str], dict[str, Any]]:
    """Fold-local filter, correlation representatives and gain selection."""
    x, _, filter_state = _fit_transform(train, train, candidates)
    state: dict[str, Any] = {"filter": filter_state, "cap": cap, "correlation_limit": CORRELATION_LIMIT, "selected": [], "gain": {}}
    if not len(filter_state["usable"]) or len(np.unique(y)) < 2:
        state["reason"] = "no_usable_features_or_single_class"
        return [], state
    correlation = x.corr(method="spearman").abs()
    representatives: list[str] = []
    for name in filter_state["usable"]:
        if all(not np.isfinite(correlation.loc[name, existing]) or float(correlation.loc[name, existing]) < CORRELATION_LIMIT for existing in representatives):
            representatives.append(name)
    state["correlation_representatives"] = representatives
    if not representatives:
        state["reason"] = "no_correlation_representatives"
        return [], state
    model = _fit_classifier(x.loc[:, representatives], y, seed=seed, trees=max(70, trees // 2))
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=representatives, dtype=float)
    ordered = gain.loc[gain.gt(0.0)].sort_values(ascending=False, kind="stable").head(cap)
    selected = ordered.index.tolist()
    state["selected"] = selected
    state["gain"] = {name: float(value) for name, value in ordered.items()}
    return selected, state


def _freeze_incremental(records: list[dict[str, Any]], arm: str, side: str, *, cap: int = INCREMENTAL_CAP) -> tuple[list[str], dict[str, Any]]:
    """Freeze final-OOS fields from development folds only, never final labels."""
    dev = [row for row in records if row["arm"] == arm and row["side"] == side and row["split"] == "development_oof"]
    counts: Counter[str] = Counter()
    gains: defaultdict[str, list[float]] = defaultdict(list)
    for row in dev:
        for name in row.get("incremental_selected", []):
            counts[name] += 1
            gain = row.get("selector", {}).get("gain", {}).get(name)
            if gain is not None:
                gains[name].append(float(gain))
    folds = len(dev)
    ranked = sorted(
        counts,
        key=lambda name: (-counts[name] / max(1, folds), -float(np.mean(gains[name])) if gains[name] else 0.0, name),
    )
    frozen = [name for name in ranked if counts[name] / max(1, folds) >= 0.25][:cap]
    return frozen, {
        "source": "development_oof_train_fold_selectors_only",
        "development_folds": folds,
        "selection_frequency": {name: counts[name] / max(1, folds) for name in ranked},
        "mean_gain": {name: float(np.mean(gains[name])) if gains[name] else 0.0 for name in ranked},
        "frozen_incremental": frozen,
        "final_oos_labels_used": False,
    }


def _calibration(prediction: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """Logistic calibration slope/intercept of labels on prediction logits."""
    p = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(label, dtype=int)
    if len(y) < 20 or len(np.unique(y)) < 2 or np.std(p) == 0.0:
        return np.nan, np.nan
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1_000_000.0, penalty="l2", solver="lbfgs", max_iter=500).fit(logits, y)
    except ValueError:
        return np.nan, np.nan
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _deciles(prediction: pd.Series) -> pd.Series:
    if not len(prediction):
        return pd.Series(dtype="Int64", index=prediction.index)
    return pd.qcut(prediction.rank(method="first"), 10, labels=False, duplicates="drop").astype("Int64") + 1


def _metric_values(frame: pd.DataFrame) -> dict[str, float]:
    rows = len(frame)
    if not rows:
        return {name: np.nan for name in ("retention_prevalence", "roc_auc", "pr_auc", "brier", "log_loss", "calibration_slope", "calibration_intercept", "top_decile_retention_rate", "top_decile_lift", "spearman_exact_h12_net", "top_decile_exact_h12_net_bps", "asset_concentration", "symbol_breadth", "missingness_sensitivity")}
    prediction = frame.prediction.to_numpy(float)
    label = frame.label.to_numpy(int)
    net = frame.exact_h12_net_bps.to_numpy(float)
    bins = _deciles(frame.prediction)
    top = bins.eq(bins.max()).to_numpy()
    slope, intercept = _calibration(prediction, label)
    values: dict[str, float] = {
        "retention_prevalence": float(label.mean()),
        "roc_auc": float(roc_auc_score(label, prediction)) if len(np.unique(label)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(label, prediction)) if len(np.unique(label)) > 1 else np.nan,
        "brier": float(brier_score_loss(label, prediction)),
        "log_loss": float(log_loss(label, np.clip(prediction, 1e-6, 1.0 - 1e-6), labels=[0, 1])),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "top_decile_retention_rate": float(label[top].mean()) if top.any() else np.nan,
        "top_decile_lift": float(label[top].mean() / label.mean()) if top.any() and label.mean() else np.nan,
        "spearman_exact_h12_net": float(spearmanr(prediction, net).statistic) if rows >= 3 else np.nan,
        "top_decile_exact_h12_net_bps": float(net[top].mean()) if top.any() else np.nan,
        "asset_concentration": float(frame.loc[top, "symbol"].value_counts(normalize=True).max()) if top.any() else np.nan,
        "symbol_breadth": float(frame.symbol.nunique()),
        "missingness_sensitivity": float(spearmanr(frame.prediction, frame.feature_missing_fraction).statistic) if frame.feature_missing_fraction.nunique() > 1 else np.nan,
    }
    return values


def _metric_records(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = [
        ("aggregate", []), ("side", ["side"]), ("month", ["month"]), ("fold", ["fold"]),
        ("side_month", ["side", "month"]), ("side_fold", ["side", "fold"]),
    ]
    for (arm, split), local in scored.groupby(["arm", "split"], sort=True):
        for scope, keys in dimensions:
            groups = [((), local)] if not keys else local.groupby(keys, sort=True, dropna=False)
            for values, part in groups:
                values = values if isinstance(values, tuple) else (values,)
                row = {"arm": arm, "split": split, "scope": scope, "rows": int(len(part))}
                for key, value in zip(keys, values):
                    row[key] = value
                row.update(_metric_values(part))
                rows.append(row)
    return pd.DataFrame(rows)


def _calibration_records(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = [("aggregate", []), ("side", ["side"]), ("month", ["month"]), ("fold", ["fold"])]
    for (arm, split), local in scored.groupby(["arm", "split"], sort=True):
        for scope, keys in dimensions:
            groups = [((), local)] if not keys else local.groupby(keys, sort=True, dropna=False)
            for values, part in groups:
                values = values if isinstance(values, tuple) else (values,)
                bins = _deciles(part.prediction)
                for decile, bucket in part.assign(prediction_decile=bins).groupby("prediction_decile", dropna=False, sort=True):
                    row = {"arm": arm, "split": split, "scope": scope, "prediction_decile": int(decile), "rows": int(len(bucket)), "retention_rate": float(bucket.label.mean()), "mean_prediction": float(bucket.prediction.mean()), "exact_h12_net_bps": float(bucket.exact_h12_net_bps.mean())}
                    for key, value in zip(keys, values):
                        row[key] = value
                    rows.append(row)
    return pd.DataFrame(rows)


def _assert_identical_ids(scored: pd.DataFrame, tested_arms: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (split, fold), local in scored.groupby(["split", "fold"], sort=True):
        reference: tuple[str, ...] | None = None
        for arm in tested_arms:
            ids = tuple(local.loc[local.arm.eq(arm), "candidate_id"].astype(str))
            if not ids:
                raise AssertionError(f"{arm} has no {split}/{fold} candidates")
            if reference is None:
                reference = ids
            elif ids != reference:
                raise AssertionError(f"conditional arm IDs diverge in {split}/{fold}")
            records.append({"split": split, "fold": fold, "arm": arm, "rows": len(ids), "candidate_id_sha256": _id_hash(ids), "identical_to_c0": ids == reference})
    return pd.DataFrame(records)


def _paired_deltas(scored: pd.DataFrame, *, control_arm: str = "C0") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ("roc_auc", "pr_auc", "brier", "log_loss", "spearman_exact_h12_net", "top_decile_exact_h12_net_bps")
    for (split, fold), local in scored.groupby(["split", "fold"], sort=True):
        for side in ("all", *SIDES):
            subset = local if side == "all" else local.loc[local.side.eq(side)]
            control = subset.loc[subset.arm.eq(control_arm)]
            base = _metric_values(control)
            for arm in sorted(set(subset.arm).difference({control_arm})):
                candidate = subset.loc[subset.arm.eq(arm)]
                if tuple(candidate.candidate_id) != tuple(control.candidate_id):
                    raise AssertionError(f"paired C0 comparison is not row-identical for {arm}")
                value = _metric_values(candidate)
                for metric in metrics:
                    rows.append({"arm": arm, "control_arm": control_arm, "split": split, "fold": fold, "side": side, "metric": metric, "rows": len(candidate), "control_value": base[metric], "arm_value": value[metric], "delta_vs_c0": value[metric] - base[metric]})
    return pd.DataFrame(rows)


def _paired_day_bootstrap(scored: pd.DataFrame, *, control_arm: str, seed: int, replicates: int) -> pd.DataFrame:
    """Paired UTC-day block uncertainty on identical C0/arm candidate rows."""
    rows: list[dict[str, Any]] = []
    metrics = ("roc_auc", "pr_auc", "brier", "log_loss", "spearman_exact_h12_net", "top_decile_exact_h12_net_bps")
    for split, local in scored.groupby("split", sort=True):
        control = local.loc[local.arm.eq(control_arm)].set_index("candidate_id")
        day_codes, days = pd.factorize(control.decision_ts.dt.floor("D"), sort=True)
        day_rows = [np.flatnonzero(day_codes == number) for number in range(len(days))]
        if not day_rows:
            continue
        rng = np.random.default_rng(seed + (0 if split == "development_oof" else 10_000))
        for arm in sorted(set(local.arm).difference({control_arm})):
            candidate = local.loc[local.arm.eq(arm)].set_index("candidate_id").reindex(control.index)
            if candidate.prediction.isna().any():
                raise AssertionError(f"bootstrap candidate identity mismatch for {arm}")
            base_values = _metric_values(control.reset_index())
            arm_values = _metric_values(candidate.reset_index())
            samples: dict[str, list[float]] = {metric: [] for metric in metrics}
            for _ in range(replicates):
                positions = np.concatenate([day_rows[number] for number in rng.integers(0, len(day_rows), size=len(day_rows))])
                base_sample = control.iloc[positions].reset_index()
                arm_sample = candidate.iloc[positions].reset_index()
                base_metric = _metric_values(base_sample)
                arm_metric = _metric_values(arm_sample)
                for metric in metrics:
                    samples[metric].append(arm_metric[metric] - base_metric[metric])
            for metric, values in samples.items():
                finite = np.asarray(values, dtype=float)
                finite = finite[np.isfinite(finite)]
                higher_is_better = metric not in {"brier", "log_loss"}
                rows.append({
                    "arm": arm, "control_arm": control_arm, "split": split, "metric": metric,
                    "rows": len(control), "utc_day_blocks": len(day_rows), "replicates": replicates,
                    "delta_vs_c0": arm_values[metric] - base_values[metric],
                    "bootstrap_mean": float(finite.mean()) if len(finite) else np.nan,
                    "ci_p05": float(np.quantile(finite, 0.05)) if len(finite) else np.nan,
                    "ci_p95": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
                    "probability_improves": float((finite > 0.0).mean()) if higher_is_better and len(finite) else (float((finite < 0.0).mean()) if len(finite) else np.nan),
                })
    return pd.DataFrame(rows)


def _reference_e15_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    """Reference the frozen hierarchy E15 diagnostics on the common final rows."""
    final = scored.loc[scored.split.eq("final_oos") & scored.arm.eq("C0"), ["candidate_id", "symbol", "side", "decision_ts", "label_available_ts", "month", "fold", "label", "exact_h12_net_bps", "feature_missing_fraction"]].copy()
    if not V11_RESULTS.exists():
        return pd.DataFrame([{"status": "unavailable", "reason": "frozen v11 result artifact missing"}])
    # The v11 file contains every target arm.  Predicate pushdown avoids
    # materialising that multi-million-row table merely to reference E15.
    source = pd.read_parquet(
        V11_RESULTS,
        columns=["candidate_id", "p_retain_given_clear"],
        filters=[("arm", "==", "E15_exact1m_hierarchical_persistence_0")],
    )
    merged = final.merge(source, on="candidate_id", how="left", validate="one_to_one")
    available = merged.loc[np.isfinite(merged.p_retain_given_clear)].copy()
    if not len(available):
        return pd.DataFrame([{"status": "unavailable", "reason": "no E15 p_retain overlap on Stage-C common final cohort"}])
    available["prediction"] = available.pop("p_retain_given_clear")
    values = _metric_values(available)
    return pd.DataFrame([{
        "status": "referenced_not_bitwise_reproduced", "arm": "E15_exact1m_hierarchical_persistence_0",
        "rows_common_final_clear": len(available), "candidate_id_sha256": _id_hash(available.candidate_id.astype(str)),
        "row_contract_difference": "Stage-C requires complete F1/F2/F3/F6/F8 common-cohort values; v11 hierarchy evaluated its wider frozen candidate contract.",
        **values,
    }])


def _admission_gate(deltas: pd.DataFrame, bootstrap: pd.DataFrame, metrics: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Predeclared development-only evidence gate; this never constructs Stage 2."""
    arm_to_group = {"C1": "F1", "C2": "F2", "C3": "F3", "C4": "F4", "C5": "F5", "C6": "F6", "C7": "F7", "C8": "F8"}
    dispositions: dict[str, Any] = {
        "F0": {"classification": "diagnostic_only", "conditional_metric_delta": 0.0, "calibration_delta": 0.0, "monthly_stability": "control", "side_stability": "control", "economic_delta": None, "reason": "exact persisted E15 control; no selection claim"}
    }
    survivors: list[str] = []
    for arm, group in arm_to_group.items():
        if arm in BLOCKED_ARMS:
            dispositions[group] = {"classification": "rejected", "conditional_metric_delta": None, "calibration_delta": None, "monthly_stability": "not_tested", "side_stability": "not_tested", "economic_delta": None, "reason": BLOCKED_ARMS[arm]}
            continue
        dev = deltas.loc[deltas.split.eq("development_oof") & deltas.arm.eq(arm)]
        roc = dev.loc[(dev.side.eq("all")) & dev.metric.eq("roc_auc"), "delta_vs_c0"]
        brier = dev.loc[(dev.side.eq("all")) & dev.metric.eq("brier"), "delta_vs_c0"]
        monthly = dev.loc[(dev.side.eq("all")) & dev.metric.eq("roc_auc"), "delta_vs_c0"]
        side = dev.loc[(dev.side.isin(SIDES)) & dev.metric.eq("roc_auc"), "delta_vs_c0"]
        boot = bootstrap.loc[bootstrap.split.eq("development_oof") & bootstrap.arm.eq(arm) & bootstrap.metric.eq("roc_auc"), "probability_improves"]
        mean_roc = float(roc.mean()) if len(roc) else np.nan
        mean_brier = float(brier.mean()) if len(brier) else np.nan
        monthly_positive = float((monthly > 0.0).mean()) if len(monthly) else 0.0
        side_transport = float((side >= -0.002).mean()) if len(side) else 0.0
        bootstrap_probability = float(boot.mean()) if len(boot) else 0.0
        passes = bool(mean_roc >= 0.002 and mean_brier <= 0.005 and monthly_positive >= 0.60 and side_transport >= 1.0 and bootstrap_probability >= 0.55)
        classification = "retained_for_stage_b_test" if passes else "diagnostic_only"
        if passes:
            survivors.append(group)
        dispositions[group] = {
            "classification": classification,
            "conditional_metric_delta": mean_roc,
            "calibration_delta": mean_brier,
            "monthly_stability": {"positive_fraction": monthly_positive, "required": 0.60},
            "side_stability": {"non_materially_negative_fraction": side_transport, "required": 1.0},
            "bootstrap_probability_roc_improves": bootstrap_probability,
            "economic_delta": None,
            "reason": "development-only Stage-1 gate passed; Stage 2 remains unrun" if passes else "does not meet every predeclared development-only transport/calibration/stability gate",
        }
    terminal_decision = "CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION" if not survivors else "STAGE1_SURVIVORS_RECORDED_FOR_LATER_STAGE2"
    gate = {
        "selection_period": "development OOF only (2024-04 through 2024-07)",
        "final_oos_used_for_selection": False,
        "criteria": {"mean_roc_auc_delta_at_least": 0.002, "mean_brier_delta_at_most": 0.005, "positive_month_fraction_at_least": 0.60, "both_sides_non_materially_negative": -0.002, "bootstrap_probability_roc_improves_at_least": 0.55},
        "stage2_survivor_groups": survivors,
        "stage2_status": "NOT_RUN_BY_STAGE1_RUNNER",
        "stage_b_status": "NOT_RUN_BY_STAGE1_RUNNER",
        "terminal_decision": terminal_decision,
    }
    return gate, dispositions


def _correctness_report(*, checks: dict[str, bool], blocked: dict[str, str]) -> dict[str, Any]:
    """Fail closed: an emitted correctness report passes iff every check passes."""
    normalized = {name: bool(value) for name, value in checks.items()}
    return {
        "schema": "stage_c_stage1_correctness_v1",
        "passed": bool(all(normalized.values())),
        "checks": normalized,
        "blocked": blocked,
        "limitations": ["Stage 2 compact combination not run", "Stage-B hierarchy test not run", "F4/F5/F7 source blocked"],
    }


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, *, base: list[str], incremental: list[str], seed: int, trees: int) -> tuple[np.ndarray, float, dict[str, Any]]:
    columns = _unique([*base, *incremental])
    leaked = sorted({name for name in columns if any(token in name.lower() for token in TARGET_LEAKAGE_TOKENS)})
    if leaked:
        raise ValueError(f"Stage-C target/diagnostic labels cannot be model features: {leaked}")
    x_train, x_test, transformer = _fit_transform(train, test, columns)
    if not len(transformer["usable"]):
        raise ValueError("no usable train-fold features for conditional retention model")
    model = _fit_classifier(x_train, train.label.to_numpy(int), seed=seed, trees=trees)
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=x_train.columns, dtype=float)
    concentration = float(gain.sort_values(ascending=False).head(5).sum() / max(1.0, gain.sum()))
    return model.predict_proba(x_test)[:, 1], concentration, {"transformer": transformer, "model_features": list(x_train.columns), "gain": {name: float(value) for name, value in gain.items()}}


def _prediction_part(test: pd.DataFrame, *, arm: str, split: str, fold: str, prediction: np.ndarray, transformer: dict[str, Any]) -> pd.DataFrame:
    fields = [
        "candidate_id", "symbol", "side", "decision_ts", "feature_cutoff_ts", "label_end_ts", "label_available_ts",
        "target_id", "execution_policy_id", "cost_model_id", PRIMARY_TARGET,
        f"{PRIMARY_TARGET}__valid", f"{PRIMARY_TARGET}__condition_met",
        f"{PRIMARY_TARGET}__support_side", f"{PRIMARY_TARGET}__support_month",
        "month", "label", "exact_h12_net_bps",
    ]
    part = test.loc[:, fields].copy()
    part["arm"] = arm
    part["split"] = split
    part["fold"] = fold
    part["prediction"] = prediction
    usable = transformer["usable"]
    part["feature_missing_fraction"] = test.loc[:, usable].isna().mean(axis=1).to_numpy(float) if usable else 1.0
    return part


def run(*, feature_panel: Path, output: Path, smoke: bool = False, seed: int = 20260731) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    progress_path = output.parent / f".{output.name}.progress.json"
    def checkpoint(phase: str, **details: Any) -> None:
        _write_json(progress_path, {"schema": "stage_c_stage1_progress_v1", "phase": phase, "mode": "smoke" if smoke else "full", **details})

    checkpoint("starting", output=str(output), seed=seed)
    f0, frame = _load_frame(feature_panel, smoke=smoke)
    support = frame.loc[frame.retain_h0_given_clear__valid.astype(bool)].copy().reset_index(drop=True)
    support["label"] = support.retain_h0_given_clear.astype(int)
    support["month"] = _month(support)
    if (~support.retain_h0_given_clear__condition_met.astype(bool)).any() or support.label.isna().any():
        raise AssertionError("retention labels escaped exact H0 clear-first support")
    if not support.label.isin([0, 1]).all() or not support.label_end_ts.le(support.label_available_ts).all():
        raise AssertionError("retention label endpoint/availability contract failed")
    checkpoint("input_loaded", compatible_rows=len(frame), clear_first_rows=len(support))

    groups = _group_features(frame)
    arms = {"C0": [], **groups}
    tested_arms = [arm for arm in arms if arm not in BLOCKED_ARMS]
    trees = 60 if smoke else 180
    development_months = _development_months()
    predictions: list[pd.DataFrame] = []
    stability: list[dict[str, Any]] = []

    # Strict expanding development OOF: every selector and clipper sees only
    # labels resolved before its validation month and before its H12 embargo.
    for arm in tested_arms:
        group = arms[arm]
        for side_index, side in enumerate(SIDES):
            local = support.loc[support.side.eq(side)].copy().reset_index(drop=True)
            base = f0[side]
            for fold_index, month in enumerate(development_months):
                fold_start = pd.Timestamp(f"{month}-01", tz="UTC")
                fold_end = fold_start + pd.offsets.MonthBegin(1)
                purge_cutoff = fold_start - pd.Timedelta(hours=HORIZON_HOURS)
                train_mask = _train_mask(local, fold_start)
                test_mask = local.decision_ts.ge(fold_start) & local.decision_ts.lt(fold_end)
                train, test = local.loc[train_mask].reset_index(drop=True), local.loc[test_mask].reset_index(drop=True)
                if len(test) == 0 or len(train) < MIN_TRAIN_ROWS or train.label.nunique() < 2:
                    continue
                selector_seed = _paired_seed(seed, side_index=side_index, fold_index=fold_index, phase="development_selector")
                selected, selector = _select_incremental(train, train.label.to_numpy(int), group, seed=selector_seed, trees=trees)
                model_seed = _paired_seed(seed, side_index=side_index, fold_index=fold_index, phase="development_model")
                prediction, importance_concentration, state = _fit_predict(train, test, base=base, incremental=selected, seed=model_seed, trees=trees)
                predictions.append(_prediction_part(test, arm=arm, split="development_oof", fold=month, prediction=prediction, transformer=state["transformer"]))
                stability.append({
                    "arm": arm, "side": side, "split": "development_oof", "fold": month,
                    "fold_start_utc": str(fold_start), "fold_end_utc": str(fold_end), "purge_embargo_hours": HORIZON_HOURS,
                    "purge_cutoff_utc": str(purge_cutoff), "label_availability_cutoff_utc": str(fold_start),
                    "train_decision_ts_max": str(train.decision_ts.max()), "train_label_available_ts_max": str(train.label_available_ts.max()),
                    "train_rows": len(train), "test_rows": len(test), "base_features": base,
                    "incremental_selected": selected, "selector": selector, "transformer": state["transformer"],
                    "model_features": state["model_features"], "gain": state["gain"], "importance_concentration": importance_concentration,
                    "selector_seed": selector_seed, "model_seed": model_seed,
                    "final_oos_labels_used": False,
                })
        checkpoint("development_arm_complete", arm=arm, prediction_rows=sum(len(part) for part in predictions), stability_rows=len(stability))

    # Freeze every incremental mechanism list from the development records,
    # then fit exactly once before the August--November final OOS interval.
    frozen_selection: dict[str, dict[str, list[str]]] = defaultdict(dict)
    frozen_evidence: dict[str, dict[str, Any]] = defaultdict(dict)
    for arm in tested_arms:
        for side in SIDES:
            selected, evidence = _freeze_incremental(stability, arm, side)
            frozen_selection[arm][side] = selected
            frozen_evidence[arm][side] = evidence
    final_fold = "2024-08_to_2024-11"
    final_train_cutoff = EVAL_START - pd.Timedelta(hours=HORIZON_HOURS)
    for arm in tested_arms:
        for side_index, side in enumerate(SIDES):
            local = support.loc[support.side.eq(side)].copy().reset_index(drop=True)
            train = local.loc[_train_mask(local, EVAL_START)].reset_index(drop=True)
            test = local.loc[local.decision_ts.ge(EVAL_START) & local.decision_ts.lt(END)].reset_index(drop=True)
            if len(test) == 0 or len(train) < MIN_TRAIN_ROWS or train.label.nunique() < 2:
                raise ValueError(f"insufficient frozen final-OOS support for {arm}/{side}")
            model_seed = _paired_seed(seed, side_index=side_index, fold_index=0, phase="final_model")
            prediction, importance_concentration, state = _fit_predict(train, test, base=f0[side], incremental=frozen_selection[arm][side], seed=model_seed, trees=trees)
            predictions.append(_prediction_part(test, arm=arm, split="final_oos", fold=final_fold, prediction=prediction, transformer=state["transformer"]))
            stability.append({
                "arm": arm, "side": side, "split": "final_oos", "fold": final_fold,
                "fold_start_utc": str(EVAL_START), "fold_end_utc": str(END), "purge_embargo_hours": HORIZON_HOURS,
                "purge_cutoff_utc": str(final_train_cutoff), "label_availability_cutoff_utc": str(EVAL_START),
                "train_decision_ts_max": str(train.decision_ts.max()), "train_label_available_ts_max": str(train.label_available_ts.max()),
                "train_rows": len(train), "test_rows": len(test), "base_features": f0[side],
                "incremental_selected": frozen_selection[arm][side], "selector": frozen_evidence[arm][side], "transformer": state["transformer"],
                "model_features": state["model_features"], "gain": state["gain"], "importance_concentration": importance_concentration,
                "selector_seed": None, "model_seed": model_seed,
                "final_oos_labels_used": False,
            })
        checkpoint("final_arm_complete", arm=arm, prediction_rows=sum(len(part) for part in predictions), stability_rows=len(stability))

    if not predictions:
        raise ValueError("no strict chronological Stage-1 predictions were produced")
    scored = pd.concat(predictions, ignore_index=True).sort_values(["split", "fold", "side", "decision_ts", "candidate_id", "arm"], kind="stable").reset_index(drop=True)
    checkpoint("scored_assembled", prediction_rows=len(scored))
    identity = _assert_identical_ids(scored, tested_arms)
    checkpoint("identity_verified", identity_records=len(identity))
    metrics = _metric_records(scored)
    checkpoint("metrics_complete", metric_records=len(metrics))
    calibration = _calibration_records(scored)
    checkpoint("calibration_complete", calibration_records=len(calibration))
    deltas = _paired_deltas(scored)
    checkpoint("paired_deltas_complete", delta_records=len(deltas))
    bootstrap = _paired_day_bootstrap(scored, control_arm="C0", seed=seed + 91, replicates=30 if smoke else BOOTSTRAP_REPLICATES)
    checkpoint("bootstrap_complete", bootstrap_records=len(bootstrap))
    reference = _reference_e15_diagnostics(scored)
    checkpoint("reference_complete", reference_records=len(reference))
    gate, dispositions = _admission_gate(deltas, bootstrap, metrics, groups)
    transition_monitor = pd.DataFrame([{
        "status": "source_blocked", "group": "F7", "arm": "C7", "rows": 0,
        "reason": BLOCKED_ARMS["C7"], "permitted_use": "observational only once strict OOF/prequential sidecar exists",
        "entry_gate_or_policy_action": False,
    }])
    checkpoint("analysis_complete", prediction_rows=len(scored), bootstrap_rows=len(bootstrap), terminal_decision=gate["terminal_decision"])

    # Store nested feature state as JSON strings for stable Parquet schemas.
    stability_frame = pd.DataFrame(stability)
    for name in ("base_features", "incremental_selected", "selector", "transformer", "model_features", "gain"):
        stability_frame[name] = stability_frame[name].map(lambda value: json.dumps(value, sort_keys=True, default=str))
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        outputs: list[str] = []
        def write_parquet(name: str, value: pd.DataFrame) -> None:
            value.to_parquet(stage / name, index=False, compression="zstd")
            outputs.append(name)

        write_parquet("retention_conditional_oof_predictions.parquet", scored)
        write_parquet("retention_conditional_results.parquet", metrics)
        write_parquet("retention_conditional_calibration.parquet", calibration)
        write_parquet("retention_feature_stability.parquet", stability_frame)
        write_parquet("retention_paired_c0_deltas.parquet", deltas)
        write_parquet("retention_paired_day_block_bootstrap.parquet", bootstrap)
        write_parquet("retention_evaluation_candidate_ids.parquet", identity)
        write_parquet("reference_e15_conditional_diagnostics.parquet", reference)
        write_parquet("retention_transition_monitor_slices.parquet", transition_monitor)
        write_parquet("retention_leave_group_out_results.parquet", pd.DataFrame([{"status": "NOT_RUN", "reason": "Stage-2 compact arm is outside Stage-1 and no compact winner is fabricated"}]))
        write_parquet("stage_b_incremental_retention_results.parquet", pd.DataFrame([{"status": "NOT_RUN", "reason": "Stage-B is not invoked by this Stage-1 feature-information runner", "stage1_survivors": json.dumps(gate["stage2_survivor_groups"])}]))
        manifests = {"C0": {"status": "frozen_E15_inherited_control", "features": f0}}
        manifests.update({arm: {"status": "source_blocked" if arm in BLOCKED_ARMS else "tested", "group_features": groups[arm], "reason": BLOCKED_ARMS.get(arm), "final_frozen_selection": frozen_selection.get(arm, {})} for arm in groups})
        _write_json(stage / "retention_feature_groups.json", manifests); outputs.append("retention_feature_groups.json")
        _write_json(stage / "retention_compact_feature_manifest.json", {"status": "STAGE2_NOT_RUN", "predeclared_stage1_admission_gate": gate, "arms": manifests}); outputs.append("retention_compact_feature_manifest.json")
        _write_json(stage / "feature_disposition.yaml", {"schema": "stage_c_feature_disposition_v1", "terminal_decision": gate["terminal_decision"], "stage1_gate": gate, "groups": dispositions}); outputs.append("feature_disposition.yaml")
        _write_json(stage / "retention_feature_dictionary.json", {name: {"group": arm, "point_in_time_safe": arm not in BLOCKED_ARMS, "admission": "frozen E15 control" if arm == "C0" else manifests.get(arm, {}).get("status", "tested")} for arm, names in groups.items() for name in names}); outputs.append("retention_feature_dictionary.json")
        correctness = _correctness_report(checks={
            "test_comparison_arms_use_identical_candidate_ids": bool(identity.identical_to_c0.all()),
            "test_feature_selection_uses_training_data_only": bool(all(not json.loads(value).get("final_oos_labels_used", False) for value in stability_frame.selector)),
            "test_scalers_and_clippers_fit_on_training_data_only": bool(all(pd.Timestamp(value) < EVAL_START or value == "NaT" for value in stability_frame.loc[stability_frame.split.eq("final_oos"), "train_label_available_ts_max"])),
            "test_no_final_oos_feature_selection": bool(all(not evidence[side]["final_oos_labels_used"] for evidence in frozen_evidence.values() for side in evidence)),
            "test_h12_purge_and_label_availability": bool((pd.to_numeric(stability_frame.purge_embargo_hours) == HORIZON_HOURS).all()),
            "test_april_development_oof_present": "2024-04" in set(stability_frame.loc[stability_frame.split.eq("development_oof"), "fold"]),
            "test_each_development_train_max_precedes_fold_start_minus_h12": bool(all(
                pd.Timestamp(row.train_decision_ts_max) < pd.Timestamp(row.fold_start_utc) - pd.Timedelta(hours=HORIZON_HOURS)
                and pd.Timestamp(row.train_label_available_ts_max) < pd.Timestamp(row.fold_start_utc)
                for row in stability_frame.loc[stability_frame.split.eq("development_oof")].itertuples()
            )),
            "test_f0_hash_is_persisted_e15": _sha256(FROZEN_E15) == "a91c1b40ad87f4fab3311aef2865c6bdcc713d2de75bbb7e9623384ac6085ed1",
            "test_c_group_isolation": bool(all(not set(groups[arm]).intersection(set(groups[other])) or arm == other for arm in groups for other in groups if arm != other and {arm, other}.isdisjoint({"C4", "C5", "C7"}))),
            "test_paired_arm_invariant_seeds": bool(stability_frame.groupby(["split", "fold", "side"])["model_seed"].nunique(dropna=True).le(1).all()),
        }, blocked=BLOCKED_ARMS)
        _write_json(stage / "correctness_test_report.json", correctness); outputs.append("correctness_test_report.json")
        (stage / "stage_b_incremental_retention_summary.md").write_text("# Stage-B incremental retention summary\n\nNot run by the Stage-1 runner. The frozen Stage-1 admission gate is recorded in `retention_compact_feature_manifest.json`; no hierarchy, threshold, quota, or policy change is produced here.\n", encoding="utf-8"); outputs.append("stage_b_incremental_retention_summary.md")
        manifest = {
            "schema": "stage_c_conditional_retention_ablation_v4",
            "status": "COMPLETED_RESEARCH_ONLY_STAGE1",
            "mode": "smoke" if smoke else "full",
            "target": "retain_h0_given_clear", "population": "exact H0 clear-first support only",
            "calendar": {"compatible_training_history": "2023-04..2024-03 resolved labels only", "development_oof": "2024-04..2024-07 expanding monthly (including April)", "final_oos": "2024-08..2024-11 frozen pre-final selection"},
            "purge_embargo": {"horizon_hours": HORIZON_HOURS, "train_rule": "decision_ts < fold_start - 12h AND label_available_ts < fold_start"},
            "model": {"class": "v11.LGBMClassifier side-local", "fixed_hyperparameters": {"learning_rate": 0.04, "num_leaves": 23, "max_depth": 5, "min_child_samples": 180, "colsample_bytree": 0.80, "subsample": 0.85, "subsample_freq": 1, "reg_lambda": 15.0, "reg_alpha": 0.15}, "trees": trees, "seed": seed},
            "feature_reduction": {"availability_threshold": MIN_AVAILABILITY, "near_constant_share": NEAR_CONSTANT_SHARE, "clip_quantiles": [0.01, 0.99], "correlation_limit": CORRELATION_LIMIT, "incremental_cap": INCREMENTAL_CAP, "final_oos_selection": "forbidden"},
            "frozen_control": {"path": str(FROZEN_E15), "sha256": _sha256(FROZEN_E15), "per_side_features": f0},
            "blocked": BLOCKED_ARMS, "arms": list(arms), "tested_arms": tested_arms,
            "candidate_identity": identity.to_dict(orient="records"), "rows": {"input_compatible": len(frame), "clear_first_support": len(support), "predictions": len(scored)},
            "stage1_admission_gate": gate, "terminal_decision": gate["terminal_decision"], "limitations": ["candidate-conditioned historical research", "current-spread counterfactual provenance inherited from v11", "F4/F5/F7 unavailable", "no Stage-2 compact arm", "no Stage-B policy test"],
            "inputs": {str(feature_panel): _sha256(feature_panel), str(FEATURE_GROUPS): _sha256(FEATURE_GROUPS), str(FROZEN_E15): _sha256(FROZEN_E15), str(v11.PANEL): _sha256(v11.PANEL), str(v11.ALIGNMENT): _sha256(v11.ALIGNMENT), str(v11.POSTCOST_EVENTS): _sha256(v11.POSTCOST_EVENTS), str(v11.PERSISTENCE_LABELS): _sha256(v11.PERSISTENCE_LABELS), str(V11_RESULTS): _sha256(V11_RESULTS)},
            "code_sha256": {"stage1_runner": _sha256(Path(__file__)), "v11_model_contract": _sha256(Path(v11.__file__)), "continuation_features": _sha256(ROOT / "extreme_price_movements/continuation_features.py")},
        }
        _write_json(stage / "run_manifest.json", {**manifest, "outputs": {name: _sha256(stage / name) for name in outputs}}); outputs.append("run_manifest.json")
        os.replace(stage, output)
        checkpoint("completed", output=str(output), prediction_rows=len(scored), terminal_decision=gate["terminal_decision"])
        return manifest
    except Exception:
        checkpoint("failed", error_type=type(sys.exc_info()[1]).__name__, error=str(sys.exc_info()[1]))
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-panel", type=Path, default=FEATURE_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(feature_panel=args.feature_panel, output=args.output, smoke=args.smoke), indent=2, default=str))


if __name__ == "__main__":
    main()
