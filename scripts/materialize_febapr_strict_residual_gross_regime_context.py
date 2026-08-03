#!/usr/bin/env python3
"""Materialize an immutable pre-entry gross-opportunity/regime context panel.

This deliberately reads only identity/timestamp fields and archived signal-time
features.  It never reads execution outcomes, path outcomes, or labels from
the March--April residual OOF source.  The panel is an input contract for a
subsequent challenger, not an outcome analysis or trained model.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESIDUAL = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/oof_predictions.parquet"
LABEL_ROOT = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
OUT = ROOT / "data_perp/artifacts/febapr2025_strict_residual_gross_regime_context_20260729_v3"

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__", "__decision_ts__")
CORE_FEATURES = (
    "range_24h_pct",
    "__meta_raw__volatility_zscore",
    "trend_r2_24",
    "jump_intensity",
    "__meta_raw__chop_score",
)
# These were archived as signal-time regime-source composites.  The one whose
# name itself contains ``path`` is intentionally omitted by the forbidden-name
# gate, even though its source construction may be pre-entry.
REGIME_SOURCE_FEATURES = (
    "__regime_source_shock_impulse_score__",
    "__regime_source_execution_quality_score__",
    "__regime_source_execution_risk_score__",
    "__regime_source_oi_agreement_score__",
    "__regime_source_location_quality_score__",
    "__regime_source_pullback_retest_score__",
    "__regime_source_compression_score__",
    "__regime_source_volume_confirmation_score__",
    "__regime_source_barrier_pressure_score__",
    "__regime_source_quiet_continuation_score__",
    "__regime_source_loud_breakout_impulse_score__",
    "__regime_source_dirty_shock_avoid_score__",
    "__regime_source_retest_reversal_score__",
    "__regime_source_compression_release_score__",
    "__regime_source_base_positive_source_score__",
    "__regime_source_prior_recent_source_strength__",
    "__regime_source_run_entry_score__",
    "__regime_source_late_run_continuation_score__",
    "__regime_source_not_dirty_shock_score__",
    "__regime_source_loud_clean_source_score__",
    "__regime_source_barrier_relief_score__",
    "__regime_source_clean_execution_context_score__",
    "__regime_source_calm_positive_source_score__",
    "__regime_source_loud_clean_execution_score__",
    "__regime_source_clean_run_entry_score__",
    "__regime_source_compression_capture_candidate_score__",
    "__regime_source_risk_adjusted_capture_candidate_score__",
    "__regime_source_clean_economic_capture_candidate_score__",
    "__regime_source_misleading_location_risk_score__",
    "__regime_source_trend_following_score__",
    "__regime_source_mean_reversion_score__",
    "__regime_source_vol_compression_score__",
    "__regime_source_breakout_impulse_score__",
    "__regime_source_dirty_avoid_score__",
)
TRANSITION_STEMS = (
    "range_24h_pct",
    "__meta_raw__volatility_zscore",
    "trend_r2_24",
    "jump_intensity",
    "__meta_raw__chop_score",
    "__regime_source_shock_impulse_score__",
    "__regime_source_compression_score__",
    "__regime_source_dirty_shock_avoid_score__",
    "__regime_source_loud_breakout_impulse_score__",
)
FORBIDDEN_TOKENS = (
    "target",
    "label",
    "outcome",
    "future",
    "path",
    "exit",
    "mfe",
    "mae",
    "first_touch",
    "realized",
    "execution_net",
    "execution_gross",
    "cost_return",
    "__y",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, IDENTITY].copy()
    ordered["__ts__"] = pd.to_datetime(ordered["__ts__"], utc=True).astype(str)
    ordered["__decision_ts__"] = pd.to_datetime(ordered["__decision_ts__"], utc=True).astype(str)
    ordered = ordered.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def forbidden_feature_names(columns: tuple[str, ...] | list[str]) -> list[str]:
    return [name for name in columns if any(token in name.lower() for token in FORBIDDEN_TOKENS)]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["candidate_id"] = output["candidate_id"].astype(str)
    output["side_name"] = output["side_name"].astype(str).str.lower()
    output["__symbol__"] = output["__symbol__"].astype(str)
    for column in ("__ts__", "__signal_ts__", "__decision_ts__"):
        if column in output:
            output[column] = pd.to_datetime(output[column], utc=True, errors="raise")
    return output


def add_causal_transition_deltas(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add exact-gap past-only deltas per side/symbol on archived source rows."""

    result = frame.sort_values(["side_name", "__symbol__", "__ts__"], kind="stable").copy()
    key_columns = ["side_name", "__symbol__", "__ts__"]
    source_index = pd.MultiIndex.from_frame(result.loc[:, key_columns])
    if not source_index.is_unique:
        raise ValueError("source side/symbol/timestamp keys are not unique")
    generated: list[str] = []
    for stem in TRANSITION_STEMS:
        for hours in (3, 12):
            previous = result.loc[:, ["side_name", "__symbol__", "__ts__"]].copy()
            previous["__ts__"] = previous["__ts__"] - pd.Timedelta(hours=hours)
            previous_index = pd.MultiIndex.from_frame(previous)
            lagged = pd.Series(
                pd.to_numeric(result[stem], errors="coerce").to_numpy(), index=source_index
            ).reindex(previous_index).to_numpy()
            column = f"preentry_transition__{stem.strip('_')}__delta_{hours}h"
            result[column] = pd.to_numeric(result[stem], errors="coerce") - lagged
            generated.append(column)
    return result, tuple(generated)


def feature_quality(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in features:
        # Explicit float conversion keeps Boolean decision-time indicators
        # compatible with quantiles (NumPy does not define Boolean
        # subtraction inside percentile interpolation).
        values = pd.to_numeric(frame[name], errors="coerce").astype(float)
        finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        finite_values = values.loc[finite]
        rows.append(
            {
                "feature": name,
                "rows": int(len(values)),
                "missing_count": int(values.isna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "finite_fraction": float(finite.mean()),
                "minimum": float(finite_values.min()) if len(finite_values) else np.nan,
                "p01": float(finite_values.quantile(0.01)) if len(finite_values) else np.nan,
                "median": float(finite_values.median()) if len(finite_values) else np.nan,
                "p99": float(finite_values.quantile(0.99)) if len(finite_values) else np.nan,
                "maximum": float(finite_values.max()) if len(finite_values) else np.nan,
                "mean": float(finite_values.mean()) if len(finite_values) else np.nan,
                "std": float(finite_values.std()) if len(finite_values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validate_panel(panel: pd.DataFrame, features: tuple[str, ...]) -> None:
    if forbidden_feature_names(features):
        raise ValueError(f"forbidden feature names: {forbidden_feature_names(features)}")
    if "spread_proxy_abs_return_bps_robust_z" in features:
        raise ValueError("pathological spread proxy must not be materialized")
    if len(panel) != 140_682:
        raise ValueError(f"expected exactly 140,682 strict OOF identities, found {len(panel)}")
    if panel.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("panel identity is not one-to-one")
    if set(panel["side_name"].unique()) != {"long", "short"}:
        raise ValueError("panel must contain both canonical sides")
    months = set(panel["__ts__"].dt.strftime("%Y-%m"))
    if months != {"2025-03", "2025-04"}:
        raise ValueError(f"wrong strict OOF months: {months}")
    if not panel["__signal_ts__"].eq(panel["__ts__"]).all():
        raise ValueError("signal timestamp differs from source timestamp")
    if not panel["__decision_ts__"].eq(panel["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("decision timestamp is not signal timestamp + 1 hour")
    values = panel.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("non-finite infinity in materialized features")
    volatility = pd.to_numeric(panel["__meta_raw__volatility_zscore"], errors="coerce")
    if volatility.notna().mean() < 0.99 or abs(float(volatility.quantile(0.99))) > 50.0:
        raise ValueError("volatility z-score fails sane-scale contract")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    static_features = tuple(dict.fromkeys(CORE_FEATURES + REGIME_SOURCE_FEATURES))
    forbidden = forbidden_feature_names(static_features)
    if forbidden:
        raise ValueError(f"configured forbidden features: {forbidden}")

    residual = normalize(pd.read_parquet(RESIDUAL, columns=[*IDENTITY, "residual_is_oof"]))
    strict = residual.loc[residual["residual_is_oof"].astype(bool), list(IDENTITY)].copy()
    if (
        len(strict) != 140_682
        or strict.duplicated(list(IDENTITY), keep=False).any()
        or strict["candidate_id"].duplicated().any()
    ):
        raise ValueError("strict residual identity contract fails")

    source_frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {str(RESIDUAL): sha256(RESIDUAL)}
    source_columns = [*IDENTITY, "__signal_ts__", *static_features]
    for month in (2, 3, 4):
        for side in ("long", "short"):
            path = LABEL_ROOT / f"train_global_{side}_5_2025_{month:02d}.parquet"
            source_hashes[str(path)] = sha256(path)
            available = set(pd.read_parquet(path, columns=None).columns)
            missing = sorted(set(source_columns).difference(available))
            if missing:
                raise ValueError(f"{path.name} lacks required source fields: {missing}")
            source = normalize(pd.read_parquet(path, columns=source_columns))
            if not source["side_name"].eq(side).all():
                raise ValueError(f"{path.name} contains unexpected side")
            if source.duplicated(list(IDENTITY), keep=False).any():
                raise ValueError(f"{path.name} has duplicated source identities")
            source_frames.append(source)
    source = pd.concat(source_frames, ignore_index=True)
    source, transition_features = add_causal_transition_deltas(source)
    all_features = tuple(dict.fromkeys(static_features + transition_features))

    source_context = source.loc[:, [*IDENTITY, "__signal_ts__", *all_features]].rename(
        columns={
            "side_name": "__source_side_name__",
            "__symbol__": "__source_symbol__",
            "__ts__": "__source_ts__",
            "__decision_ts__": "__source_decision_ts__",
        }
    )
    if source_context["candidate_id"].duplicated().any():
        raise ValueError("archival source candidate_id is not globally unique")
    panel = strict.merge(
        source_context,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not panel["_merge"].eq("both").all():
        raise ValueError("strict residual identity has missing archival signal-time context")
    symbol_match = panel["__symbol__"].str.replace("_", "/", regex=False).eq(panel["__source_symbol__"])
    identity_match = (
        panel["side_name"].eq(panel["__source_side_name__"])
        & panel["__ts__"].eq(panel["__source_ts__"])
        & panel["__decision_ts__"].eq(panel["__source_decision_ts__"])
        & symbol_match
    )
    if not identity_match.all():
        raise ValueError("candidate-id source identity disagrees with strict residual identity")
    panel = panel.drop(
        columns=["_merge", "__source_side_name__", "__source_symbol__", "__source_ts__", "__source_decision_ts__"]
    ).sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    validate_panel(panel, all_features)

    coverage = (
        panel.assign(month=panel["__ts__"].dt.strftime("%Y-%m"))
        .groupby(["month", "side_name"], observed=True, sort=True)
        .agg(rows=("candidate_id", "size"), symbols=("__symbol__", "nunique"))
        .reset_index()
    )
    quality = feature_quality(panel, all_features)
    temp = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=f".{OUT.name}."))
    panel.to_parquet(temp / "panel.parquet", index=False, compression="zstd")
    coverage.to_parquet(temp / "coverage_by_side_month.parquet", index=False, compression="zstd")
    quality.to_parquet(temp / "feature_quality.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "febapr2025_strict_residual_gross_regime_context_v1",
        "status": "IMMUTABLE_PREENTRY_ONLY_INPUT_PANEL",
        "rows": int(len(panel)),
        "identity_sha256": identity_sha256(panel),
        "identity_columns": list(IDENTITY),
        "feature_columns": list(all_features),
        "feature_groups": {
            "core_gross_opportunity": list(CORE_FEATURES),
            "archived_regime_source_composites": list(REGIME_SOURCE_FEATURES),
            "past_only_transition_deltas": list(transition_features),
        },
        "source_hashes_sha256": source_hashes,
        "causality": {
            "source": "archived signal-time/pre-entry fields only",
            "signal_timestamp_equals_source_timestamp": True,
            "decision_timestamp": "source timestamp + 1 hour",
            "transition_deltas": "same-side/symbol values at t minus exact-gap values at t-3h or t-12h",
            "no_april_outcomes_read_or_inspected": True,
        },
        "exclusions": {
            "forbidden_name_tokens": list(FORBIDDEN_TOKENS),
            "pathological_spread_proxy": "spread_proxy_abs_return_bps_robust_z excluded; no repaired train-only transform is supplied here",
            "post_entry_fields": "all target/path/exit/future/outcome/MFE/MAE fields excluded",
            "constant_archived_regime_flags": [
                "__regime_vol_12h__",
                "__regime_vol_48h__",
                "__regime_volume_12h__",
                "__regime_volume_48h__",
                "__regime_trend_12h__",
                "__regime_trend_48h__",
            ],
        },
        "validation": {
            "one_to_one_identity": True,
            "strict_side_month_coverage": coverage.to_dict(orient="records"),
            "no_infinite_features": True,
            "volatility_zscore_sane_scale": True,
        },
        "outputs_sha256": {path.name: sha256(path) for path in sorted(temp.glob("*.parquet"))},
        "checksum_convention": "All parquet outputs are SHA256-listed; manifest.json is verified by detached manifest.sha256.",
    }
    (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (temp / "manifest.sha256").write_text(f"{sha256(temp / 'manifest.json')}  manifest.json\n")
    os.replace(temp, OUT)
    print(json.dumps({"output": str(OUT), "rows": len(panel), "features": len(all_features)}, sort_keys=True))


if __name__ == "__main__":
    main()
