#!/usr/bin/env python3
"""Diagnostic-only oracle and incremental-potency audit for causal S/R heads.

This runner intentionally does *not* fit or deploy a new trade model.  It
answers the prerequisite question for further S/R head work: whether a perfect
view of the next, policy-relevant S/R interaction could add value beyond the
existing E2/H4 state.  Its ``sr_oracle_*`` fields are deliberately non-causal
and are exported only for offline ceiling tests.  They must never be consumed
by inference, calibration, feature selection, or a causal replay.

The audit also reports accepted-break tail calibration and the incremental
information in existing OOF S/R predictions after residualising against the
H4 feature contract.  No network, exchange, live-state, or execution code is
touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_causal_sr_heads as heads
from scripts import run_causal_sr_continuation_ablation as continuation
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as h4_panel
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4


ENGINE_ROOT = ROOT / "data_perp/artifacts/causal_sr_engine_2025_train_2026_score_20260830_v1"
HEADS_ROOT = ROOT / "data_perp/artifacts/causal_sr_heads_oof_20260830_v3_entrypivotfix"
FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2"
ENTRY_SELECTION = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_agreement_ablation_20260830_v1/E2_q50_agreement_selection_target_free.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_oracle_audit_20260830_v1"
HORIZON = pd.Timedelta(hours=12)
TAILS = (0.90, 0.95, 0.98)

# Causal state snapshots that are useful as a raw-geometry comparator.  The
# oracle targets are held in a separate family and are never mixed with these.
RAW_ZONE_FIELDS = (
    "distance_atr", "highest_timeframe", "independent_confluence_count",
    "raw_candidate_count", "zone_width_atr", "level_age_hours",
    "time_since_last_touch_hours", "historical_ESS",
    "shrunk_historical_strength", "median_reaction_MFE_atr",
    "median_penetration_MAE_atr", "accepted_break_rate",
    "reaction_strength_slope", "penetration_depth_slope",
    "source_swing_1h", "source_swing_4h", "source_rolling_extreme",
    "source_prior_day", "source_prior_week", "source_vwap",
    "source_range_boundary", "source_role_reversal",
)
RAW_FEATURES = tuple(f"sr_raw_{side}_{field}" for side in ("support", "resistance") for field in RAW_ZONE_FIELDS)

ORACLE_SIDE_FIELDS = (
    "reaction_strength", "accepted_break", "reaction_mfe_atr",
    "penetration_mae_atr", "label_available_ts",
)
ORACLE_FEATURES = tuple(f"sr_oracle_{side}_{field}" for side in ("support", "resistance") for field in ORACLE_SIDE_FIELDS[:-1]) + (
    "sr_oracle_long_structure_balance",
    "sr_oracle_any_interaction",
)

TARGET_SPECS = (
    ("support_reaction", "sr_oracle_support_reaction_strength", "sr_support_conditional_strength"),
    ("resistance_reaction", "sr_oracle_resistance_reaction_strength", "sr_resistance_conditional_strength"),
    ("support_break", "sr_oracle_support_accepted_break", "sr_support_accepted_break_probability"),
    ("resistance_break", "sr_oracle_resistance_accepted_break", "sr_resistance_accepted_break_probability"),
    ("support_mfe", "sr_oracle_support_reaction_mfe_atr", "sr_support_reaction_magnitude_q50"),
    ("resistance_mfe", "sr_oracle_resistance_reaction_mfe_atr", "sr_resistance_reaction_magnitude_q50"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_entry_ts(values: pd.Series) -> pd.Series:
    """Candidate IDs are canonical ``symbol|long|decision-ts`` identities."""
    token = values.astype(str).str.rsplit("|", n=1).str[-1]
    result = pd.to_datetime(token, utc=True, errors="coerce")
    if result.isna().any():
        raise AssertionError("S/R snapshot has an unparsable candidate decision identity")
    return result


def _snapshot_base(snapshots: pd.DataFrame) -> pd.DataFrame:
    value = snapshots.copy().reset_index(drop=True)
    value["snapshot_ts"] = pd.to_datetime(value.snapshot_ts, utc=True, errors="raise")
    value["__entry_ts"] = _candidate_entry_ts(value.candidate_id)
    value["__oracle_deadline"] = value["__entry_ts"] + HORIZON
    value["__snapshot_row_id"] = np.arange(len(value), dtype=np.int64)
    # Nullable state bars identify entry snapshots.  Preserve the original
    # nullable field in output and use a private sentinel only for joins.
    value["__state_key"] = pd.to_numeric(value.state_bar_15m, errors="coerce").fillna(-1).astype("int16")
    return value


def _first_resolved_interaction(
    base: pd.DataFrame,
    events: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    """Find the first strictly future, fully H12-resolved interaction per level.

    This is deliberately an *oracle* lookup.  The strict deadline makes its
    content policy-relevant: an event is admitted only when its entire 8-hour
    S/R label is known before that candidate's 12-hour parent-policy timeout.
    Missing means there was no relevant fully-resolved interaction, not a zero
    reaction and not an imputed future label.
    """
    zone_col = f"{side}_zone_id"
    available_col = f"{side}_available"
    selected = base.loc[base[available_col].fillna(False) & base[zone_col].notna(), [
        "__snapshot_row_id", "__symbol__", "snapshot_ts", "__oracle_deadline", zone_col,
    ]].copy()
    selected = selected.rename(columns={zone_col: "zone_id"})
    if selected.empty:
        return pd.DataFrame(columns=["__snapshot_row_id", *ORACLE_SIDE_FIELDS])
    event_columns = [
        "__symbol__", "zone_id", "event_ts", "label_available_ts",
        "y_reaction_strength", "y_accepted_break", "reaction_MFE_atr", "penetration_MAE_atr",
    ]
    right = events.loc[:, event_columns].copy()
    right["event_ts"] = pd.to_datetime(right.event_ts, utc=True, errors="raise")
    right["label_available_ts"] = pd.to_datetime(right.label_available_ts, utc=True, errors="raise")
    # Zone IDs contain the symbol and side, but retain symbol in the key to
    # make the identity invariant explicit and auditable.
    grouped: dict[tuple[str, str], pd.DataFrame] = {
        (str(symbol), str(zone)): frame.sort_values("event_ts", kind="stable").reset_index(drop=True)
        for (symbol, zone), frame in right.groupby(["__symbol__", "zone_id"], sort=False)
    }
    records: list[dict[str, object]] = []
    for (symbol, zone), rows in selected.groupby(["__symbol__", "zone_id"], sort=False):
        event_rows = grouped.get((str(symbol), str(zone)))
        if event_rows is None or event_rows.empty:
            continue
        event_ns = event_rows.event_ts.astype("int64").to_numpy()
        snapshot_ns = rows.snapshot_ts.astype("int64").to_numpy()
        locations = np.searchsorted(event_ns, snapshot_ns, side="right")
        for (_, snap), position in zip(rows.iterrows(), locations, strict=True):
            if int(position) >= len(event_rows):
                continue
            event = event_rows.iloc[int(position)]
            if pd.Timestamp(event.label_available_ts) > pd.Timestamp(snap["__oracle_deadline"]):
                continue
            records.append({
                "__snapshot_row_id": int(snap["__snapshot_row_id"]),
                "reaction_strength": float(event.y_reaction_strength),
                "accepted_break": float(event.y_accepted_break),
                "reaction_mfe_atr": float(event.reaction_MFE_atr),
                "penetration_mae_atr": float(event.penetration_MAE_atr),
                "label_available_ts": pd.Timestamp(event.label_available_ts),
            })
    return pd.DataFrame(records, columns=["__snapshot_row_id", *ORACLE_SIDE_FIELDS])


def _materialize_oracle_labels(snapshots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    base = _snapshot_base(snapshots)
    output = base.loc[:, [
        "__snapshot_row_id", "candidate_id", "snapshot_ts", "target_kind", "target_id", "state_bar_15m", "__state_key", "__entry_ts",
    ]].copy()
    for side in ("support", "resistance"):
        # Distance uses a single underscore in the snapshot schema; every
        # other raw zone feature is namespaced as ``support__*`` / ``resistance__*``.
        source_fields = {
            field: (f"{side}_{field}" if field == "distance_atr" else f"{side}__{field}")
            for field in RAW_ZONE_FIELDS
        }
        present = {field: source for field, source in source_fields.items() if source in base.columns}
        raw = base.loc[:, ["__snapshot_row_id", *present.values()]].copy()
        raw = raw.rename(columns={source: f"sr_raw_{side}_{field}" for field, source in present.items()})
        output = output.merge(raw, on="__snapshot_row_id", how="left", validate="one_to_one")
        found = _first_resolved_interaction(base, events, side)
        found = found.rename(columns={field: f"sr_oracle_{side}_{field}" for field in ORACLE_SIDE_FIELDS})
        output = output.merge(found, on="__snapshot_row_id", how="left", validate="one_to_one")
        output[f"sr_oracle_{side}_available"] = output[f"sr_oracle_{side}_label_available_ts"].notna()
    support_react = pd.to_numeric(output.sr_oracle_support_reaction_strength, errors="coerce").fillna(0.0)
    resistance_break = pd.to_numeric(output.sr_oracle_resistance_accepted_break, errors="coerce").fillna(0.0)
    support_break = pd.to_numeric(output.sr_oracle_support_accepted_break, errors="coerce").fillna(0.0)
    resistance_react = pd.to_numeric(output.sr_oracle_resistance_reaction_strength, errors="coerce").fillna(0.0)
    output["sr_oracle_long_structure_balance"] = support_react + resistance_break - support_break - resistance_react
    output["sr_oracle_any_interaction"] = (output.sr_oracle_support_available | output.sr_oracle_resistance_available).astype(float)
    output["sr_oracle_label_available_ts"] = pd.concat([
        pd.to_datetime(output.sr_oracle_support_label_available_ts, utc=True),
        pd.to_datetime(output.sr_oracle_resistance_label_available_ts, utc=True),
    ], axis=1).max(axis=1)
    if output.duplicated(["candidate_id", "target_kind", "target_id", "__state_key", "snapshot_ts"]).any():
        raise AssertionError("oracle snapshot identity is not unique")
    return output


def _tail_potency(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tail lift/calibration; magnitude is deliberately compared with MFE, not strength."""
    specs = (
        ("sr_prior_strength", "y_reaction_strength", "continuous"),
        ("sr_conditional_strength", "y_reaction_strength", "continuous"),
        ("sr_accepted_break_probability", "y_accepted_break", "binary"),
        ("sr_reaction_magnitude_q50", "reaction_MFE_atr", "continuous"),
    )
    result: list[dict[str, object]] = []
    calibration: list[dict[str, object]] = []
    events = events.copy()
    events["event_ts"] = pd.to_datetime(events.event_ts, utc=True, errors="raise")
    events["held_month"] = events.event_ts.dt.strftime("%Y-%m")
    for held, month in events.groupby("held_month", sort=True):
        for head, target, kind in specs:
            pred = pd.to_numeric(month[head], errors="coerce")
            actual = pd.to_numeric(month[target], errors="coerce")
            valid = pred.notna() & actual.notna()
            if int(valid.sum()) < 100:
                continue
            p, y = pred.loc[valid], actual.loc[valid]
            base = float(y.mean())
            for q in TAILS:
                threshold = float(p.quantile(q, interpolation="higher"))
                tail = y.loc[p.ge(threshold)]
                tail_mean = float(tail.mean())
                result.append({
                    "held_month": held, "head": head, "target": target, "target_kind": kind,
                    "percentile": q, "threshold": threshold, "rows": int(len(tail)),
                    "base_mean": base, "tail_mean": tail_mean,
                    "absolute_lift": tail_mean - base,
                    "relative_lift": tail_mean / base if abs(base) > 1e-12 else np.nan,
                })
            quantiles = p.quantile([0.0, .50, .80, .90, .95, .98, 1.0]).to_numpy(float)
            for lower, upper, label in zip(quantiles[:-1], quantiles[1:], ("p00_50", "p50_80", "p80_90", "p90_95", "p95_98", "p98_100"), strict=True):
                subset = y.loc[p.ge(lower) & (p.le(upper) if label == "p98_100" else p.lt(upper))]
                if subset.empty:
                    continue
                calibration.append({
                    "held_month": held, "head": head, "target": target, "target_kind": kind,
                    "bin": label, "rows": int(len(subset)), "prediction_mean": float(p.loc[subset.index].mean()),
                    "actual_mean": float(subset.mean()), "actual_median": float(subset.median()),
                })
    return pd.DataFrame(result), pd.DataFrame(calibration)


def _model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=180, learning_rate=.035,
        max_depth=3, num_leaves=7, min_child_samples=120, subsample=.80,
        colsample_bytree=.85, reg_lambda=12.0, random_state=1729, n_jobs=2, verbosity=-1,
    )


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...], target: str) -> np.ndarray:
    if not features:
        return np.full(len(test), np.nan)
    x_train = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    x_test = test.loc[:, features].apply(pd.to_numeric, errors="coerce")
    model = _model()
    model.fit(x_train, pd.to_numeric(train[target], errors="raise"))
    return model.predict(x_test)


def _conditional_information(oracle: pd.DataFrame, heads_wide: pd.DataFrame) -> pd.DataFrame:
    """Test whether OOF S/R predictions retain information beyond H4 features."""
    panel = h4_panel._load_panel(h4_panel.TARGET_PANEL, h4_panel.VWAP_PANEL)
    panel = panel.copy()
    panel["snapshot_ts"] = pd.to_datetime(panel.state_decision_ts, utc=True, errors="raise")
    panel["__state_key"] = pd.to_numeric(panel.state_bar_15m, errors="raise").astype("int16")
    selected = set(pd.read_parquet(ENTRY_SELECTION, columns=["candidate_id"]).candidate_id.astype(str))
    oracle_cont = oracle.loc[oracle.target_kind.eq("continuation")].copy()
    head_cont = heads_wide.loc[heads_wide.target_kind.eq("continuation")].copy()
    for frame in (oracle_cont, head_cont):
        frame["snapshot_ts"] = pd.to_datetime(frame.snapshot_ts, utc=True, errors="raise")
        frame["__state_key"] = pd.to_numeric(frame.state_bar_15m, errors="coerce").fillna(-1).astype("int16")
    oracle_cols = ["candidate_id", "snapshot_ts", "__state_key", *RAW_FEATURES, *ORACLE_FEATURES, "sr_oracle_label_available_ts"]
    oracle_cols = [item for item in oracle_cols if item in oracle_cont]
    head_cols = ["candidate_id", "snapshot_ts", "__state_key", *(item[2] for item in TARGET_SPECS)]
    head_cols = [item for item in head_cols if item in head_cont]
    panel = panel.merge(oracle_cont.loc[:, oracle_cols], on=["candidate_id", "snapshot_ts", "__state_key"], how="left", validate="one_to_one")
    panel = panel.merge(head_cont.loc[:, head_cols], on=["candidate_id", "snapshot_ts", "__state_key"], how="left", validate="one_to_one")
    panel["entry_decision_ts"] = pd.to_datetime(panel.entry_decision_ts, utc=True, errors="raise")
    panel["policy_label_available_ts"] = pd.to_datetime(panel.policy_label_available_ts, utc=True, errors="raise")
    held_months = tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    feature_map = h4._features_by_month(FEATURE_STUDY / "stable_selected_features.parquet", "C4_normalized_vwap_fs", held_months)
    rows: list[dict[str, object]] = []
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=4)
        base = tuple(feature_map[held])
        for name, target, head in TARGET_SPECS:
            all_valid = panel.loc[pd.to_numeric(panel[target], errors="coerce").notna()].copy()
            all_valid["sr_oracle_label_available_ts"] = pd.to_datetime(all_valid.sr_oracle_label_available_ts, utc=True, errors="coerce")
            train = all_valid.loc[
                pd.to_numeric(all_valid.MC1_expected_bps, errors="coerce").ge(30.0)
                & all_valid.entry_decision_ts.ge(start) & all_valid.entry_decision_ts.lt(held)
                & all_valid.sr_oracle_label_available_ts.lt(held)
            ].copy()
            test = all_valid.loc[
                all_valid.entry_decision_ts.ge(held) & all_valid.entry_decision_ts.lt(end)
                & all_valid.candidate_id.astype(str).isin(selected)
            ].copy()
            if len(train) < 500 or len(test) < 100 or head not in train:
                continue
            raw = tuple(item for item in RAW_FEATURES if item in train and item in test)
            feature_sets = {
                "h4_only": base,
                "h4_plus_raw_sr": tuple((*base, *raw)),
                "h4_plus_oof_sr_head": tuple((*base, head)),
            }
            base_pred = _fit_predict(train, test, base, target)
            y = pd.to_numeric(test[target], errors="raise").to_numpy(float)
            residual = y - base_pred
            head_values = pd.to_numeric(test[head], errors="coerce").to_numpy(float)
            for model_name, features in feature_sets.items():
                prediction = _fit_predict(train, test, features, target)
                valid = np.isfinite(prediction) & np.isfinite(y)
                rows.append({
                    "held_month": held.strftime("%Y-%m"), "target_name": name, "target": target,
                    "model": model_name, "rows": int(valid.sum()),
                    "spearman": float(pd.Series(y[valid]).corr(pd.Series(prediction[valid]), method="spearman")),
                    "mae": float(mean_absolute_error(y[valid], prediction[valid])),
                })
            valid = np.isfinite(residual) & np.isfinite(head_values)
            rows.append({
                "held_month": held.strftime("%Y-%m"), "target_name": name, "target": target,
                "model": "oof_head_residual_ic", "rows": int(valid.sum()),
                "spearman": float(pd.Series(residual[valid]).corr(pd.Series(head_values[valid]), method="spearman")),
                "mae": np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, default=ENGINE_ROOT)
    parser.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-conditional-audit", action="store_true")
    args = parser.parse_args()
    engine_root, heads_root, output = args.engine_root.resolve(), args.heads_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    snapshots = pd.read_parquet(engine_root / "sr_snapshots.parquet")
    events = pd.read_parquet(engine_root / "interaction_events.parquet")
    oracle = _materialize_oracle_labels(snapshots, events)
    event_predictions = pd.read_parquet(heads_root / "interaction_head_oof_predictions.parquet")
    snapshot_predictions = pd.read_parquet(heads_root / "snapshot_head_oof_features.parquet")
    tail, calibration = _tail_potency(event_predictions)
    conditional = pd.DataFrame() if args.skip_conditional_audit else _conditional_information(oracle, snapshot_predictions)
    output.mkdir(parents=True, exist_ok=False)
    oracle.to_parquet(output / "snapshot_oracle_labels_NONCAUSAL_DIAGNOSTIC_ONLY.parquet", index=False, compression="zstd")
    oracle.loc[oracle.target_kind.eq("entry")].to_parquet(output / "entry_oracle_labels_NONCAUSAL_DIAGNOSTIC_ONLY.parquet", index=False, compression="zstd")
    oracle.loc[oracle.target_kind.eq("continuation")].to_parquet(output / "continuation_oracle_labels_NONCAUSAL_DIAGNOSTIC_ONLY.parquet", index=False, compression="zstd")
    tail.to_parquet(output / "head_tail_potency_by_month.parquet", index=False)
    calibration.to_parquet(output / "head_tail_calibration_by_month.parquet", index=False)
    conditional.to_parquet(output / "conditional_information_h4_by_month.parquet", index=False)
    coverage = oracle.groupby("target_kind", as_index=False).agg(
        snapshots=("candidate_id", "size"), candidates=("candidate_id", "nunique"),
        any_interaction=("sr_oracle_any_interaction", "sum"),
        support_interaction=("sr_oracle_support_available", "sum"),
        resistance_interaction=("sr_oracle_resistance_available", "sum"),
    )
    coverage.to_parquet(output / "oracle_label_coverage.parquet", index=False)
    manifest = {
        "schema": "causal-sr-oracle-audit-v1",
        "scope": "offline diagnostic only; no live/canonical mutation, exchange IO, or order submission",
        "engine_root": str(engine_root), "engine_manifest_sha256": _sha256(engine_root / "run_manifest.json"),
        "heads_root": str(heads_root), "heads_manifest_sha256": _sha256(heads_root / "run_manifest.json"),
        "oracle_contract": "first strictly future interaction for the snapshot's frozen zone whose complete 8h S/R label resolves by the candidate H12 timeout; missing is retained as missing/neutral only for diagnostic models",
        "noncausal_fields": list(ORACLE_FEATURES),
        "raw_geometry_fields": list(RAW_FEATURES),
        "tail_contract": "held-month OOF predictions; p90/p95/p98 lift and calibration, with reaction-magnitude evaluated against reaction_MFE_atr",
        "conditional_contract": "June-August held months; four-month strictly prior fit for H4 feature residualisation; current S/R predictions are OOF, while oracle labels are deliberately noncausal diagnostics",
        "causality_warning": "oracle labels are forbidden from causal inference, model selection, calibration, or live execution",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
