#!/usr/bin/env python3
"""Frozen forward replay for interpretable residual-overlay artifacts.

This script intentionally has no model fitting, feature screening, threshold
search, or outcome-derived transform.  It joins a scored parent stream to a
pre-materialized observable state artifact, applies each serialized local
bundle, and evaluates the already-frozen side x archetype overlay policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts import run_meta_residual_event_balanced_error_overlay as base


RISK_SCORE = "interpretable_rule_risk_score"
RISK_PCT = "interpretable_rule_risk_percentile"
ADJUSTED_RANK = "parent_rank_v9_interpretable_rule_overlay"
FLAGGED = "interpretable_rule_overlay_flagged"


def _forward_breakdown(frame: pd.DataFrame, selector: str, rank: np.ndarray) -> pd.DataFrame:
    """Outcome-only forward diagnostics on an identical resolved-row universe."""
    selected = frame.loc[rank >= 0.90].copy()
    selected["day"] = selected["__ts__"].dt.strftime("%Y-%m-%d")
    day = selected["__ts__"].dt.floor("D")
    selected["week_start"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
    selected["month"] = selected["__ts__"].dt.strftime("%Y-%m")
    reports: list[pd.DataFrame] = []
    aggregations: dict[str, tuple[str, Any]] = {
        "selected_rows": ("ev_after_1pct", "size"),
        "mean_ev_after_1pct": ("ev_after_1pct", "mean"),
        "sum_ev_after_1pct": ("ev_after_1pct", "sum"),
        "positive_ev_rate": ("ev_after_1pct", lambda values: float((values > 0.0).mean())),
        "clean_exec_precision": ("clean_exec", "mean"),
        "dirty_positive_rate": ("dirty_positive", "mean"),
        "first_touch_bad_mae_rate": ("first_touch_bad_mae_1r", "mean"),
        "full_path_bad_mae_rate": ("full_path_bad_mae_1r", "mean"),
        "timeout_rate": ("timeout", "mean"),
    }
    for scope, groups in (
        ("overall", []),
        ("day", ["day"]),
        ("week", ["week_start"]),
        ("month", ["month"]),
        ("side", ["side_name"]),
        ("archetype", ["archetype_policy_key"]),
        ("day_side_archetype", ["day", "side_name", "archetype_policy_key"]),
    ):
        if groups:
            report = selected.groupby(groups, observed=True, dropna=False).agg(**aggregations).reset_index()
        else:
            # DataFrame.aggregate with named aggregations returns a one-row
            # frame on recent pandas versions, unlike GroupBy.aggregate.
            # Build the global row explicitly so the forward artifact has the
            # same schema across pandas versions.
            report = pd.DataFrame(
                [{
                    "selected_rows": int(len(selected)),
                    "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
                    "sum_ev_after_1pct": float(selected["ev_after_1pct"].sum()),
                    "positive_ev_rate": float((selected["ev_after_1pct"] > 0.0).mean()),
                    "clean_exec_precision": float(selected["clean_exec"].mean()),
                    "dirty_positive_rate": float(selected["dirty_positive"].mean()),
                    "first_touch_bad_mae_rate": float(selected["first_touch_bad_mae_1r"].mean()),
                    "full_path_bad_mae_rate": float(selected["full_path_bad_mae_1r"].mean()),
                    "timeout_rate": float(selected["timeout"].mean()),
                }]
            )
        report["scope"] = scope
        report["selector"] = selector
        reports.append(report)
    return pd.concat(reports, ignore_index=True, sort=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _hash_strings(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bundle_path(overlay: Path, row: dict[str, Any]) -> Path:
    name = "model__{arm}__{side}__{archetype}.joblib".format(
        arm=str(row["model_arm"]),
        side=str(row["side_name"]),
        archetype=str(row["archetype_policy_key"]),
    )
    path = overlay / name
    if not path.exists():
        raise FileNotFoundError(f"Frozen bundle is missing: {path}")
    return path


def _accepted_rows(overlay: Path) -> list[dict[str, Any]]:
    path = overlay / "accepted_overlays.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _load_state_columns(state_path: Path, required: list[str]) -> pd.DataFrame:
    schema = pq.read_schema(state_path)
    missing = sorted(set(required) - set(schema.names))
    if missing:
        raise ValueError(f"State artifact lacks required frozen features: {missing}")
    state = pd.read_parquet(state_path, columns=required)
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
    state["__overlay_state_present__"] = np.int8(1)
    if state.duplicated(base.KEYS).any():
        raise ValueError("State artifact has duplicate forward keys")
    return state


def _score_overlay(
    parent: pd.DataFrame,
    state: pd.DataFrame,
    overlay: Path,
    *,
    output: Path,
) -> dict[str, Any]:
    accepted_rows = _accepted_rows(overlay)
    if not accepted_rows:
        raise ValueError(f"No frozen accepted overlays in {overlay}")
    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    feature_lists: dict[tuple[str, str], list[str]] = {}
    for row in accepted_rows:
        key = (str(row["side_name"]), str(row["archetype_policy_key"]))
        bundle = joblib.load(_bundle_path(overlay, row))
        features = [str(name) for name in bundle["features"]]
        forbidden = {
            base.TARGET,
            base.EVENT,
            base.SIDE_EVENT,
            "clean_exec",
            "dirty_positive",
            "ev_after_1pct",
            "exec_margin",
            "timeout",
            "first_touch_bad_mae_1r",
            "full_path_bad_mae_1r",
            "episode_onset_target",
            "episode_persistent_target",
            "episode_recovery_target",
        }
        leakage = sorted(
            name for name in features
            if name in forbidden
            or name.endswith("_target")
            or "false_positive_target" in name
        )
        if leakage:
            raise ValueError(
                f"Frozen bundle contains outcome-derived inference features: {leakage}"
            )
        bundles[key] = bundle
        feature_lists[key] = features

    required = list(
        dict.fromkeys([*base.KEYS, "__overlay_state_present__", *sum(feature_lists.values(), [])])
    )
    available_state = state.reindex(columns=required)
    joined = parent.merge(
        available_state,
        on=base.KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "__state"),
    )
    state_coverage = float(joined["__overlay_state_present__"].fillna(0).mean())
    if state_coverage < 1.0:
        raise ValueError(f"Frozen state join coverage is incomplete: {state_coverage:.4%}")
    joined.drop(columns="__overlay_state_present__", inplace=True)

    # Parent forward ledgers may already contain sparse, historical versions of
    # a residual feature.  The compact state artifact was generated expressly
    # from the frozen observable feature contract, so it must override those
    # colliding columns.  Leaving the parent copy in place silently turns a
    # fully materialized feature into a sparse inference input.
    for name in dict.fromkeys(sum(feature_lists.values(), [])):
        state_name = f"{name}__state"
        if state_name not in joined.columns:
            continue
        joined[name] = joined[state_name]
        joined.drop(columns=state_name, inplace=True)

    joined[RISK_SCORE] = np.float32(np.nan)
    joined[RISK_PCT] = np.float32(0.5)
    # Period overlays are defined only for the parent policy's top-10 decision
    # population.  Scoring a lower-ranked row would be both unnecessary and
    # unsafe when its timestamp has no top-20 context state in the compact
    # parity ledger.
    parent_top10 = pd.to_numeric(joined["parent_rank_v9"], errors="coerce").ge(0.90)
    scored_rows = 0
    group_contracts: list[dict[str, Any]] = []
    for row in accepted_rows:
        side = str(row["side_name"])
        archetype = str(row["archetype_policy_key"])
        key = (side, archetype)
        mask = (
            joined["side_name"].astype(str).eq(side)
            & joined["archetype_policy_key"].astype(str).eq(archetype)
            & parent_top10
        ).to_numpy()
        if not mask.any():
            continue
        features = feature_lists[key]
        local = joined.loc[mask, features]
        missing_rows = int(local.isna().all(axis=1).sum())
        missing_features = [name for name in features if local[name].notna().sum() == 0]
        if missing_features or missing_rows:
            raise ValueError(
                f"Frozen feature parity failed for {side}|{archetype}: "
                f"missing_features={missing_features}, all_missing_rows={missing_rows}"
            )
        matrix = local.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        bundle = bundles[key]
        transformed = bundle["robust"].transform(matrix)
        score = bundle["model"].predict_proba(transformed)
        percentile = base._midrank(score, np.asarray(bundle["reference"], dtype=np.float32))
        idx = np.flatnonzero(mask)
        joined.loc[idx, RISK_SCORE] = score
        joined.loc[idx, RISK_PCT] = percentile
        scored_rows += int(mask.sum())
        group_contracts.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "rows": int(mask.sum()),
                "feature_count": len(features),
                "feature_schema_hash": _hash_strings(features),
                "all_missing_rows": missing_rows,
                "feature_coverage_min": float(local.notna().mean().min()),
            }
        )

    params: dict[tuple[str, str], dict[str, Any]] = {}
    for row in accepted_rows:
        local = dict(row)
        local["risk_variant"] = RISK_PCT
        params[(str(local["side_name"]), str(local["archetype_policy_key"]))] = local
    adjusted, flagged = base._apply_selected_overlays(joined, params, "parent_rank_v9")
    joined[ADJUSTED_RANK] = adjusted
    joined[FLAGGED] = flagged

    event_labels_available = base.EVENT in joined.columns
    if not event_labels_available:
        # July forward outcomes contain executable EV/path fields but not the
        # hand-curated adverse calendar. Keep event metrics explicitly empty.
        joined[base.EVENT] = np.int8(0)
    # Metrics must compare the same resolved rows. The final timestamps can be
    # present in the parent prediction stream while their execution horizon is
    # still open, so including them would alter only one policy's denominator.
    outcome_mask = pd.to_numeric(joined["ev_after_1pct"], errors="coerce").notna()
    metrics_frame = joined.loc[outcome_mask].copy()
    metrics_adjusted = adjusted[outcome_mask.to_numpy()]
    top10 = 0.90
    parent_metrics = base._selection_metrics(
        metrics_frame,
        metrics_frame["parent_rank_v9"].to_numpy(np.float32),
        top10,
    )
    overlay_metrics = base._selection_metrics(metrics_frame, metrics_adjusted, top10)
    summary = pd.DataFrame([
        {"selector": "parent", **parent_metrics},
        {"selector": "frozen_interpretable_overlay", **overlay_metrics},
    ])
    for metric in ("mean_ev", "positive_ev_rate", "clean_precision", "event_mean_ev", "normal_mean_ev"):
        summary[f"delta_{metric}_vs_parent"] = summary[metric] - parent_metrics[metric]
    output.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output / "forward_predictions.parquet", index=False, compression="zstd")
    summary.to_csv(output / "summary.csv", index=False)
    detailed = pd.concat(
        [
            _forward_breakdown(metrics_frame, "parent", metrics_frame["parent_rank_v9"].to_numpy(np.float32)),
            _forward_breakdown(metrics_frame, "frozen_interpretable_overlay", metrics_adjusted),
        ],
        ignore_index=True,
        sort=False,
    )
    detailed.to_csv(output / "forward_metrics_detailed.csv", index=False)
    manifest = {
        "schema": "frozen_interpretable_residual_overlay_forward_v1",
        "overlay_artifact": str(overlay),
        "parent_rows": int(len(parent)),
        "resolved_outcome_rows": int(len(metrics_frame)),
        "unresolved_outcome_rows_excluded_from_metrics": int((~outcome_mask).sum()),
        "state_rows": int(len(state)),
        "joined_state_coverage": state_coverage,
        "event_labels_available": event_labels_available,
        "scored_overlay_rows": int(scored_rows),
        "accepted_groups": group_contracts,
        "feature_contract": "Only serialized bundle feature lists are read from state; outcome columns are never included in a scorer matrix.",
        "training_contract": "No fitting, feature selection, threshold search, or normalization refit occurs in this forward replay.",
        "threshold_contract": "Accepted OOF thresholds and overlay strengths are loaded unchanged from the source artifact.",
        "metric_contract": "All selector metrics use the same rows with resolved ev_after_1pct; no July outcome is used to fit or select an overlay.",
    }
    (output / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    parent = pd.read_parquet(args.parent)
    parent["__ts__"] = pd.to_datetime(parent["__ts__"], utc=True)
    if parent.duplicated(base.KEYS).any():
        raise ValueError("Parent forward ledger has duplicate keys")
    for overlay in args.overlay:
        accepted_rows = _accepted_rows(overlay)
        if not accepted_rows:
            print(json.dumps({"overlay_artifact": str(overlay), "status": "no_active_overlays"}))
            continue
        bundle_features: list[str] = []
        for row in accepted_rows:
            bundle = joblib.load(_bundle_path(overlay, row))
            bundle_features.extend(str(name) for name in bundle["features"])
        state = _load_state_columns(args.state, list(dict.fromkeys([*base.KEYS, *bundle_features])))
        out = args.output_root / overlay.name
        print(json.dumps(_json_safe(_score_overlay(parent, state, overlay, output=out)), indent=2))


if __name__ == "__main__":
    main()
