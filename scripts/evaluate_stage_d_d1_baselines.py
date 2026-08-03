#!/usr/bin/env python3
"""Seal Stage-D D1 deterministic EXIT_NOW versus CONTINUE baselines only.

This evaluator deliberately has no model, threshold, ranking, entry, sizing, or
portfolio logic.  It compares the two paired D0 v2 counterfactual arms on their
unchanged common candidate population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
COUNTERFACTUALS = ART / "stage_d_action_counterfactuals_20260731_v2/stage_d_action_counterfactuals.parquet"
FEATURE_ROOT = ART / "stage_d_action_features_20260731_v3"
FEATURES = FEATURE_ROOT / "stage_d_action_features.parquet"
FEATURE_MANIFEST = FEATURE_ROOT / "manifest.json"
DEFAULT_OUTPUT = ART / "stage_d_d1_deterministic_baselines_20260731_v4"
SCHEMA = "stage_d_d1_deterministic_baselines_v4"
SEED = 20260731
BOOTSTRAP_REPS = 2_000

TIME_BUCKETS = [-np.inf, 5, 15, 30, 60, 120, 240, 480, np.inf]
TIME_LABELS = ["01-05m", "06-15m", "16-30m", "31-60m", "61-120m", "121-240m", "241-480m", "481-718m"]
VOLUME_BUCKETS = [-np.inf, -1.0, 0.0, 1.0, np.inf]
VOLUME_LABELS = ["z<=-1", "-1<z<=0", "0<z<=1", "z>1"]
VOLATILITY_BUCKETS = [-np.inf, 25.0, 50.0, 100.0, np.inf]
VOLATILITY_LABELS = ["<=25bps", "25-50bps", "50-100bps", ">100bps"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def candidate_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(values.astype(str).tolist()).encode()).hexdigest()


def _required_counterfactual_columns() -> set[str]:
    return {
        "candidate_id", "side", "action_decision_ts", "first_clear_bar_index",
        "net_continue_gross_bps", "net_continue_cost_bps", "net_continue_bps",
        "net_exit_now_gross_bps", "net_exit_now_cost_bps", "net_exit_now_bps",
        "delta_continue_bps",
    }


def attach_causal_buckets(counterfactuals: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Join only causal feature values and create predeclared descriptive buckets."""
    missing = _required_counterfactual_columns().difference(counterfactuals.columns)
    if missing:
        raise ValueError(f"counterfactual columns missing: {sorted(missing)}")
    if counterfactuals.candidate_id.duplicated().any() or features.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique before D1 join")
    if not set(counterfactuals.candidate_id).issubset(set(features.candidate_id)):
        raise ValueError("causal feature pack does not cover the D0 v2 population")
    if "realised_volatility" not in features.columns:
        raise ValueError("D1 requires causal realised_volatility for the complete D0 v2 population")
    if "feature_available_ts" not in features.columns or "action_decision_ts" not in features.columns:
        raise ValueError("D1 requires action-time availability and action-decision timestamps in the feature pack")
    feature_time = pd.to_datetime(features.feature_available_ts, utc=True, errors="coerce")
    feature_decision = pd.to_datetime(features.action_decision_ts, utc=True, errors="coerce")
    if feature_time.isna().any() or feature_decision.isna().any() or not feature_time.le(feature_decision).all():
        raise ValueError("D1 feature availability is not causal at the action decision")
    timing = counterfactuals[["candidate_id", "action_decision_ts"]].merge(
        features[["candidate_id", "action_decision_ts"]], on="candidate_id", validate="one_to_one", suffixes=("_d0", "_feature"),
    )
    if not np.array_equal(pd.to_datetime(timing.action_decision_ts_d0, utc=True).astype(str), pd.to_datetime(timing.action_decision_ts_feature, utc=True).astype(str)):
        raise ValueError("D1 feature action-decision timestamps differ from D0 v2")
    allowed = ["candidate_id"]
    for name in ("volume_z_at_clear", "realised_volatility"):
        if name in features.columns:
            allowed.append(name)
    rows = counterfactuals.merge(features[allowed], on="candidate_id", how="left", validate="one_to_one")
    rows["month"] = pd.to_datetime(rows.action_decision_ts, utc=True).dt.strftime("%Y-%m")
    rows["utc_day"] = pd.to_datetime(rows.action_decision_ts, utc=True).dt.strftime("%Y-%m-%d")
    rows["time_to_clear_minutes"] = rows.first_clear_bar_index.astype(int) + 1
    rows["time_to_clear_bucket"] = pd.cut(rows.time_to_clear_minutes, TIME_BUCKETS, labels=TIME_LABELS, right=True, include_lowest=True).astype(str)
    status = {"regime_bucket": "NOT_REPORTED_A8_REJECTED_OOF_LINEAGE"}
    if "volume_z_at_clear" in rows and rows.volume_z_at_clear.notna().any():
        rows["volume_bucket"] = pd.cut(rows.volume_z_at_clear, VOLUME_BUCKETS, labels=VOLUME_LABELS, right=True, include_lowest=True).astype(str)
        status["volume_bucket"] = "REPORTED_A3_CAUSAL_VOLUME_Z_AT_CLEAR"
    else:
        status["volume_bucket"] = "NOT_REPORTED_A3_SOURCE_FIELD_ABSENT_OR_UNAVAILABLE"
    if not np.isfinite(pd.to_numeric(rows.realised_volatility, errors="coerce")).all():
        raise ValueError("D1 requires finite causal realised_volatility for every D0 v2 candidate")
    if "realised_volatility" in rows:
        rows["volatility_bucket"] = pd.cut(rows.realised_volatility, VOLATILITY_BUCKETS, labels=VOLATILITY_LABELS, right=True, include_lowest=True).astype(str)
        status["volatility_bucket"] = "REPORTED_A4_CAUSAL_REALISED_VOLATILITY_BPS"
    return rows, status


def paired_metrics(rows: pd.DataFrame, group_name: str, group_values: pd.Series | None = None) -> pd.DataFrame:
    """Summarise paired B0/B1 economics without selecting an action."""
    frame = rows.copy()
    frame["__group__"] = "ALL" if group_values is None else group_values.astype(str).to_numpy()
    grouped = frame.groupby("__group__", dropna=False, observed=True)
    out = grouped.agg(
        rows=("candidate_id", "size"),
        continue_gross_sum_bps=("net_continue_gross_bps", "sum"),
        continue_cost_sum_bps=("net_continue_cost_bps", "sum"),
        continue_net_sum_bps=("net_continue_bps", "sum"),
        exit_gross_sum_bps=("net_exit_now_gross_bps", "sum"),
        exit_cost_sum_bps=("net_exit_now_cost_bps", "sum"),
        exit_net_sum_bps=("net_exit_now_bps", "sum"),
    ).reset_index(names="group_value")
    for prefix in ("continue_gross", "continue_cost", "continue_net", "exit_gross", "exit_cost", "exit_net"):
        out[f"{prefix}_mean_bps"] = out[f"{prefix}_sum_bps"] / out.rows
    incremental = frame.net_exit_now_bps - frame.net_continue_bps
    frame["__exit_incremental_bps__"] = incremental
    frame["__loss_avoided_bps__"] = incremental.clip(lower=0.0)
    frame["__false_exit_opportunity_cost_bps__"] = (-incremental).clip(lower=0.0)
    extra = frame.groupby("__group__", dropna=False, observed=True).agg(
        exit_minus_continue_sum_bps=("__exit_incremental_bps__", "sum"),
        exit_better_row_rate=("__exit_incremental_bps__", lambda x: float((x > 0.0).mean())),
        tied_row_rate=("__exit_incremental_bps__", lambda x: float(np.isclose(x, 0.0, atol=1e-6).mean())),
        loss_avoided_sum_bps=("__loss_avoided_bps__", "sum"),
        loss_avoided_mean_bps=("__loss_avoided_bps__", "mean"),
        loss_avoided_p50_bps=("__loss_avoided_bps__", "median"),
        loss_avoided_p90_bps=("__loss_avoided_bps__", lambda x: float(x.quantile(.90))),
        false_exit_opportunity_cost_sum_bps=("__false_exit_opportunity_cost_bps__", "sum"),
        false_exit_opportunity_cost_mean_bps=("__false_exit_opportunity_cost_bps__", "mean"),
        false_exit_opportunity_cost_p50_bps=("__false_exit_opportunity_cost_bps__", "median"),
        false_exit_opportunity_cost_p90_bps=("__false_exit_opportunity_cost_bps__", lambda x: float(x.quantile(.90))),
    ).reset_index(names="group_value")
    out = out.merge(extra, on="group_value", validate="one_to_one")
    out["exit_minus_continue_mean_bps"] = out.exit_minus_continue_sum_bps / out.rows
    out["continue_better_row_rate"] = 1.0 - out.exit_better_row_rate - out.tied_row_rate
    out.insert(0, "group_type", group_name)
    return out.sort_values(["group_type", "group_value"], kind="stable").reset_index(drop=True)


def daily_comparisons(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    day = rows.groupby("utc_day", observed=True).agg(
        rows=("candidate_id", "size"),
        continue_net_sum_bps=("net_continue_bps", "sum"),
        exit_net_sum_bps=("net_exit_now_bps", "sum"),
    ).reset_index()
    day["exit_minus_continue_sum_bps"] = day.exit_net_sum_bps - day.continue_net_sum_bps
    day["exit_minus_continue_mean_bps"] = day.exit_minus_continue_sum_bps / day.rows
    rng = np.random.default_rng(SEED)
    day_rows = day.rows.to_numpy(dtype=np.int64)
    day_delta = day.exit_minus_continue_sum_bps.to_numpy(dtype=float)
    bootstrap = np.empty(BOOTSTRAP_REPS, dtype=float)
    for rep in range(BOOTSTRAP_REPS):
        sampled = rng.integers(0, len(day), size=len(day))
        # Resample whole UTC-day blocks, then recompute the pooled per-trade
        # effect.  Equal-weighted averaging of day means is incorrect when day
        # support differs.
        bootstrap[rep] = day_delta[sampled].sum() / day_rows[sampled].sum()
    pooled = float(day.exit_minus_continue_sum_bps.sum() / day.rows.sum())
    report = {
        "utc_day_blocks": int(len(day)), "bootstrap_reps": BOOTSTRAP_REPS, "seed": SEED,
        "exit_better_day_rate": float((day.exit_minus_continue_sum_bps > 0.0).mean()),
        "exit_minus_continue_equal_weight_day_mean_bps_diagnostic_only": float(day.exit_minus_continue_mean_bps.mean()),
        "exit_minus_continue_pooled_mean_bps": pooled,
        "bootstrap_estimator": "resample whole UTC-day blocks; sum sampled EXIT_MINUS_CONTINUE bps / sum sampled rows",
        "paired_utc_day_block_bootstrap_95pct_ci_bps": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
    }
    return day, report


def evaluate(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return deterministic B0/B1 summaries, paired rows, daily comparisons, and audit facts."""
    if rows.candidate_id.duplicated().any():
        raise ValueError("D1 population candidate IDs are not unique")
    if not np.allclose(rows.net_continue_gross_bps - rows.net_continue_cost_bps, rows.net_continue_bps, atol=1e-6):
        raise ValueError("B0 frozen costs are not applied once")
    if not np.allclose(rows.net_exit_now_gross_bps - rows.net_exit_now_cost_bps, rows.net_exit_now_bps, atol=1e-6):
        raise ValueError("B1 exit costs are not applied once")
    if not np.allclose(rows.delta_continue_bps, rows.net_continue_bps - rows.net_exit_now_bps, atol=1e-6):
        raise ValueError("D0 paired delta contract failed")
    groups = [("overall", None), ("side", rows.side), ("month", rows.month), ("time_to_clear_bucket", rows.time_to_clear_bucket)]
    if "volume_bucket" in rows:
        groups.append(("volume_bucket", rows.volume_bucket))
    if "volatility_bucket" in rows:
        groups.append(("volatility_bucket", rows.volatility_bucket))
    summary = pd.concat([paired_metrics(rows, name, values) for name, values in groups], ignore_index=True)
    paired = rows[["candidate_id", "side", "utc_day", "month", "time_to_clear_minutes", "time_to_clear_bucket", "net_continue_bps", "net_exit_now_bps", "delta_continue_bps"]].copy()
    paired["exit_minus_continue_bps"] = paired.net_exit_now_bps - paired.net_continue_bps
    paired["mechanical_exit_is_better"] = paired.exit_minus_continue_bps > 0.0
    paired["loss_avoided_bps"] = paired.exit_minus_continue_bps.clip(lower=0.0)
    paired["false_exit_opportunity_cost_bps"] = (-paired.exit_minus_continue_bps).clip(lower=0.0)
    daily, daily_report = daily_comparisons(rows)
    overall = summary.loc[summary.group_type.eq("overall")].iloc[0]
    facts = {
        "rows": int(len(rows)), "ordered_candidate_id_sha256": candidate_hash(rows.candidate_id),
        "mechanical_exit_superior_overall": bool(float(overall.exit_minus_continue_mean_bps) > 0.0),
        "baseline_uplift_exit_minus_continue_mean_bps": float(overall.exit_minus_continue_mean_bps),
        "positive_loss_avoided_if_exit": {
            "definition": "max(EXIT_NOW minus CONTINUE, 0)",
            "sum_bps": float(overall.loss_avoided_sum_bps), "unconditional_mean_bps": float(overall.loss_avoided_mean_bps),
            "p50_bps": float(overall.loss_avoided_p50_bps), "p90_bps": float(overall.loss_avoided_p90_bps),
        },
        "false_exit_opportunity_cost": {
            "definition": "max(CONTINUE minus EXIT_NOW, 0)",
            "sum_bps": float(overall.false_exit_opportunity_cost_sum_bps), "unconditional_mean_bps": float(overall.false_exit_opportunity_cost_mean_bps),
            "p50_bps": float(overall.false_exit_opportunity_cost_p50_bps), "p90_bps": float(overall.false_exit_opportunity_cost_p90_bps),
        },
        "candidate_ids_fixed_across_b0_b1": True, "day_comparison": daily_report,
    }
    return summary, paired, daily, facts


def run(*, counterfactuals_path: Path, features_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"fresh output root required: {output}")
    if not features_path.exists():
        raise FileNotFoundError(
            f"D1 waits for the causal Stage-D feature pack with complete A4 realised volatility: {features_path}"
        )
    counter = pd.read_parquet(counterfactuals_path)
    features = pd.read_parquet(features_path)
    rows, bucket_status = attach_causal_buckets(counter, features)
    summary, paired, daily, facts = evaluate(rows)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        summary.to_parquet(stage / "stage_d_d1_baseline_summary.parquet", index=False, compression="zstd")
        paired.to_parquet(stage / "stage_d_d1_paired_row_comparisons.parquet", index=False, compression="zstd")
        daily.to_parquet(stage / "stage_d_d1_paired_utc_day_comparisons.parquet", index=False, compression="zstd")
        report = ["# Stage-D D1 deterministic baseline audit", "", f"- Fixed D0 v2 clear-first population: **{facts['rows']:,}** candidate IDs (`{facts['ordered_candidate_id_sha256']}`).", "- B0 is the unchanged frozen CONTINUE outcome; B1 is the paired EXIT_NOW counterfactual. No learned score, threshold, top-k rule, entry, sizing, or portfolio policy is used.", f"- Signed baseline uplift, EXIT_NOW minus CONTINUE: **{facts['baseline_uplift_exit_minus_continue_mean_bps']:.6f} bps/trade**; mechanical exit superior overall: **{facts['mechanical_exit_superior_overall']}**. This signed difference is not labelled a giveback cost.", f"- Positive loss avoided by correct mechanical exits: sum **{facts['positive_loss_avoided_if_exit']['sum_bps']:.6f} bps**, unconditional mean **{facts['positive_loss_avoided_if_exit']['unconditional_mean_bps']:.6f} bps/trade**, p90 **{facts['positive_loss_avoided_if_exit']['p90_bps']:.6f} bps**.", f"- False-exit opportunity cost: sum **{facts['false_exit_opportunity_cost']['sum_bps']:.6f} bps**, unconditional mean **{facts['false_exit_opportunity_cost']['unconditional_mean_bps']:.6f} bps/trade**, p90 **{facts['false_exit_opportunity_cost']['p90_bps']:.6f} bps**.", f"- Paired UTC-day block bootstrap (2,000 deterministic resamples; pooled estimator within every draw): {facts['day_comparison']['paired_utc_day_block_bootstrap_95pct_ci_bps'][0]:.6f} to {facts['day_comparison']['paired_utc_day_block_bootstrap_95pct_ci_bps'][1]:.6f} bps/trade.", f"- Volume bucket: `{bucket_status['volume_bucket']}`; no volume result is claimed when exact 1m volume is absent. Volatility bucket: `{bucket_status['volatility_bucket']}`. Regime bucket: `{bucket_status['regime_bucket']}`.", "- Aggregate sums are unweighted sums of per-trade bps, presented alongside bps/trade means; they are not a portfolio P&L claim.", ""]
        (stage / "stage_d_d1_baseline_audit.md").write_text("\n".join(report))
        write_json(stage / "stage_d_d1_evaluation_contract.json", {
            "schema": SCHEMA, "population": "exact D0 v2 clear-first actionable rows", "baselines": {"B0": "CONTINUE_FROZEN_POLICY", "B1": "EXIT_NOW_AT_FIRST_CLEAR"},
            "economics": "gross - one frozen row cost = net for each paired arm", "fixed_candidate_ids": facts["ordered_candidate_id_sha256"],
            "effect_semantics": {"signed_baseline_uplift": "EXIT_NOW minus CONTINUE; may be negative; never called giveback cost", "positive_loss_avoided": "max(EXIT_NOW minus CONTINUE, 0)", "false_exit_opportunity_cost": "max(CONTINUE minus EXIT_NOW, 0)", "row_distribution_artifact": "stage_d_d1_paired_row_comparisons.parquet"},
            "bucket_contract": {"time_to_clear_minutes": dict(zip(TIME_LABELS, ["1-5", "6-15", "16-30", "31-60", "61-120", "121-240", "241-480", "481-718"])), "volume_z_at_clear": dict(zip(VOLUME_LABELS, ["<=-1", "(-1,0]", "(0,1]", ">1"])), "realised_volatility_bps": dict(zip(VOLATILITY_LABELS, ["<=25", "(25,50]", "(50,100]", ">100"]))},
            "bucket_status": bucket_status, "regime": "not reported: A8 rejected OOF lineage", "bootstrap": facts["day_comparison"],
            "prohibited": ["model fit", "threshold selection", "top-k action rule", "entry-policy change", "portfolio-policy change"],
        })
        outputs = {path.name: sha256(path) for path in sorted(stage.iterdir()) if path.is_file()}
        manifest = {
            "schema": SCHEMA, "status": "SEALED_DETERMINISTIC_D1_BASELINES_NO_MODEL_OR_POLICY_CHANGE", "facts": facts,
            "bucket_status": bucket_status, "inputs": {str(counterfactuals_path): sha256(counterfactuals_path), str(features_path): sha256(features_path)},
            "outputs_sha256": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactuals", type=Path, default=COUNTERFACTUALS)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(counterfactuals_path=args.counterfactuals, features_path=args.features, output=args.output), indent=2, default=str))


if __name__ == "__main__":
    main()
