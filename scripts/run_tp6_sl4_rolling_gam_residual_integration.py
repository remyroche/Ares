#!/usr/bin/env python3
"""Integrate the one-month gated rolling GAM into the residual/meta stack.

The structural GAM is treated as a prequential, long-side-only feature source:
the row's one-month GAM output was itself fitted before that target month.  We
compare a matched residual/meta control against three ablations:

* ``gam_input``: keep the base bps anchor, add the one canonical GAM
  disagreement field (``gam_delta_bps``) to both LambdaRank heads;
* ``gam_modulation``: replace the base bps anchor with the saved one-month
  gated GAM score, without exposing GAM fields to the heads;
* ``gam_input_modulation``: do both.

All arms use the existing consensus and residual LambdaRank contracts, fit
before each held 2025 month and ranked globally after monthly/side percentile
normalisation.  The selected GAM is exactly one-month ``zero`` GAM gamma .25;
when its local structural contract is invalid, its score is the base anchor.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import (  # noqa: E402
    MONTHS,
    TAILS,
    _load,
    _map_base,
    _month_weights,
    _pct,
    _prep,
)


DEFAULT_ROLLING = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet"
DEFAULT_CONTROL = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/predictions_2025.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_rolling_gam_residual_integration_20260815_v1"
DEFAULT_CANONICAL_BASELINE = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/predictions_2025.parquet"
SEED = 20260815
SIDE = "long"
# The residual/meta contract receives one canonical GAM signal only.  The
# residual and expected-value fields are algebraically redundant
# (gam_residual_bps == 4 * gam_delta_bps for this GAM), while mass/count
# diagnostics are transport metadata rather than decision features.  Invalid
# transport is hard-gated to the exact control before this field is exposed.
GAM_INPUT_FIELDS = ["gam_delta_bps"]
GAM_INTERNAL_FIELDS = [
    "gam_expected_bps",
    "gam_delta_bps",
    "gam_residual_bps",
    "gam_transport_valid",
    "gam_matched_mass",
    "gam_unmatched_mass",
    "gam_archetype_count",
    "gam_cluster_count",
]
ARMS = ("control", "gam_input", "gam_modulation", "gam_input_modulation")


def _hard_gate_gam_score(
    gam_score: np.ndarray,
    control_score: np.ndarray,
    transport_valid: np.ndarray,
) -> np.ndarray:
    """Use the GAM score only under a valid structural transport contract."""
    gam_score = np.asarray(gam_score, dtype=float)
    control_score = np.asarray(control_score, dtype=float)
    valid = np.asarray(transport_valid, dtype=bool)
    if gam_score.shape != control_score.shape or gam_score.shape != valid.shape:
        raise ValueError("GAM, control, and gate arrays must have identical shapes")
    return np.where(valid, gam_score, control_score).astype(np.float32)


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 32 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def _group(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = pd.to_datetime(frame["__ts__"], utc=True).dt.floor("4h").astype("int64").astype(str)
    order = np.argsort(q.to_numpy(), kind="stable")
    qs = q.iloc[order]
    counts = qs.groupby(qs, sort=False).size()
    valid = counts.index[counts.to_numpy() >= 2]
    keep = qs.isin(valid).to_numpy()
    order = order[keep]
    groups = qs.iloc[keep].groupby(qs.iloc[keep], sort=False).size().to_numpy(dtype=np.int32)
    return order, groups


def _rank_fit(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: Sequence[str],
    label: np.ndarray,
    *,
    equal_month: bool,
    seed: int,
    feature_fraction: float = 0.82,
) -> tuple[np.ndarray, np.ndarray]:
    xtr, med = _prep(train, list(fields))
    order, groups = _group(train)
    if len(groups) == 0:
        return np.zeros(len(train), dtype=np.float32), np.zeros(len(held), dtype=np.float32)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=10,
        n_estimators=140,
        learning_rate=0.035,
        max_depth=5,
        num_leaves=31,
        min_child_samples=180,
        feature_fraction=feature_fraction,
        bagging_fraction=0.82,
        bagging_freq=1,
        lambda_l1=0.02,
        lambda_l2=2.0,
        max_bin=127,
        label_gain=[0.0, 0.25, 1.0, 3.0, 7.0],
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    weights = _month_weights(train) if equal_month else np.ones(len(train), dtype=np.float32)
    model.fit(xtr.iloc[order], label[order], group=groups, sample_weight=weights[order])
    raw_train = np.asarray(model.predict(xtr), dtype=np.float32)
    xte, _ = _prep(held, list(fields), med)
    raw_held = np.asarray(model.predict(xte), dtype=np.float32)
    del model, xtr, xte
    gc.collect()
    return raw_train, raw_held


def _join_gam(x: pd.DataFrame, rolling_path: Path) -> pd.DataFrame:
    rolling = pd.read_parquet(rolling_path)
    rolling = rolling.loc[rolling.window_months.eq(1)].copy()
    rolling = rolling.loc[rolling.month.astype(str).isin([*MONTHS, "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"])].copy()
    if rolling.empty:
        raise ValueError("one-month rolling GAM artifact has no usable rows")
    rolling["gam_expected_bps"] = np.where(
        rolling["rolling_transport_valid"].astype(bool),
        rolling["rolling_gam_zero_gamma025"].to_numpy(float),
        rolling["base_expected_bps"].to_numpy(float),
    )
    rolling["gam_delta_bps"] = rolling["gam_expected_bps"] - rolling["base_expected_bps"]
    rolling["gam_residual_bps"] = np.where(
        rolling["rolling_transport_valid"].astype(bool),
        rolling["rolling_gam_zero_residual"].to_numpy(float),
        0.0,
    )
    rolling["gam_transport_valid"] = rolling["rolling_transport_valid"].astype(float)
    rolling["gam_matched_mass"] = rolling["archetype_matched_mass"].astype(float)
    rolling["gam_unmatched_mass"] = rolling["archetype_unmatched_mass"].astype(float)
    rolling["gam_archetype_count"] = rolling["archetype_count"].astype(float)
    rolling["gam_cluster_count"] = rolling["rolling_cluster_count"].astype(float)
    gam = rolling[["candidate_id", "__ts__", "month", *GAM_INTERNAL_FIELDS]].drop_duplicates(["candidate_id", "__ts__"])
    # Keep the complete residual/meta population.  Rows before the first
    # available rolling-GAM month receive an explicit neutral fallback below;
    # dropping them would give the GAM arms a different training substrate.
    out = x.merge(gam, on=["candidate_id", "__ts__", "month"], how="left", validate="one_to_one")
    if out.empty:
        raise ValueError("canonical panel is empty")
    return out


def _fill_gam_history(frame: pd.DataFrame, base_anchor: np.ndarray) -> None:
    """Neutralize rows predating the first available one-month GAM output."""
    frame["gam_expected_bps"] = pd.to_numeric(frame["gam_expected_bps"], errors="coerce").fillna(pd.Series(base_anchor, index=frame.index))
    frame["gam_delta_bps"] = pd.to_numeric(frame["gam_delta_bps"], errors="coerce").fillna(0.0)
    frame["gam_residual_bps"] = pd.to_numeric(frame["gam_residual_bps"], errors="coerce").fillna(0.0)
    frame["gam_transport_valid"] = pd.to_numeric(frame["gam_transport_valid"], errors="coerce").fillna(0.0)
    frame["gam_matched_mass"] = pd.to_numeric(frame["gam_matched_mass"], errors="coerce").fillna(0.0)
    frame["gam_unmatched_mass"] = pd.to_numeric(frame["gam_unmatched_mass"], errors="coerce").fillna(1.0)
    frame["gam_archetype_count"] = pd.to_numeric(frame["gam_archetype_count"], errors="coerce").fillna(0.0)
    frame["gam_cluster_count"] = pd.to_numeric(frame["gam_cluster_count"], errors="coerce").fillna(0.0)


def _fit_heads(train: pd.DataFrame, held: pd.DataFrame, anchor_train: np.ndarray, anchor_held: np.ndarray, *, use_gam_inputs: bool, month: str, extra_fields: Sequence[str] | None = None, feature_fraction: float = 0.82, seed_base: int | None = None, reverse_feature_order: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = train.copy(); held = held.copy()
    train["base_anchor"] = anchor_train; held["base_anchor"] = anchor_held
    residual = train.exact_net_bps.to_numpy(float) - anchor_train
    context = list(train.attrs.get("context_fields", []))
    base_fields = ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context]
    if use_gam_inputs:
        base_fields += GAM_INPUT_FIELDS
    if extra_fields:
        base_fields += list(extra_fields)
    base_fields = list(dict.fromkeys(f for f in base_fields if f in train.columns))
    if reverse_feature_order:
        base_fields = list(reversed(base_fields))
    seed_root = SEED if seed_base is None else int(seed_base)
    consensus_parts: list[np.ndarray] = []
    for cap in (25, 40, 60, min(73, len(context))):
        fields = [f for f in base_fields if f not in context] + [f for f in context[:cap] if f in train.columns]
        for equal_month in (False, True):
            grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
            tr_raw, te_raw = _rank_fit(train, held, fields, grade, equal_month=equal_month, seed=seed_root + int(month[-2:]) * 100 + cap + int(equal_month), feature_fraction=feature_fraction)
            consensus_parts.append(_pct(te_raw, tr_raw))
    consensus = np.nanmedian(np.column_stack(consensus_parts), axis=1).astype(np.float32)
    residual_grade = np.digitize(residual, [-100.0, -25.0, 25.0, 100.0]).astype(np.int32)
    tr_res, te_res = _rank_fit(train, held, base_fields, residual_grade, equal_month=True, seed=seed_root + int(month[-2:]) * 1000 + 99, feature_fraction=feature_fraction)
    return consensus, _pct(te_res, tr_res), residual, np.asarray(anchor_held, dtype=np.float32)


def _metrics(pred: pd.DataFrame, score_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    glob, monthly, stability = [], [], []
    for arm in score_cols:
        for tail in TAILS:
            n = max(1, int(math.ceil(len(pred) * tail)))
            top = pred.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            glob.append({"arm": arm, "scope": "global_long_2025", "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": _rank_ic(pred[arm].to_numpy(float), pred.exact_net_bps.to_numpy(float))})
        vals, ics = [], []
        for month, block in pred.groupby("month", sort=True):
            n = max(1, int(math.ceil(len(block) * 0.05)))
            top = block.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            vals.append(float(top.exact_net_bps.mean())); ics.append(_rank_ic(block[arm].to_numpy(float), block.exact_net_bps.to_numpy(float)))
            monthly.append({"arm": arm, "month": month, "tail": 0.05, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": vals[-1], "rank_ic": ics[-1]})
        arr = np.asarray(vals, dtype=float); med = float(np.median(arr)); mad = float(np.median(np.abs(arr - med)))
        stability.append({"arm": arm, "months": len(arr), "mean_top5_net_bps": float(np.mean(arr)), "median_top5_net_bps": med, "mad_top5_net_bps": mad, "worst_month_top5_net_bps": float(np.min(arr)), "positive_months_top5": int(np.sum(arr > 0)), "mean_month_rank_ic": float(np.nanmean(ics))})
    return pd.DataFrame(glob), pd.DataFrame(monthly), pd.DataFrame(stability)


def _metrics_by_arm(pred: pd.DataFrame, score_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate each ablation on its own matched candidate population."""
    glob, monthly, stability = [], [], []
    for arm, block in pred.groupby("arm", sort=True):
        g, m, s = _metrics(block.copy(), score_cols)
        for df in (g, m, s):
            if not df.empty:
                df["arm"] = df["arm"].map(lambda value: f"{arm}__{value}")
        glob.append(g); monthly.append(m); stability.append(s)
    return pd.concat(glob, ignore_index=True), pd.concat(monthly, ignore_index=True), pd.concat(stability, ignore_index=True)


def _canonical_metrics(pred: pd.DataFrame, score_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate scores on the exact Base+Consensus candidate population."""
    global_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for score in score_cols:
        values, ics = [], []
        for tail in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
            n = max(1, int(math.ceil(len(pred) * tail)))
            top = pred.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            global_rows.append({
                "arm": score, "scope": "global_long_2025", "tail": tail, "trades": n,
                "gross_bps_per_trade": float(top.exact_gross_bps.mean()),
                "net_bps_per_trade": float(top.exact_net_bps.mean()),
                "rank_ic": _rank_ic(pred[score].to_numpy(float), pred.exact_net_bps.to_numpy(float)),
            })
        for month, block in pred.groupby("month", sort=True):
            n = max(1, int(math.ceil(len(block) * 0.05)))
            top = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            net = float(top.exact_net_bps.mean())
            ic = _rank_ic(block[score].to_numpy(float), block.exact_net_bps.to_numpy(float))
            values.append(net); ics.append(ic)
            monthly_rows.append({
                "arm": score, "month": month, "tail": 0.05, "trades": n,
                "gross_bps_per_trade": float(top.exact_gross_bps.mean()),
                "net_bps_per_trade": net, "rank_ic": ic,
            })
        arr = np.asarray(values, dtype=float)
        med = float(np.nanmedian(arr))
        stability_rows.append({
            "arm": score, "months": len(arr),
            "mean_top5_net_bps": float(np.nanmean(arr)),
            "median_top5_net_bps": med,
            "mad_top5_net_bps": float(np.nanmedian(np.abs(arr - med))),
            "worst_month_top5_net_bps": float(np.nanmin(arr)),
            "positive_months_top5": int(np.sum(arr > 0.0)),
            "mean_month_rank_ic": float(np.nanmean(ics)),
        })
    return pd.DataFrame(global_rows), pd.DataFrame(monthly_rows), pd.DataFrame(stability_rows)


def run_canonical_baseline(
    *,
    rolling_path: Path,
    baseline_path: Path,
    output_dir: Path,
) -> Path:
    """Run the GAM ablations against the frozen 75/25 Base+Consensus arm.

    The no-GAM row is copied from the canonical handover artifact, rather than
    reconstructed by this script.  This keeps the control, exits, candidate
    IDs, and monthly normalization identical to the declared baseline.
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    x, context, context_hash = _load()
    x = x.loc[x.side_name.eq(SIDE)].copy()
    x = _join_gam(x, rolling_path)
    baseline = pd.read_parquet(baseline_path)
    baseline = baseline.loc[
        baseline.side_name.eq(SIDE) & baseline.month.astype(str).isin(MONTHS),
        ["candidate_id", "month", "base_plus_consensus25", "exact_net_bps", "exact_gross_bps"],
    ].copy()
    if baseline.candidate_id.duplicated().any():
        raise ValueError("canonical baseline has duplicate candidate IDs")
    if len(baseline) != len(x.loc[x.month.astype(str).isin(MONTHS)]):
        raise ValueError("canonical baseline and GAM population have different row counts")
    if not np.allclose(
        pd.to_numeric(baseline.exact_net_bps, errors="coerce"),
        pd.to_numeric(x.set_index("candidate_id").loc[baseline.candidate_id, "exact_net_bps"].to_numpy(), errors="coerce"),
        equal_nan=True,
    ):
        raise ValueError("canonical baseline and GAM labels differ")

    parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for month in MONTHS:
        held = x.loc[x.month.astype(str).eq(month)].copy()
        train = x.loc[
            x.__ts__.lt(pd.Timestamp(month, tz="UTC"))
            & x.label_available_ts.lt(pd.Timestamp(month, tz="UTC"))
        ].copy()
        if held.empty or len(train) < 300:
            continue
        base_train, base_held = _map_base(train, held)
        _fill_gam_history(train, base_train)
        _fill_gam_history(held, base_held)
        train.attrs["context_fields"] = context; held.attrs["context_fields"] = context
        base_rank = _pct(held.base_score.to_numpy(float), train.base_score.to_numpy(float))
        gam_rank = _pct(held.gam_expected_bps.to_numpy(float), train.gam_expected_bps.to_numpy(float))

        # GAM as a residual/meta input, keeping the declared base anchor.
        gam_consensus, _, _, _ = _fit_heads(
            train, held, base_train, base_held,
            use_gam_inputs=True, month=month,
        )
        # GAM modulation is retained only as a diagnostic; it is not the
        # canonical promotion candidate.
        mod_consensus, _, _, _ = _fit_heads(
            train, held, train.gam_expected_bps.to_numpy(float),
            held.gam_expected_bps.to_numpy(float),
            use_gam_inputs=False, month=month,
        )
        both_consensus, _, _, _ = _fit_heads(
            train, held, train.gam_expected_bps.to_numpy(float),
            held.gam_expected_bps.to_numpy(float),
            use_gam_inputs=True, month=month,
        )
        out = held[["candidate_id", "__ts__", "month", "exact_net_bps", "exact_gross_bps"]].copy()
        out["base_plus_consensus25"] = np.nan
        out["gam_input"] = 0.75 * base_rank + 0.25 * gam_consensus
        out["gam_modulation"] = 0.75 * gam_rank + 0.25 * mod_consensus
        out["gam_input_modulation"] = 0.75 * gam_rank + 0.25 * both_consensus
        out["base_rank"] = base_rank
        out["gam_delta_bps"] = held.gam_delta_bps.to_numpy(float)
        out["transport_valid"] = held.gam_transport_valid.to_numpy(float)
        out["fit_month"] = month
        parts.append(out)
        audit.append({
            "month": month, "train_rows": len(train), "held_rows": len(held),
            "gam_valid_train_fraction": float(train.gam_transport_valid.mean()),
            "gam_valid_held_fraction": float(held.gam_transport_valid.mean()),
            "gam_field": "gam_delta_bps", "base_contract": "canonical Base+Consensus 75/25",
        })
    pred = pd.concat(parts, ignore_index=True)
    pred = pred.merge(
        baseline[["candidate_id", "month", "base_plus_consensus25"]],
        on=["candidate_id", "month"], how="left", validate="one_to_one", suffixes=("", "__baseline"),
    )
    pred["base_plus_consensus25"] = pred["base_plus_consensus25__baseline"]
    pred = pred.drop(columns=["base_plus_consensus25__baseline"])
    # Enforce the frozen production gate explicitly.  The GAM head may be
    # fitted for diagnostic purposes on every fold, but an invalid structural
    # transport contract must expose the exact canonical control score, never
    # a partially trained GAM score.
    valid = pred["transport_valid"].fillna(0.0).astype(bool).to_numpy()
    pred["gam_input"] = _hard_gate_gam_score(
        pred["gam_input"].to_numpy(float),
        pred["base_plus_consensus25"].to_numpy(float),
        valid,
    )
    pred["gam_input_modulation"] = _hard_gate_gam_score(
        pred["gam_input_modulation"].to_numpy(float),
        pred["base_plus_consensus25"].to_numpy(float),
        valid,
    )
    # The canonical baseline is already monthly normalized.  Re-rank only the
    # newly generated scores in the same monthly/side space.
    for col in ("gam_input", "gam_modulation", "gam_input_modulation"):
        pred[col] = pred.groupby(["fit_month"], sort=False)[col].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    global_metrics, monthly_metrics, stability_metrics = _canonical_metrics(
        pred, ["base_plus_consensus25", "gam_input", "gam_modulation", "gam_input_modulation"]
    )
    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    global_metrics.to_parquet(output_dir / "metrics_global.parquet", index=False)
    monthly_metrics.to_parquet(output_dir / "metrics_monthly.parquet", index=False)
    stability_metrics.to_parquet(output_dir / "metrics_stability.parquet", index=False)
    pd.DataFrame(audit).to_parquet(output_dir / "fit_audit.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_gam_canonical_base_consensus_integration_v1",
        "status": "COMPLETE", "side": SIDE, "held_months": list(MONTHS),
        "canonical_baseline": str(baseline_path),
        "canonical_baseline_score": "base_plus_consensus25 = 0.75 base_rank + 0.25 consensus_rank",
        "rolling_gam": str(rolling_path), "rolling_window": 1, "gamma": 0.25,
        "gam_input_contract": ["gam_delta_bps"], "transport_invalid_rule": "exact control fallback",
        "exits": "TP6/SL4, 12h, 100 bps cost", "global_ranking": "pooled global after monthly normalization",
        "base_ev_modulation_in_canonical": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (output_dir / "TP6_SL4_GAM_CANONICAL_BASELINE_REPORT.md").write_text(
        "# TP6/SL4 GAM comparison against canonical Base+Consensus baseline\n\n"
        + global_metrics.round(3).to_string(index=False) + "\n\n## Stability\n\n"
        + stability_metrics.round(3).to_string(index=False) + "\n"
    )
    return output_dir


def run(*, rolling_path: Path = DEFAULT_ROLLING, control_path: Path = DEFAULT_CONTROL, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    x, context, context_hash = _load()
    x = x.loc[x.side_name.eq(SIDE)].copy()
    x = _join_gam(x, rolling_path)
    control = pd.read_parquet(control_path)
    control = control.loc[control.side_name.eq(SIDE) & control.month.astype(str).isin(MONTHS), ["candidate_id", "month", "full_base_consensus_residual"]].copy()
    parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for month in MONTHS:
        held = x.loc[x.month.astype(str).eq(month)].copy()
        train = x.loc[(x.__ts__ < pd.Timestamp(month, tz="UTC")) & (x.label_available_ts < pd.Timestamp(month, tz="UTC"))].copy()
        if held.empty or len(train) < 300:
            continue
        base_train, base_held = _map_base(train, held)
        _fill_gam_history(train, base_train)
        _fill_gam_history(held, base_held)
        train.attrs["context_fields"] = context; held.attrs["context_fields"] = context
        for arm in ARMS:
            use_gam_inputs = arm in {"gam_input", "gam_input_modulation"}
            use_modulation = arm in {"gam_modulation", "gam_input_modulation"}
            anchor_train = train.gam_expected_bps.to_numpy(float) if use_modulation else base_train
            anchor_held = held.gam_expected_bps.to_numpy(float) if use_modulation else base_held
            consensus, residual_rank, residual_target, _ = _fit_heads(train, held, anchor_train, anchor_held, use_gam_inputs=use_gam_inputs, month=month)
            # Preserve the current stack's native base-score ranking unless
            # explicit GAM modulation is requested.  The mapped bps anchor is
            # still the residual/meta target reference; it is not the control
            # ranking signal.
            base_rank = _pct(anchor_held, anchor_train) if use_modulation else _pct(held.base_score.to_numpy(float), train.base_score.to_numpy(float))
            residual_only = residual_rank
            stack = (0.50 * base_rank + 0.25 * consensus + 0.25 * residual_only).astype(np.float32)
            held["base_expected_bps"] = np.asarray(base_held, dtype=np.float32)
            out = held[["candidate_id", "__ts__", "month", "exact_net_bps", "exact_gross_bps", "base_score", "base_expected_bps", *GAM_INPUT_FIELDS]].copy()
            out["arm"] = arm
            out["anchor_rank"] = base_rank
            out["consensus_rank"] = consensus
            out["residual_rank"] = residual_only
            out["stack_rank"] = stack
            out["residual_target_train_mean"] = float(np.mean(residual_target))
            out["gam_inputs_used"] = bool(use_gam_inputs)
            out["gam_modulation_used"] = bool(use_modulation)
            parts.append(out)
            audit.append({"month": month, "arm": arm, "train_rows": len(train), "held_rows": len(held), "gam_valid_train_fraction": float(train.gam_transport_valid.mean()), "gam_valid_held_fraction": float(held.gam_transport_valid.mean()), "gam_fields_added": GAM_INPUT_FIELDS if use_gam_inputs else [], "modulation": "gam_expected_bps" if use_modulation else "base_expected_bps", "query_groups": int(_group(train)[1].size)})
    pred = pd.concat(parts, ignore_index=True)
    # Normalize each arm in the same monthly/side space before pooled ranking.
    pred["stack_score"] = pred.groupby(["arm", "month"], sort=False)["stack_rank"].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    pred["anchor_score"] = pred.groupby(["arm", "month"], sort=False)["anchor_rank"].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    pred = pred.merge(control.rename(columns={"full_base_consensus_residual": "existing_control_stack"}), on=["candidate_id", "month"], how="left", validate="many_to_one")
    score_cols = ["stack_score", "anchor_score"]
    metrics_global, metrics_monthly, metrics_stability = _metrics_by_arm(pred, score_cols)
    # The existing control is already a normalized score; evaluate it on the
    # same long-side 2025 candidate population for a direct reference.
    control_eval = pred.drop_duplicates(["candidate_id", "month"]).copy()
    if control_eval.existing_control_stack.notna().any():
        cg, cm, cs = _metrics(control_eval.rename(columns={"existing_control_stack": "existing_control_stack_score"}), ["existing_control_stack_score"])
        cg["arm"] = "existing_control__existing_control_stack_score"; cm["arm"] = "existing_control__existing_control_stack_score"; cs["arm"] = "existing_control__existing_control_stack_score"
        metrics_global = pd.concat([cg, metrics_global], ignore_index=True)
        metrics_monthly = pd.concat([cm, metrics_monthly], ignore_index=True)
        metrics_stability = pd.concat([cs, metrics_stability], ignore_index=True)
    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    metrics_global.to_parquet(output_dir / "metrics_global.parquet", index=False)
    metrics_monthly.to_parquet(output_dir / "metrics_monthly.parquet", index=False)
    metrics_stability.to_parquet(output_dir / "metrics_stability.parquet", index=False)
    pd.DataFrame(audit).to_parquet(output_dir / "fit_audit.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_rolling_gam_residual_integration_v1",
        "status": "COMPLETE",
        "side": SIDE,
        "held_months": list(MONTHS),
        "rolling_gam": str(rolling_path),
        "rolling_window": 1,
        "rolling_gam_arm": "gated zero-exposure gamma .25; base fallback when transport invalid",
        "gam_input_contract": ["gam_delta_bps"],
        "arms": list(ARMS),
        "residual_meta": "existing 4-hour x side LambdaRank consensus and residual contracts; matched 2025 monthly refits",
        "context_sha256": context_hash,
        "global_ranking": "monthly long-side percentile normalization followed by pooled global top-k",
        "no_held_outcomes_in_gam": True,
        "artifacts": [p.name for p in output_dir.iterdir()],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = [
        "# TP6/SL4 one-month gated GAM residual/meta integration",
        "",
        "Long-side matched 2025 replay. The GAM is tested as input, modulation, or both inside the existing residual/meta stack.",
        "",
        "## Global metrics",
        "",
        metrics_global.round(3).to_string(index=False),
        "",
        "## Stability",
        "",
        metrics_stability.round(3).to_string(index=False),
    ]
    (output_dir / "TP6_SL4_ROLLING_GAM_RESIDUAL_INTEGRATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": len(pred), "metric_rows": len(metrics_global)}, indent=2))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", type=Path, default=DEFAULT_ROLLING)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical-baseline", action="store_true", help="Use TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807 as the exact no-GAM control.")
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_CANONICAL_BASELINE)
    args = parser.parse_args()
    if args.canonical_baseline:
        run_canonical_baseline(rolling_path=args.rolling, baseline_path=args.baseline_path, output_dir=args.output_dir)
    else:
        run(rolling_path=args.rolling, control_path=args.control, output_dir=args.output_dir)
