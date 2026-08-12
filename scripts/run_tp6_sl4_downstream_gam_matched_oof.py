#!/usr/bin/env python3
"""Matched OOF test of a causal GAM inside the TP6/SL4 Base+Consensus stack.

The canonical R3 output and 8-head LambdaRank consensus contract are kept
fixed.  A small causal GAM is fitted before each held month from earlier
resolved rows, using the base score plus a portable subset of the canonical
context fields.  It is then tested either as a meta input or as a modulation
of the base bps anchor before the residual heads are trained.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

ROOT = Path("/Users/remyroche/Documents/Ares")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS  # noqa: E402
from scripts.run_tp6_sl4_downstream_retrain_2025 import (  # noqa: E402
    MONTHS,
    TAILS,
    INPUT,
    _group,
    _load,
    _map_base,
    _month_weights,
    _pct,
    _prep,
    _rank_fit,
)


DEFAULT_CONTROL = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_downstream_gam_matched_oof_20260812_v1"
SEED = 20260812
GAM_CONTEXT_CAP = 12
GAM_VARIANTS = ("gam_meta_feature", "gam_pre_meta25", "gam_pre_meta50", "gam_pre_meta100")


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def _top_mean(y: np.ndarray, score: np.ndarray, fraction: float = 0.10) -> float:
    mask = np.isfinite(y) & np.isfinite(score)
    if not mask.any():
        return float("nan")
    y, score = np.asarray(y, float)[mask], np.asarray(score, float)[mask]
    n = max(1, int(math.ceil(len(y) * fraction)))
    return float(np.mean(y[np.argpartition(score, -n)[-n:]]))


def _portable_score(values: Sequence[float]) -> float:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return -np.inf
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med - 0.75 * mad - max(0.0, -float(np.min(x)))


def _binned_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray, bins: int = 8) -> np.ndarray:
    x_fit, y_fit, x_eval = np.asarray(x_fit, float), np.asarray(y_fit, float), np.asarray(x_eval, float)
    finite = np.isfinite(x_fit)
    fill = float(np.nanmedian(x_fit[finite])) if finite.any() else 0.0
    x_fit = np.where(np.isfinite(x_fit), x_fit, fill)
    x_eval = np.where(np.isfinite(x_eval), x_eval, fill)
    if len(x_fit) < 128 or np.unique(x_fit).size < 2:
        return np.full(len(x_eval), float(np.mean(y_fit)) if len(y_fit) else 0.0)
    edges = np.unique(np.nanquantile(x_fit, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.full(len(x_eval), float(np.mean(y_fit)))
    code = np.clip(np.digitize(x_fit, edges[1:-1]), 0, len(edges) - 2)
    means = np.full(len(edges) - 1, float(np.mean(y_fit)), float)
    for i in range(len(means)):
        if np.any(code == i):
            means[i] = float(np.mean(y_fit[code == i]))
    return means[np.clip(np.digitize(x_eval, edges[1:-1]), 0, len(edges) - 2)]


def _portable_gam_fields(train: pd.DataFrame, context: Sequence[str], cap: int = GAM_CONTEXT_CAP) -> tuple[list[str], pd.DataFrame]:
    """Select context terms from earlier expanding blocks only."""
    candidates = [f for f in context if f in train.columns and f != "base_score"]
    months = sorted(train["month"].dropna().astype(str).unique())
    blocks: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for month in months[-3:]:
        start = pd.Timestamp(month, tz="UTC")
        fit = train.loc[(train["__ts__"] + pd.Timedelta(hours=12)).lt(start)].copy()
        val = train.loc[train["month"].eq(month)].copy()
        if len(fit) >= 256 and len(val) >= 128:
            blocks.append((fit, val))
    rows: list[dict[str, Any]] = []
    y_all = train["exact_net_bps"].to_numpy(float)
    for field in candidates:
        values, ics = [], []
        raw = pd.to_numeric(train[field], errors="coerce").to_numpy(float)
        coverage = float(np.isfinite(raw).mean()) if len(raw) else 0.0
        for fit, val in blocks:
            y_fit = fit["exact_net_bps"].to_numpy(float)
            y_val = val["exact_net_bps"].to_numpy(float)
            pred = _binned_predict(fit[field].to_numpy(float), y_fit, val[field].to_numpy(float))
            values.append(_top_mean(y_val, pred))
            ics.append(_rank_ic(pred, y_val))
        finite_ic = np.asarray([v for v in ics if np.isfinite(v)], float)
        rows.append({
            "feature": field,
            "coverage": coverage,
            "portable_score_bps": _portable_score(values),
            "top10_median_bps": float(np.median(values)) if values else float("nan"),
            "top10_worst_bps": float(np.min(values)) if values else float("nan"),
            "positive_block_fraction": float(np.mean(np.asarray(values) > 0.0)) if values else 0.0,
            "median_rank_ic": float(np.median(finite_ic)) if len(finite_ic) else float("nan"),
            "rank_ic_sign_fraction": float(np.mean(finite_ic > 0.0)) if len(finite_ic) else 0.0,
            "validation_blocks": len(values),
        })
    audit = pd.DataFrame(rows)
    if audit.empty:
        return ["base_score"], audit
    audit = audit.sort_values(["portable_score_bps", "median_rank_ic", "feature"], ascending=[False, False, True], kind="stable")
    stable = audit.loc[(audit.rank_ic_sign_fraction >= 0.5) | (audit.positive_block_fraction >= 0.5)]
    chosen = stable.head(cap) if len(stable) >= cap else audit.head(cap)
    selected = ["base_score", *chosen.feature.astype(str).tolist()]
    audit = audit.assign(selected=audit.feature.isin(selected))
    return list(dict.fromkeys(selected)), audit.reset_index(drop=True)


def _fit_simple_gam(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, med = _prep(train, list(fields))
    x_held, _ = _prep(held, list(fields), med)
    y = train.exact_net_bps.to_numpy(float)
    model = Pipeline([
        ("splines", SplineTransformer(n_knots=2, degree=1, knots="quantile", extrapolation="linear", include_bias=False)),
        ("ridge", Ridge(alpha=20.0)),
    ])
    model.fit(x_train, y)
    raw_train = np.asarray(model.predict(x_train), float)
    raw_held = np.asarray(model.predict(x_held), float)
    ok = np.isfinite(raw_train) & np.isfinite(y)
    if ok.sum() >= 32 and np.unique(raw_train[ok]).size >= 2:
        mapper = IsotonicRegression(out_of_bounds="clip", y_min=-1000.0, y_max=1000.0).fit(raw_train[ok], y[ok])
        mapped_train = np.asarray(mapper.predict(raw_train), float)
        mapped_held = np.asarray(mapper.predict(raw_held), float)
    else:
        mean = float(np.mean(y)) if len(y) else 0.0
        mapped_train = np.full(len(train), mean, float)
        mapped_held = np.full(len(held), mean, float)
    return raw_train, raw_held, mapped_train, mapped_held


def _fit_consensus_variant(train: pd.DataFrame, held: pd.DataFrame, anchor_train: np.ndarray, anchor_held: np.ndarray, gam_train: np.ndarray, gam_held: np.ndarray, context: Sequence[str], variant: str, month: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Refit the unchanged 8-head consensus contract with GAM where declared."""
    use_gam_feature = variant in {"gam_meta_feature", "gam_pre_meta25", "gam_pre_meta50", "gam_pre_meta100"}
    fields_base = ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context]
    if use_gam_feature:
        fields_base += ["gam_expected_bps", "gam_delta_bps"]
    train = train.copy(); held = held.copy()
    train["base_anchor"] = anchor_train; held["base_anchor"] = anchor_held
    train["gam_expected_bps"] = gam_train; held["gam_expected_bps"] = gam_held
    train["gam_delta_bps"] = gam_train - anchor_train; held["gam_delta_bps"] = gam_held - anchor_held
    residual = train.exact_net_bps.to_numpy(float) - anchor_train
    if variant == "gam_pre_meta25":
        residual = train.exact_net_bps.to_numpy(float) - (0.75 * anchor_train + 0.25 * gam_train)
    elif variant == "gam_pre_meta50":
        residual = train.exact_net_bps.to_numpy(float) - (0.50 * anchor_train + 0.50 * gam_train)
    elif variant == "gam_pre_meta100":
        residual = train.exact_net_bps.to_numpy(float) - gam_train
    grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
    raw_outputs = []
    for cap in (25, 40, 60, min(73, len(context))):
        fields = fields_base[:4 + cap]
        if use_gam_feature:
            fields += ["gam_expected_bps", "gam_delta_bps"]
        for equal_month in (False, True):
            tr_raw, te_raw = _rank_fit(
                train, held, fields, grade, equal_month=equal_month,
                seed=SEED + int(month[-2:]) * 10000 + (1 if side == "long" else 2) * 100 + cap + int(equal_month),
            )
            raw_outputs.append(_pct(te_raw, tr_raw))
    return np.nanmedian(np.column_stack(raw_outputs), axis=1).astype("float32"), residual


def _metric_rows(frame: pd.DataFrame, score_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    glob, monthly, stability = [], [], []
    for col in score_cols:
        for tail in TAILS:
            n = max(1, int(math.ceil(len(frame) * tail)))
            top = frame.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            glob.append({"arm": col, "scope": "global_2025", "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": _rank_ic(frame[col].to_numpy(float), frame.exact_net_bps.to_numpy(float))})
        vals, ics = [], []
        for month, g in frame.groupby("month", sort=True):
            n = max(1, int(math.ceil(len(g) * 0.05)))
            top = g.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            vals.append(float(top.exact_net_bps.mean())); ics.append(_rank_ic(g[col].to_numpy(float), g.exact_net_bps.to_numpy(float)))
            monthly.append({"arm": col, "month": month, "tail": 0.05, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": vals[-1], "rank_ic": ics[-1]})
        a = np.asarray(vals, float); med = float(np.median(a)); mad = float(np.median(np.abs(a - med)))
        stability.append({"arm": col, "months": len(a), "mean_top5_net_bps": float(np.mean(a)), "median_top5_net_bps": med, "mad_top5_net_bps": mad, "worst_month_top5_net_bps": float(np.min(a)), "positive_months_top5": int(np.sum(a > 0)), "portability_score_bps": med - 0.75 * mad - max(0.0, -float(np.min(a))), "mean_month_rank_ic": float(np.nanmean(ics))})
    return pd.DataFrame(glob), pd.DataFrame(monthly), pd.DataFrame(stability)


def run(*, output_dir: Path = DEFAULT_OUTPUT, control_dir: Path = DEFAULT_CONTROL) -> Path:
    output_dir, control_dir = Path(output_dir), Path(control_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    x, context, context_hash = _load()
    control = pd.read_parquet(control_dir / "predictions_2025.parquet")
    control = control.loc[control.month.isin(MONTHS)].copy()
    required_control = {"candidate_id", "base_plus_consensus25", "base_only"}
    if not required_control.issubset(control.columns):
        raise ValueError(f"canonical control lacks {sorted(required_control - set(control.columns))}")
    scores: list[pd.DataFrame] = []
    gam_rows: list[dict[str, Any]] = []
    selector_parts: list[pd.DataFrame] = []
    for month in MONTHS:
        held_all = x.loc[x.month.eq(month)].copy()
        train_all = x.loc[(x.__ts__ < pd.Timestamp(month, tz="UTC")) & (x.label_available_ts < pd.Timestamp(month, tz="UTC"))].copy()
        if len(held_all) == 0 or len(train_all) < 1000:
            continue
        for side in ("long", "short"):
            train = train_all.loc[train_all.side_name.eq(side)].copy(); held = held_all.loc[held_all.side_name.eq(side)].copy()
            if len(train) < 300 or len(held) == 0:
                continue
            base_train, base_held = _map_base(train, held)
            gam_fields, audit = _portable_gam_fields(train, context, cap=GAM_CONTEXT_CAP)
            if not audit.empty:
                selector_parts.append(audit.assign(month=month, side_name=side, selected_fields=audit.feature.isin(gam_fields)))
            raw_tr, raw_te, gam_tr, gam_te = _fit_simple_gam(train, held, gam_fields)
            for variant in GAM_VARIANTS:
                if variant == "gam_pre_meta25":
                    anchor_tr, anchor_te = 0.75 * base_train + 0.25 * gam_tr, 0.75 * base_held + 0.25 * gam_te
                elif variant == "gam_pre_meta50":
                    anchor_tr, anchor_te = 0.50 * base_train + 0.50 * gam_tr, 0.50 * base_held + 0.50 * gam_te
                elif variant == "gam_pre_meta100":
                    anchor_tr, anchor_te = gam_tr, gam_te
                else:
                    anchor_tr, anchor_te = base_train, base_held
                consensus_te, residual = _fit_consensus_variant(train, held, anchor_tr, anchor_te, gam_tr, gam_te, context, variant, month, side)
                base_signal_te = train.base_score.to_numpy(float)
                if variant != "gam_meta_feature":
                    base_signal_te = anchor_te
                    base_signal_tr = anchor_tr
                else:
                    base_signal_tr, base_signal_te = train.base_score.to_numpy(float), held.base_score.to_numpy(float)
                base_rank_te = _pct(base_signal_te, base_signal_tr)
                gam_rank_te = _pct(gam_te, gam_tr)
                out = held.loc[:, ["candidate_id", "__ts__", "side_name", "month", "exact_net_bps", "exact_gross_bps"]].copy()
                out["gam_anchor_bps"] = gam_te; out["gam_delta_bps"] = gam_te - base_held; out["gam_only"] = gam_rank_te
                out["base_component"] = base_rank_te; out["consensus_component"] = consensus_te
                out[variant] = 0.75 * base_rank_te + 0.25 * consensus_te
                out["gam_train_rows"] = len(train); out["gam_fields"] = json.dumps(gam_fields); out["gam_raw"] = raw_te
                scores.append(out)
                gam_rows.append({"month": month, "side_name": side, "variant": variant, "train_rows": len(train), "held_rows": len(held), "gam_fields": gam_fields, "gam_raw_rank_ic": _rank_ic(raw_te, held.exact_net_bps.to_numpy(float)), "gam_mapped_rank_ic": _rank_ic(gam_te, held.exact_net_bps.to_numpy(float)), "base_anchor_rank_ic": _rank_ic(base_held, held.exact_net_bps.to_numpy(float)), "label_available_before_evaluation": bool((train.label_available_ts < pd.Timestamp(month, tz="UTC")).all())})
    integ = pd.concat(scores, ignore_index=True)
    if integ.empty:
        raise ValueError("no integrated GAM predictions were produced")
    # Use a single row per candidate.  GAM variants share the same candidate
    # support, so pivoting keeps all scores matched exactly.
    value_cols = ["gam_anchor_bps", "gam_delta_bps", "gam_only", "base_component", "consensus_component", *GAM_VARIANTS, "gam_train_rows", "gam_fields", "gam_raw"]
    integ = integ.sort_values(["candidate_id", "month", "side_name"], kind="stable")
    # The loop emits one row per variant; combine those rows without averaging.
    key = ["candidate_id", "__ts__", "side_name", "month", "exact_net_bps", "exact_gross_bps"]
    base_rows = integ.groupby(key, sort=False, as_index=False).first()
    for col in GAM_VARIANTS:
        # groupby.first above preserves each variant only in its own row; fill
        # the missing columns by candidate/variant pivot.
        vals = integ.loc[:, key + [col]].dropna(subset=[col]).drop_duplicates(key)
        base_rows = base_rows.drop(columns=[col], errors="ignore").merge(vals, on=key, how="left", validate="one_to_one")
    # The preceding merge is intentionally replaced by a direct pivot for
    # robust support when pandas retains NaN columns in the groupby result.
    for col in GAM_VARIANTS:
        vals = integ.loc[:, key + ["gam_fields", col]].dropna(subset=[col]).drop_duplicates(key).rename(columns={col: f"__{col}"})
        base_rows = base_rows.drop(columns=[col], errors="ignore").merge(vals.loc[:, key + [f"__{col}"]], on=key, how="left", validate="one_to_one").rename(columns={f"__{col}": col})
    control_scores = control.loc[:, ["candidate_id", "base_plus_consensus25", "base_only"]]
    pred = base_rows.merge(control_scores, on="candidate_id", how="inner", validate="one_to_one")
    pred = pred.rename(columns={"base_plus_consensus25": "control_base_plus_consensus25", "base_only": "control_base_only"})
    score_cols = ["control_base_only", "control_base_plus_consensus25", "gam_only", *GAM_VARIANTS]
    for col in score_cols:
        pred[col] = pred.groupby(["month", "side_name"], sort=False)[col].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    glob, monthly, stability = _metric_rows(pred, score_cols)
    side_rows = []
    for col in score_cols:
        for side, g in pred.groupby("side_name", sort=True):
            n = max(1, int(math.ceil(len(pred) * 0.05)))
            top = pred.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            top_side = top.loc[top.side_name.eq(side)]
            side_rows.append({"arm": col, "side_name": side, "candidate_rows": len(g), "global_top5_rows": len(top_side), "global_top5_net_bps": float(top_side.exact_net_bps.mean()) if len(top_side) else float("nan"), "rank_ic": _rank_ic(g[col].to_numpy(float), g.exact_net_bps.to_numpy(float))})
    output_dir.mkdir(parents=True); pred.to_parquet(output_dir / "matched_predictions.parquet", index=False, compression="zstd")
    glob.to_parquet(output_dir / "metrics_global.parquet", index=False); monthly.to_parquet(output_dir / "metrics_monthly.parquet", index=False); stability.to_parquet(output_dir / "metrics_stability.parquet", index=False); pd.DataFrame(side_rows).to_parquet(output_dir / "metrics_side_top5.parquet", index=False); pd.DataFrame(gam_rows).to_parquet(output_dir / "gam_fold_audit.parquet", index=False)
    if selector_parts:
        pd.concat(selector_parts, ignore_index=True).to_parquet(output_dir / "gam_feature_selection_audit.parquet", index=False)
    manifest = {"schema": "tp6_sl4_downstream_gam_matched_oof_v1", "status": "COMPLETE", "input": str(INPUT), "control": str((control_dir / "predictions_2025.parquet").resolve()), "rows_scored": len(pred), "held_months": list(MONTHS), "gam": "causal prior-only 2-knot degree-1 spline + Ridge(alpha=20), isotonic net-bps map", "gam_fields": "base_score plus <=12 canonical causal context fields selected on earlier monthly blocks", "variants": list(GAM_VARIANTS), "meta": "same 8 native LambdaRank consensus heads, 4-hour UTC x side queries and fixed gains/parameters", "ranking": "monthly side percentile normalization followed by one pooled global ranking", "cost_contract": "exact_net_bps = exact_gross_bps - one TP6/SL4 cost encoded in source", "no_held_outcomes_in_fit": True, "context_sha256": context_hash, "artifacts": [p.name for p in output_dir.iterdir()]}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = ["# TP6/SL4 downstream GAM matched OOF replay", "", "GAM is tested inside the canonical Base+Consensus ecosystem; all variants use identical monthly folds and global ranking.", "", "## Global metrics", "", glob.round(3).to_string(index=False), "", "## Stability", "", stability.round(3).to_string(index=False)]
    (output_dir / "TP6_SL4_DOWNSTREAM_GAM_MATCHED_OOF_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": len(pred), "global_metric_rows": len(glob)}, indent=2))
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(output_dir=args.output_dir, control_dir=args.control_dir)
