#!/usr/bin/env python3
"""Causal, MC1_d2-only admission ablations.

This producer deliberately leaves the sealed live mapper and its bundle alone.
It recreates the *research* MC1_d2 contract with chronological monthly static
fits, then changes exactly one of: residual-shift cadence, target-free
candidate domain, or an additive agreement feature.  It contains no R5 input,
fallback, blend, target, or comparison.

The purpose of the first stage is control integrity.  A challenger is not
eligible for target/loss or LambdaRank work until the monthly control exhibits
the expected MC1-style economics and its causal ledger invariants pass.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_lockstep_history_long_2024apr_jul2026_"
    "strictfull_prior28_optimizedpolicy_20260813_v2/"
    "walkforward_scored_label_ledger.parquet"
)
DEFAULT_PREPARED = ROOT / (
    "data_perp/artifacts/strict_r3_mc1_admission_ablation_v2_prepared_"
    "20260816_v1/candidate_static_panel.parquet"
)
CORE = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)
ADDITIVE = ("agr_rank_iqr", "agr_frac_far_10sd", "agr_head_mean")
SEED = 1729
WINDOW = pd.Timedelta(days=21)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _robust_mean(values: Iterable[float], trim: float = .10) -> float:
    data = np.asarray(list(values), dtype=float)
    data = np.sort(data[np.isfinite(data)])
    if not len(data):
        return float("nan")
    count = int(math.floor(trim * len(data)))
    if count and len(data) > 2 * count:
        data = data[count:-count]
    return float(np.mean(data))


def _score_bands(frame: pd.DataFrame) -> np.ndarray:
    """Timestamp-local score deciles from the complete candidate universe."""
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "final_score"]].copy()
    work["__order__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable", na_position="last",
    )
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["score_band"] = np.minimum(9, ((rank - .5) / size * 10.0).astype(np.int8))
    return work.sort_values("__order__", kind="stable")["score_band"].to_numpy(np.int8)


def _pool_flag(frame: pd.DataFrame, field: str, fraction: float = .30) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__order__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", field, "candidate_id"],
        ascending=[True, False, True], kind="stable", na_position="last",
    )
    position = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    work["__keep__"] = position < np.maximum(1, np.ceil(count * fraction).astype(int))
    return work.sort_values("__order__", kind="stable")["__keep__"].to_numpy(bool)


def _load_panel(ledger_path: Path, prepared_path: Path) -> pd.DataFrame:
    """Join the complete-universe score-band contract to the compact panel.

    The prepared panel already has immutable causal agreement aggregates and
    target-free routing masks.  Only the score band is rebuilt from the full
    source universe, before any pool restriction, to prevent a narrowed pool
    from changing the residual-shift coordinate.
    """
    ledger_columns = [
        "candidate_id", "__decision_ts__", "final_score", "policy_label_available_ts",
        "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m",
    ]
    universe = pd.read_parquet(ledger_path, columns=ledger_columns)
    universe["__decision_ts__"] = pd.to_datetime(universe["__decision_ts__"], utc=True)
    universe["policy_label_available_ts"] = pd.to_datetime(
        universe["policy_label_available_ts"], utc=True,
    )
    universe["score_band"] = _score_bands(universe)
    universe["pool_consensus30_full"] = _pool_flag(universe, "final_score")
    # The historical all-universe candidate identity comes only from this
    # source.  The compact panel is a target-free subset with persistent heads.
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "policy_exit_bar_15m", "pool_base30", "pool_consensus30", "pool_union30",
        *CORE, *ADDITIVE,
    ]
    available = set(pq.ParquetFile(prepared_path).schema_arrow.names)
    missing = sorted(set(columns).difference(available))
    if missing:
        raise ValueError(f"prepared MC1 panel is missing required fields: {missing}")
    panel = pd.read_parquet(prepared_path, columns=columns)
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True)
    panel["policy_label_available_ts"] = pd.to_datetime(panel["policy_label_available_ts"], utc=True)
    if panel.candidate_id.duplicated().any() or universe.candidate_id.duplicated().any():
        raise ValueError("candidate identity must be unique")
    lookup = universe.set_index("candidate_id")["score_band"]
    panel["score_band"] = panel.candidate_id.map(lookup)
    if panel.score_band.isna().any():
        raise ValueError("prepared candidate lacks a complete-universe score band")
    panel["score_band"] = panel.score_band.astype(np.int8)
    # Model output availability is intentionally the existing target-free
    # candidate panel.  The runner compares its union and consensus routes;
    # it never uses label validity for membership.
    panel["route_union30"] = panel.pool_union30.fillna(False).astype(bool)
    panel["route_consensus30"] = panel.pool_consensus30.fillna(False).astype(bool)
    panel["day"] = panel["__decision_ts__"].dt.normalize()
    panel["bucket_6h"] = panel["__decision_ts__"].dt.floor("6h")
    del universe
    gc.collect()
    return panel


def _day_balanced(frame: pd.DataFrame) -> pd.DataFrame:
    """Original MC1-style deterministic daily substrate without held rows."""
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("day", sort=True):
        ordered = group.sort_values(
            ["__decision_ts__", "final_score", "candidate_id"],
            ascending=[True, False, True], kind="stable",
        ).copy()
        ordered["__rank_n__"] = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
        top = ordered.loc[ordered["__rank_n__"].le(50)]
        rest = ordered.loc[ordered["__rank_n__"].gt(50)]
        if len(rest):
            rest = rest.sample(min(250, len(rest)), random_state=SEED)
        pieces.append(pd.concat([top, rest], ignore_index=False))
    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["policy_label_available_ts", "candidate_id"], kind="stable",
    )


def _structural_curve(train: pd.DataFrame) -> np.ndarray:
    global_mean = _robust_mean(train["policy_net_bps"])
    curve = np.full(10, global_mean, dtype=float)
    for band, group in train.groupby("score_band", sort=True):
        values = pd.to_numeric(group["policy_net_bps"], errors="coerce").dropna().to_numpy(float)
        if not len(values):
            continue
        mean = float(np.mean(values))
        std = max(float(np.std(values)), 1.0)
        precision = len(values) / (std * std + 1.0)
        prior_precision = 80.0 / (250.0**2)
        curve[int(band)] = (
            precision * mean + prior_precision * global_mean
        ) / (precision + prior_precision)
    return -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -curve)


def _fit_hgb(train: pd.DataFrame, features: Sequence[str]) -> tuple[HistGradientBoostingRegressor, pd.Series, np.ndarray, tuple[float, float]]:
    clean = train.dropna(subset=["policy_net_bps"]).copy()
    medians = clean.loc[:, features].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x = clean.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(medians)
    y = pd.to_numeric(clean["policy_net_bps"], errors="coerce")
    low, high = y.quantile([.02, .98]).to_numpy(float)
    y = y.clip(low, high)
    if len(x) > 50_000:
        take = x.sample(50_000, random_state=SEED).index
        x = x.loc[take]
        y = y.loc[take]
    model = HistGradientBoostingRegressor(
        max_depth=2, max_iter=80, learning_rate=.04, l2_regularization=20.0,
        min_samples_leaf=100, random_state=SEED,
    ).fit(x, y)
    return model, medians, _structural_curve(clean), (float(low), float(high))


def _causal_shifts(
    resolved: pd.DataFrame, curve: np.ndarray, buckets: pd.DatetimeIndex, cadence: str,
) -> pd.Series:
    """Prior-resolved residual location adjustment for daily or 6h decisions.

    A result at bucket ``t`` only consumes labels available strictly before
    ``t``.  The history window is based on decision timestamps, as in MC1_d2.
    """
    if cadence not in {"1d", "6h"}:
        raise ValueError(f"unsupported cadence: {cadence}")
    events = resolved.loc[
        resolved.policy_path_valid.fillna(False).astype(bool)
        & resolved.policy_net_bps.notna()
    ].loc[:, ["__decision_ts__", "policy_label_available_ts", "policy_net_bps", "score_band"]].copy()
    events["available_bucket"] = events["policy_label_available_ts"].dt.floor(cadence)
    events["residual"] = (
        pd.to_numeric(events["policy_net_bps"], errors="coerce").to_numpy(float)
        - curve[events["score_band"].to_numpy(int)]
    )
    grouped = {
        key: group.loc[:, ["__decision_ts__", "residual"]].copy()
        for key, group in events.groupby("available_bucket", sort=True)
    }
    keys = deque(sorted(grouped))
    active: deque[pd.DataFrame] = deque()
    answer: dict[pd.Timestamp, float] = {}
    for bucket in sorted(pd.DatetimeIndex(buckets).unique()):
        while keys and keys[0] < bucket:
            active.append(grouped[keys.popleft()])
        cutoff = bucket - WINDOW
        while active and active[0]["__decision_ts__"].max() < cutoff:
            active.popleft()
        if active:
            values = np.concatenate([
                part.loc[part["__decision_ts__"].ge(cutoff), "residual"].to_numpy(float)
                for part in active
            ])
            answer[bucket] = _robust_mean(values)
        else:
            answer[bucket] = float("nan")
    return pd.Series(answer, dtype=float)


def _auction(frame: pd.DataFrame, expected: str) -> pd.DataFrame:
    """Target-free 2-new/8-concurrent auction, ranked solely by final_score."""
    work = frame.loc[pd.to_numeric(frame[expected], errors="coerce").ge(50.0)].copy()
    work = work.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    active: list[pd.Timestamp] = []
    accepted: list[bool] = []
    for decision, group in work.groupby("__decision_ts__", sort=True):
        active = [exit_ts for exit_ts in active if exit_ts > decision]
        new_entries = 0
        for row in group.itertuples(index=False):
            bars = pd.to_numeric(pd.Series([getattr(row, "policy_exit_bar_15m")]), errors="coerce").iloc[0]
            holding = max(48.0, float(bars) if np.isfinite(bars) else 48.0)
            allowed = new_entries < 2 and len(active) < 8
            accepted.append(allowed)
            if allowed:
                active.append(decision + pd.Timedelta(minutes=15 * holding))
                new_entries += 1
    work["portfolio_accepted"] = accepted
    return work


def _metrics(frame: pd.DataFrame, expected: str, arm: str, period: str) -> dict[str, object]:
    valid = frame.policy_path_valid.fillna(False).astype(bool) & frame.policy_net_bps.notna()
    admitted = frame.loc[pd.to_numeric(frame[expected], errors="coerce").ge(50.0)]
    accepted = _auction(frame, expected)
    realised = accepted.loc[
        accepted.portfolio_accepted & accepted.policy_path_valid.fillna(False).astype(bool)
        & accepted.policy_net_bps.notna()
    ].copy()
    y = pd.to_numeric(realised.policy_net_bps, errors="coerce")
    p = pd.to_numeric(realised[expected], errors="coerce")
    slope = intercept = ic = float("nan")
    admitted_valid = admitted.loc[
        admitted.policy_path_valid.fillna(False).astype(bool) & admitted.policy_net_bps.notna()
    ].copy()
    admitted_y = pd.to_numeric(admitted_valid.policy_net_bps, errors="coerce")
    admitted_p = pd.to_numeric(admitted_valid[expected], errors="coerce")
    if len(admitted_valid) >= 20 and admitted_p.nunique() > 1:
        slope, intercept = np.polyfit(admitted_p, admitted_y, 1)
        def _within_timestamp_ic(group: pd.DataFrame) -> float:
            score = pd.to_numeric(group[expected], errors="coerce")
            outcome = pd.to_numeric(group["policy_net_bps"], errors="coerce")
            valid_group = score.notna() & outcome.notna()
            score, outcome = score.loc[valid_group], outcome.loc[valid_group]
            if len(score) < 3 or score.nunique() < 2 or outcome.nunique() < 2:
                return float("nan")
            return float(score.corr(outcome, method="spearman"))
        ic = float(admitted_valid.groupby("__decision_ts__", sort=False).apply(
            _within_timestamp_ic, include_groups=False,
        ).dropna().mean())
    monthly = realised.groupby(realised.__decision_ts__.dt.strftime("%Y-%m"), sort=True).policy_net_bps.mean()
    weekly = realised.groupby(realised.__decision_ts__.dt.strftime("%G-W%V"), sort=True).policy_net_bps.mean()
    contested = admitted.groupby("__decision_ts__", sort=False).filter(lambda group: len(group) > 2)
    if len(contested):
        picked_ids = set(_auction(contested, expected).query("portfolio_accepted").candidate_id)
        contest_pick = contested.loc[contested.candidate_id.isin(picked_ids) & contested.policy_path_valid.fillna(False)]
        contest_reject = contested.loc[~contested.candidate_id.isin(picked_ids) & contested.policy_path_valid.fillna(False)]
        picked_ev = float(contest_pick.policy_net_bps.mean()) if len(contest_pick) else float("nan")
        rejected_ev = float(contest_reject.policy_net_bps.mean()) if len(contest_reject) else float("nan")
    else:
        picked_ev = rejected_ev = float("nan")
    return {
        "arm": arm, "period": period, "candidate_rows": int(len(frame)),
        "valid_candidate_rows": int(valid.sum()), "admitted_rows": int(len(admitted)),
        "portfolio_selected_rows": int(len(realised)),
        "selected_label_coverage": float(len(realised) / max(1, int(accepted.portfolio_accepted.sum()))),
        "portfolio_net_ev_bps": float(y.mean()) if len(y) else float("nan"),
        "portfolio_net_sum_bps": float(y.sum()), "within_admission_ic": ic,
        "calibration_slope": float(slope), "calibration_intercept": float(intercept),
        "worst_month_bps": float(monthly.min()) if len(monthly) else float("nan"),
        "worst_week_bps": float(weekly.min()) if len(weekly) else float("nan"),
        "positive_month_fraction": float(monthly.gt(0).mean()) if len(monthly) else float("nan"),
        "contested_selected_bps": picked_ev, "contested_rejected_bps": rejected_ev,
    }


def _arms() -> list[dict[str, object]]:
    return [
        {"id": "d2_core_daily_union", "features": list(CORE), "cadence": "1d", "score_route": "route_union30", "fit_route": "route_union30"},
        {"id": "d2_core_6h_union", "features": list(CORE), "cadence": "6h", "score_route": "route_union30", "fit_route": "route_union30"},
        # Candidate-scope test: same static fit, narrower target-free scoring
        # envelope.  This is the clean answer to whether consensus-top30 adds
        # useful opportunities without changing how MC1 learns its value map.
        {"id": "d2_core_6h_consensus", "features": list(CORE), "cadence": "6h", "score_route": "route_consensus30", "fit_route": "route_union30"},
        {"id": "d2_core_6h_consensus_fitconsensus", "features": list(CORE), "cadence": "6h", "score_route": "route_consensus30", "fit_route": "route_consensus30"},
        {"id": "d2_iqr_6h_consensus", "features": [*CORE, "agr_rank_iqr"], "cadence": "6h", "score_route": "route_consensus30", "fit_route": "route_union30"},
        {"id": "d2_iqr_far_6h_consensus", "features": [*CORE, "agr_rank_iqr", "agr_frac_far_10sd"], "cadence": "6h", "score_route": "route_consensus30", "fit_route": "route_union30"},
        {"id": "d2_agreement9_6h_consensus", "features": [*CORE, *ADDITIVE], "cadence": "6h", "score_route": "route_consensus30", "fit_route": "route_union30"},
    ]


def _run_arm(panel: pd.DataFrame, arm: dict[str, object], *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    features = list(arm["features"])
    score_route = str(arm["score_route"])
    fit_route = str(arm["fit_route"])
    blocks: list[pd.DataFrame] = []
    for fold_start in pd.date_range(start, end, freq="MS", tz="UTC"):
        fold_end = min(fold_start + pd.offsets.MonthBegin(1), end)
        if fold_start >= end:
            break
        fit = panel.loc[
            panel.policy_label_available_ts.lt(fold_start)
            & panel.policy_path_valid.fillna(False).astype(bool)
            & panel.policy_net_bps.notna()
            & panel[fit_route]
        ].copy()
        held = panel.loc[
            panel.__decision_ts__.ge(fold_start) & panel.__decision_ts__.lt(fold_end) & panel[score_route]
        ].copy()
        if len(fit) < 5_000 or held.empty:
            continue
        substrate = _day_balanced(fit)
        model, medians, curve, _ = _fit_hgb(substrate, features)
        x = held.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(medians)
        held["static_expected_bps"] = model.predict(x)
        cadence = str(arm["cadence"])
        index = held["__decision_ts__"].dt.floor(cadence)
        shifts = _causal_shifts(fit, curve, pd.DatetimeIndex(index.unique()), cadence)
        held["recent_shift_bps"] = index.map(shifts).to_numpy(float)
        held["mc1_expected_bps"] = held.static_expected_bps + held.recent_shift_bps
        held["fold_start"] = fold_start
        held["arm"] = str(arm["id"])
        blocks.append(held.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__", "policy_label_available_ts",
            "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m", "final_score",
            "score_band", "static_expected_bps", "recent_shift_bps", "mc1_expected_bps",
            "fold_start", "arm",
        ]])
        del fit, held, substrate, model, x
        gc.collect()
    if not blocks:
        return panel.iloc[0:0].copy()
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--prediction-dir", type=Path, default=None,
                        help="recompute metrics from immutable prediction artifacts only")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    start, end = _utc(args.start), _utc(args.end)
    selected = _arms()
    if args.arms:
        wanted = set(args.arms)
        selected = [arm for arm in selected if str(arm["id"]) in wanted]
        missing = wanted.difference(str(arm["id"]) for arm in selected)
        if missing:
            raise ValueError(f"unknown arm(s): {sorted(missing)}")
    metric_rows: list[dict[str, object]] = []
    if args.prediction_dir is None:
        panel = _load_panel(args.ledger, args.prepared)
        if not panel.side_name.astype(str).str.lower().eq("long").all():
            raise ValueError("this ablation is intentionally long-only")
    for arm in selected:
        print(json.dumps({"event": "arm_start", "arm": arm["id"]}), flush=True)
        if args.prediction_dir is None:
            prediction = _run_arm(panel, arm, start=start, end=end)
        else:
            source = args.prediction_dir / f"predictions_{arm['id']}.parquet"
            if not source.exists():
                raise FileNotFoundError(f"missing immutable prediction artifact: {source}")
            prediction = pd.read_parquet(source)
            prediction["__decision_ts__"] = pd.to_datetime(prediction["__decision_ts__"], utc=True)
            prediction["policy_label_available_ts"] = pd.to_datetime(
                prediction["policy_label_available_ts"], utc=True,
            )
        for year in (2025, 2026):
            part = prediction.loc[prediction.__decision_ts__.dt.year.eq(year)]
            if len(part):
                metric_rows.append(_metrics(part, "mc1_expected_bps", str(arm["id"]), str(year)))
        if args.prediction_dir is None:
            prediction.to_parquet(
                args.out_dir / f"predictions_{arm['id']}.parquet", index=False, compression="zstd",
            )
        print(json.dumps({"event": "arm_complete", "arm": arm["id"], "rows": len(prediction)}), flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    metrics.to_csv(args.out_dir / "portfolio_metrics.csv", index=False)
    manifest = {
        "schema": "strict_r3_mc1_d2_controlled_ablation_v1",
        "status": "complete",
        "purpose": "offline MC1_d2-only ablations; sealed live bundle untouched",
        "explicit_exclusions": ["R5", "live_state", "exchange_io", "held-out outcomes as inputs"],
        "ledger": str(args.ledger), "ledger_sha256": _sha256(args.ledger),
        "prepared": str(args.prepared), "prepared_sha256": _sha256(args.prepared),
        "prediction_source": str(args.prediction_dir) if args.prediction_dir else None,
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "base_contract": {
            "model": "HistGradientBoostingRegressor depth=2 iter=80 lr=.04 l2=20 min_leaf=100 seed=1729",
            "features": list(CORE), "shift": "21-day 10%-trimmed score-band residual; strictly prior-resolved",
            "threshold": "+50 bps", "auction": "final_score only; 2 new / 8 concurrent",
        },
        "arms": selected,
        "causality": {
            "candidate_domain": "target-free pool flags are created before labels are consulted",
            "score_band": "derived from complete timestamp universe before prepared-pool restriction",
            "training": "policy_label_available_ts < monthly fold start",
            "shift": "policy_label_available_ts < current daily/6h bucket",
            "metrics": "outcomes join only after predictions and portfolio selection",
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
