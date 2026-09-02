#!/usr/bin/env python3
"""Strict-OOF residual screen for P8U market-state/Kalman representations.

The state features are timestamp-global, so the screen evaluates them at the
natural time × Base-band granularity.  This prevents a repeated market value
from receiving spurious candidate-level weight while retaining the requested
conditional relationship ``feature ; residual | BaseBand`` in the top 20% of
Base ranks.  The script only opens labels after the target-free state receipt
has been checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import materialize_strict_r3_p8u_meta_base_state_v1 as base_state


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_market_state_transition_screen_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
TOP20_BANDS = 4
TOP20_START = 0.80
FEATURE_KINDS = {
    "kalman_innovation_z", "kalman_fast_minus_slow", "kalman_fast_slow_normalized",
    "direct_transition", "innovation_magnitude", "innovation_breadth",
    "innovation_dispersion", "innovation_mahalanobis",
}


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month(token: str) -> pd.Timestamp:
    return pd.Timestamp(f"{token}-01", tz="UTC")


def _months(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    result = []
    value = start
    while value < end:
        result.append(value)
        value += pd.offsets.MonthBegin(1)
    return tuple(result)


def _base_path(early: Path, later: Path, month: pd.Timestamp) -> tuple[Path, tuple[str, str]]:
    early_path = early / f"month={month:%Y-%m}.parquet"
    later_path = later / f"month={month:%Y-%m}.parquet"
    if early_path.exists():
        return early_path, ("base_score", "base_rank_ts")
    if later_path.exists():
        return later_path, ("base_score", "base_rank_ts")
    raise FileNotFoundError(f"no Base target-free panel for {month:%Y-%m}")


def _assert_target_free(path: Path) -> None:
    names = set(pd.read_parquet(path, columns=None).columns) if False else set()
    # Schema-only check avoids loading a target-free Base panel twice.
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(path).schema_arrow.names)
    forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "outcome", "label", "mfe", "mae"}
    leaked = sorted(name for name in names if name.lower() in forbidden)
    if leaked:
        raise AssertionError(f"{path}: target/outcome feature leak {leaked}")


def _read_base(early: Path, later: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for month in _months(start, end):
        path, columns = _base_path(early, later, month)
        _assert_target_free(path)
        part = pd.read_parquet(path, columns=[*IDENTITY, *columns])
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        part = part.loc[part.__decision_ts__.ge(start) & part.__decision_ts__.lt(end)].copy()
        if part.duplicated(IDENTITY).any() or not part.side_name.eq("long").all():
            raise AssertionError(f"{month:%Y-%m}: invalid Base identity")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame.duplicated(IDENTITY).any():
        raise AssertionError("Base target-free rows overlap")
    return frame


def _read_policy(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])


def _time_band_events(base: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    """Use the established strict-prequential Base residual event construction."""
    events = base_state._policy_events(base, policy)
    # The shared event helper deliberately returns only target state columns;
    # retain the immutable target-free Base rank as the conditioning variable.
    events = events.merge(base.loc[:, ["candidate_id", "base_rank_ts"]], on="candidate_id", how="left", validate="one_to_one")
    if events.base_rank_ts.isna().any():
        raise AssertionError("strict-prequential residual event lost its Base rank")
    events = events.loc[events.base_rank_ts.ge(TOP20_START)].copy()
    if events.empty:
        raise AssertionError("no top-20% strict-prequential residual events")
    events["base_band"] = np.minimum(
        TOP20_BANDS - 1,
        np.maximum(0, np.floor((events.base_rank_ts.to_numpy(float) - TOP20_START) / ((1.0 - TOP20_START) / TOP20_BANDS))),
    ).astype(np.int16)
    events["month"] = events.__decision_ts__.dt.strftime("%Y-%m")
    events["week"] = events.__decision_ts__.dt.to_period("W-SUN").astype(str)
    grouped = events.groupby(["__decision_ts__", "month", "week", "base_band"], sort=True).agg(
        residual_bps=("residual_bps", "mean"), policy_net_bps=("policy_net_bps", "mean"), n=("candidate_id", "size"),
    ).reset_index()
    return grouped


def _conditional_ic(frame: pd.DataFrame, feature: str) -> tuple[float, int]:
    total, weight, good = 0.0, 0.0, 0
    for _, part in frame.groupby("base_band", sort=True):
        sample = part.loc[np.isfinite(part[feature]) & np.isfinite(part.residual_bps)]
        if len(sample) < 12 or sample[feature].nunique() < 3:
            continue
        value = spearmanr(sample[feature], sample.residual_bps).statistic
        if np.isfinite(value):
            total += float(value) * len(sample)
            weight += len(sample)
            good += 1
    return (total / weight if weight else np.nan), good


def _bins(values: pd.Series, count: int) -> np.ndarray:
    ranked = values.rank(method="average", pct=True).to_numpy(float)
    return np.minimum(count - 1, np.floor(np.clip(ranked, 0.0, 0.999999) * count)).astype(np.int16)


def _conditional_mi(frame: pd.DataFrame, feature: str) -> float:
    """Binned I(feature; residual | Base band), weighted by time-band support."""
    total, weight = 0.0, 0.0
    for _, part in frame.groupby("base_band", sort=True):
        sample = part.loc[np.isfinite(part[feature]) & np.isfinite(part.residual_bps)]
        if len(sample) < 20 or sample[feature].nunique() < 4:
            continue
        xb, yb = _bins(sample[feature], 8), _bins(sample.residual_bps, 6)
        table = np.zeros((8, 6), dtype=float)
        np.add.at(table, (xb, yb), sample.n.to_numpy(float))
        joint = table / max(1.0, table.sum())
        px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
        valid = joint > 0.0
        value = float((joint[valid] * np.log(joint[valid] / (px @ py)[valid])).sum())
        total += value * table.sum()
        weight += table.sum()
    return total / weight if weight else np.nan


def _screen_period(frame: pd.DataFrame, features: Sequence[str], period: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for token, part in frame.groupby(period, sort=True):
        for feature in features:
            ic, bands = _conditional_ic(part, feature)
            mi = _conditional_mi(part, feature)
            rows.append({"period_kind": period, "period": str(token), "feature": feature, "conditional_ic": ic, "binned_cmi": mi, "bands_with_ic": bands, "rows": len(part), "weight": float(part.n.sum())})
    return pd.DataFrame(rows)


def _screen_from_state(
    events: pd.DataFrame, state_path: Path, features: Sequence[str], period: str, *, batch_size: int = 64,
) -> pd.DataFrame:
    """Join only a bounded feature block to residual events at one time.

    The state lattice is intentionally wide but only hourly.  Joining all
    fields to every candidate-band event creates a large avoidable copy; this
    batched equivalent has identical metrics and bounded memory.
    """
    pieces: list[pd.DataFrame] = []
    for start in range(0, len(features), batch_size):
        block = list(features[start : start + batch_size])
        state = pd.read_parquet(state_path, columns=["__decision_ts__", *block])
        state["__decision_ts__"] = pd.to_datetime(state["__decision_ts__"], utc=True, errors="raise")
        panel = events.merge(state, on="__decision_ts__", how="inner", validate="many_to_one")
        if len(panel) != len(events):
            raise AssertionError("market state missing for Base residual time-band events")
        pieces.append(_screen_period(panel, block, period))
    return pd.concat(pieces, ignore_index=True)


def _summary(monthly: pd.DataFrame, dictionary: pd.DataFrame, era: str) -> pd.DataFrame:
    grouped = monthly.groupby("feature", sort=True).agg(
        mean_conditional_ic=("conditional_ic", "mean"),
        mean_abs_conditional_ic=("conditional_ic", lambda x: float(np.nanmean(np.abs(x)))),
        mean_binned_cmi=("binned_cmi", "mean"),
        positive_ic_months=("conditional_ic", lambda x: int((x > 0).sum())),
        negative_ic_months=("conditional_ic", lambda x: int((x < 0).sum())),
        observed_months=("period", "nunique"),
        min_conditional_ic=("conditional_ic", "min"),
    ).reset_index()
    grouped["sign_consistency"] = np.maximum(grouped.positive_ic_months, grouped.negative_ic_months) / grouped.observed_months.clip(lower=1)
    grouped["era"] = era
    return grouped.merge(dictionary.drop_duplicates("feature"), on="feature", how="left", validate="one_to_one")


def _selection(train: pd.DataFrame) -> pd.DataFrame:
    work = train.copy()
    # A mild standardised composite: CMI and conditional rank association must
    # both recur.  No downstream policy result is used at this selection step.
    for name in ("mean_binned_cmi", "mean_abs_conditional_ic"):
        median = work[name].median(skipna=True)
        mad = (work[name] - median).abs().median(skipna=True)
        work[f"z_{name}"] = (work[name] - median) / max(float(mad), 1e-9)
    work["stability_score"] = work.sign_consistency * np.minimum(1.0, work.observed_months / 8.0)
    work["screen_score"] = (0.5 * work.z_mean_binned_cmi + 0.5 * work.z_mean_abs_conditional_ic) * work.stability_score
    work["eligible"] = work.observed_months.ge(6) & work.sign_consistency.ge(.60) & work.mean_binned_cmi.gt(0.0)
    return work.sort_values(["eligible", "screen_score", "feature"], ascending=[False, False, True], kind="stable").reset_index(drop=True)


def _redundancy_representatives(selection: pd.DataFrame, state_path: Path, *, limit: int = 30) -> pd.DataFrame:
    """Keep a stable representative when state encodings are nearly identical.

    Fast/slow level and normalised-level are often algebraic near-duplicates.
    They stay in the diagnostic screen, but interaction search receives one
    representative per >.95 absolute-Spearman group so beam search can spend
    capacity on distinct market mechanisms.
    """
    candidates = selection.loc[selection.eligible].head(180).copy()
    if candidates.empty:
        return candidates
    names = candidates.feature.tolist()
    values = pd.read_parquet(state_path, columns=names).rank(pct=True)
    kept: list[str] = []
    max_correlations: list[float] = []
    for name in names:
        maximum = 0.0
        for prior in kept:
            sample = pd.concat([values[name], values[prior]], axis=1).dropna()
            if len(sample) >= 100:
                maximum = max(maximum, abs(float(sample.corr(method="spearman").iloc[0, 1])))
        if maximum < .95:
            kept.append(name)
            max_correlations.append(maximum)
        if len(kept) >= limit:
            break
    result = candidates.loc[candidates.feature.isin(kept)].copy()
    order = pd.Categorical(result.feature, categories=kept, ordered=True)
    result = result.assign(_order=order).sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
    result["max_abs_spearman_to_higher_ranked_selected"] = max_correlations
    return result


def _write(root: Path, *, monthly: pd.DataFrame, weekly: pd.DataFrame, summary_train: pd.DataFrame, summary_confirm: pd.DataFrame, selected: pd.DataFrame, source: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    monthly.to_parquet(root / "monthly_conditional_metrics.parquet", index=False)
    weekly.to_parquet(root / "weekly_conditional_metrics_top30.parquet", index=False)
    summary_train.to_parquet(root / "summary_2024_2025.parquet", index=False)
    summary_confirm.to_parquet(root / "summary_2026.parquet", index=False)
    selected.to_parquet(root / "selected_top30_preprobe.parquet", index=False)
    correctness = {
        "schema": SCHEMA,
        "market_state_receipt_target_free": True,
        "base_scores_target_free_before_label_join": True,
        "policy_labels_opened_only_for_evaluation": True,
        "residual_anchor_is_strict_prequential": True,
        "top20_base_gate_applied_before_metrics": True,
        "conditional_metrics_condition_on_base_band": True,
        "no_meta_mc1_admission_portfolio_or_live_mutation": True,
    }
    _once(root / "correctness_report.json", correctness)
    _once(root / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline strict-OOF feature screen only", "source": source,
        "selected_top30_count": int(len(selected)), "correctness": correctness,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--early-base-root", required=True)
    parser.add_argument("--later-base-root", required=True)
    parser.add_argument("--policy-labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state_root, early, later, policy_path, out = (ROOT / args.state_root, ROOT / args.early_base_root, ROOT / args.later_base_root, ROOT / args.policy_labels, ROOT / args.out)
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((state_root / "correctness_report.json").read_text())
    if not all(value is True or key in {"schema", "fast_slow_pairs_predeclared"} for key, value in receipt.items()):
        raise AssertionError("market state correctness receipt is not clean")
    state_path = state_root / "market_state_hourly.parquet"
    state_times = pd.read_parquet(state_path, columns=["__decision_ts__"])
    state_times["__decision_ts__"] = pd.to_datetime(state_times["__decision_ts__"], utc=True, errors="raise")
    dictionary = pd.read_parquet(state_root / "market_state_feature_dictionary.parquet")
    dictionary = dictionary.loc[dictionary.kind.isin(FEATURE_KINDS)].copy()
    features = tuple(sorted(dictionary.feature.unique()))
    if len(features) < 300:
        raise AssertionError("unexpectedly few innovation/transition fields")
    first, last = state_times.__decision_ts__.min(), state_times.__decision_ts__.max()
    start = pd.Timestamp(year=first.year, month=first.month, day=1, tz="UTC")
    end = pd.Timestamp(year=last.year, month=last.month, day=1, tz="UTC") + pd.offsets.MonthBegin(1)
    base = _read_base(early, later, start, end)
    policy = _read_policy(policy_path)
    events = _time_band_events(base, policy)
    monthly = _screen_from_state(events, state_path, features, "month")
    train_monthly = monthly.loc[monthly.period.str[:4].astype(int).le(2025)].copy()
    confirm_monthly = monthly.loc[monthly.period.str[:4].astype(int).eq(2026)].copy()
    summary_train = _summary(train_monthly, dictionary, "2024_2025")
    summary_confirm = _summary(confirm_monthly, dictionary, "2026")
    selection = _selection(summary_train)
    selected = _redundancy_representatives(selection, state_path, limit=30)
    # Weekly stability is purposefully calculated only for predeclared top-30
    # candidates selected entirely on the 2024–25 monthly screen.
    weekly = _screen_from_state(events, state_path, selected.feature.tolist(), "week")
    _write(out, monthly=monthly, weekly=weekly, summary_train=summary_train, summary_confirm=summary_confirm, selected=selected, source={
        "state_root": str(state_root.relative_to(ROOT)), "early_base_root": str(early.relative_to(ROOT)),
        "later_base_root": str(later.relative_to(ROOT)), "policy_labels": str(policy_path.relative_to(ROOT)),
    })
    print(json.dumps({"out": str(out), "features_screened": len(features), "top30": len(selected), "time_band_rows": len(events)}, sort_keys=True))


if __name__ == "__main__":
    main()
