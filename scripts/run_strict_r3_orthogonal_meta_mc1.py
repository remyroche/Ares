#!/usr/bin/env python3
"""Evaluate retained orthogonal-meta scores as target-free MC1 inputs.

This is intentionally an *experimental* mapper.  It leaves the frozen live
MC1 artifact untouched.  The new five-head score coordinates, their
disagreement, and their deltas from the incumbent consensus are target-free
model outputs; rich-policy outcomes are joined only after those panels have
been assembled for strict prequential MC1 fitting and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402


SCHEMA = "strict_r3_orthogonal_meta_mc1_v1"
SEED = 1729
TRAIN_MONTHS = 6
THRESHOLD_BPS = 30.0
SCORE_MONTHS = parent.SCORE_MONTHS
EVALUATION_PERIODS = parent.EVALUATION_PERIODS
PROHIBITED_META_LABEL_COLUMNS = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "policy_cost_bps", "semantic_path_valid", "semantic_sequence", "semantic_speed_bin",
    "semantic_persistence_bin", "semantic_pre_adverse_bin", "semantic_policy_conversion_bin",
    "semantic_exit_reason", "semantic_composite", "semantic_tbm_event",
})


def _progress(out: Path, **payload: object) -> None:
    """Persist a small append-only stage receipt for long offline replays."""
    line = json.dumps(payload, sort_keys=True)
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _robust_mean(values: Sequence[float], trim: float = .10) -> float:
    values = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    if not len(values):
        return float("nan")
    count = int(math.floor(len(values) * trim))
    values = values[count:len(values) - count] if count and len(values) > 2 * count else values
    return float(values.mean())


def _score_bands(frame: pd.DataFrame) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "final_score"]].copy()
    work["__pos__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["score_band"] = np.minimum(9, (10.0 * (rank + .5) / count).astype(np.int8))
    return work.sort_values("__pos__", kind="stable")["score_band"].to_numpy(np.int8)


def _candidate_feature_names(arms: Sequence[str]) -> tuple[str, ...]:
    features = list(parent.MC1_FEATURES)
    for arm in arms:
        token = arm.lower()
        features.extend((
            f"om__{token}__consensus_rank", f"om__{token}__head_rank_std",
            f"om__{token}__delta_parent_consensus",
        ))
        features.extend(f"om__{token}__{head}_rank" for head in (
            "h1_raw_residual", "h2_base_query_geometry", "h3_state_transition",
            "h4_support_ood_disagreement", "h5_compact_raw_control",
        ))
    return tuple(features)


def _load_family(
    p2_root: Path,
    funnel_root: Path,
    family: str,
    arms: Sequence[str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    parent_columns = (
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
        "enhanced_base_bps", *parent.MC1_FEATURES,
    )
    for month in SCORE_MONTHS:
        token = f"{month:%Y-%m}"
        source = p2_root / "target_free_scores" / family / f"month={token}.parquet"
        if not source.exists():
            continue
        frame = pd.read_parquet(source, columns=parent_columns)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        for arm in arms:
            score_path = funnel_root / "target_free_scores" / arm / f"month={token}.parquet"
            if not score_path.exists():
                raise FileNotFoundError(score_path)
            fields = [
                "candidate_id", "orthogonal_consensus_rank", "orthogonal_head_rank_std",
                "h1_raw_residual_rank", "h2_base_query_geometry_rank", "h3_state_transition_rank",
                "h4_support_ood_disagreement_rank", "h5_compact_raw_control_rank",
            ]
            meta = pd.read_parquet(score_path, columns=fields)
            prefix = f"om__{arm.lower()}__"
            meta = meta.rename(columns={
                "orthogonal_consensus_rank": f"{prefix}consensus_rank",
                "orthogonal_head_rank_std": f"{prefix}head_rank_std",
                "h1_raw_residual_rank": f"{prefix}h1_raw_residual_rank",
                "h2_base_query_geometry_rank": f"{prefix}h2_base_query_geometry_rank",
                "h3_state_transition_rank": f"{prefix}h3_state_transition_rank",
                "h4_support_ood_disagreement_rank": f"{prefix}h4_support_ood_disagreement_rank",
                "h5_compact_raw_control_rank": f"{prefix}h5_compact_raw_control_rank",
            })
            frame = frame.merge(meta, on="candidate_id", how="left", validate="one_to_one")
            frame[f"{prefix}delta_parent_consensus"] = (
                pd.to_numeric(frame[f"{prefix}consensus_rank"], errors="coerce")
                - pd.to_numeric(frame["conditional_consensus_rank"], errors="coerce")
            )
        pieces.append(frame)
    if not pieces:
        raise FileNotFoundError(f"no {family} source scores under {p2_root}")
    output = pd.concat(pieces, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError(f"{family}: duplicated target-free candidate identities")
    return output


def _fit_mc1(train: pd.DataFrame, features: Sequence[str]):
    fit = train.copy()
    fit["score_band"] = _score_bands(fit)
    fit["day"] = fit["__decision_ts__"].dt.normalize()
    selected = []
    for _, part in fit.groupby("day", sort=True):
        part = part.sort_values(["__decision_ts__", "final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
        selected.append(pd.concat((part.head(50), part.iloc[50:].sample(min(250, max(0, len(part) - 50)), random_state=SEED))))
    work = pd.concat(selected, ignore_index=True)
    target = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    low, high = target.quantile([.02, .98])
    work["target"] = target.clip(low, high)
    if len(work) > 50_000:
        work = work.sample(50_000, random_state=SEED)
    medians = work.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
    matrix = work.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").fillna(medians)
    model = HistGradientBoostingRegressor(
        max_depth=2, max_iter=80, learning_rate=.04, l2_regularization=20.0,
        min_samples_leaf=100, random_state=SEED,
    ).fit(matrix, work["target"])
    global_mean = _robust_mean(work["target"])
    curve = np.full(10, global_mean, dtype=float)
    for band, part in work.groupby("score_band", sort=True):
        mean, std, count = float(part["target"].mean()), max(float(part["target"].std(ddof=0)), 1.0), len(part)
        precision, prior = count / (std * std + 1.0), 80.0 / (250.0 ** 2)
        curve[int(band)] = (precision * mean + prior * global_mean) / (precision + prior)
    curve = -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -curve)
    return model, medians, curve, (float(low), float(high))


def _predictions(frame: pd.DataFrame, features: Sequence[str], family: str, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["score_band"] = _score_bands(work)
    output: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for month in SCORE_MONTHS:
        if month < pd.Timestamp("2025-10-01T00:00:00Z"):
            continue
        end = _month_end(month)
        train_start = month - pd.DateOffset(months=TRAIN_MONTHS)
        train = work.loc[
            work["__decision_ts__"].ge(train_start) & work["__decision_ts__"].lt(month)
            & work["policy_path_valid"].fillna(False).astype(bool)
            & work["policy_label_available_ts"].lt(month)
            & np.isfinite(pd.to_numeric(work["policy_net_bps"], errors="coerce"))
        ].copy()
        held = work.loc[work["__decision_ts__"].ge(month) & work["__decision_ts__"].lt(end)].copy()
        if len(train) < 5_000 or held.empty:
            audit.append({"family": family, "month": f"{month:%Y-%m}", "status": "insufficient", "train_rows": int(len(train)), "held_rows": int(len(held))})
            continue
        model, medians, curve, clip = _fit_mc1(train, features)
        matrix = held.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").fillna(medians)
        held["static_expected_bps"] = model.predict(matrix)
        shifts: dict[pd.Timestamp, float] = {}
        for day in pd.date_range(month.normalize(), (end - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC"):
            history = work.loc[
                work["__decision_ts__"].ge(day - pd.Timedelta(days=21)) & work["__decision_ts__"].lt(day)
                & work["policy_path_valid"].fillna(False).astype(bool)
                & work["policy_label_available_ts"].lt(day)
                & np.isfinite(pd.to_numeric(work["policy_net_bps"], errors="coerce"))
            ]
            residual = pd.to_numeric(history["policy_net_bps"], errors="coerce").to_numpy(float) - curve[history["score_band"].to_numpy(int)]
            shifts[day] = _robust_mean(residual) if len(residual) else 0.0
        held["recent_shift_bps"] = held["__decision_ts__"].dt.normalize().map(shifts).fillna(0.0)
        held["mc1_expected_bps"] = held["static_expected_bps"] + held["recent_shift_bps"]
        held["mc1_family"] = family
        output.append(held)
        audit.append({"family": family, "month": f"{month:%Y-%m}", "status": "scored", "train_rows": int(len(train)), "held_rows": int(len(held)), "clip_low": clip[0], "clip_high": clip[1]})
    prediction = pd.concat(output, ignore_index=True)
    prediction.to_parquet(out / f"{family}_experimental_mc1_predictions.parquet", index=False, compression="zstd")
    return prediction, pd.DataFrame(audit)


def _combine(current: pd.DataFrame, bcf: pd.DataFrame) -> pd.DataFrame:
    labels = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    current_keep = ["candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed", "final_score", "mc1_expected_bps", *labels[1:]]
    left = current.loc[:, current_keep].rename(columns={"final_score": "current_final_score", "mc1_expected_bps": "current_mc1_expected_bps"})
    right = bcf.loc[:, ["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"]].rename(columns={"final_score": "bcf_final_score", "mc1_expected_bps": "bcf_mc1_expected_bps"})
    out = left.merge(right, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    out["__symbol__"] = out["candidate_id"].astype(str).str.split("|", n=1, expand=True)[0]
    return out


def _live_baseline(current_path: Path, bcf_path: Path, policy: pd.DataFrame) -> pd.DataFrame:
    """Load the immutable current-live control on the canonical policy ledger.

    The archived MC1 receipts carry a historical outcome snapshot.  A matched
    challenger/control replay must instead attach both score families to the
    *same* canonical reconciled policy materialisation; otherwise an apparent
    model delta can be entirely caused by different exits or label coverage.
    """
    fields = ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score", "mc1_expected_bps"]
    current = pd.read_parquet(current_path, columns=fields)
    bcf = pd.read_parquet(bcf_path, columns=["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"])
    for frame in (current, bcf):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    current["enhanced_base_routed"] = True
    current = current.rename(columns={"final_score": "current_final_score", "mc1_expected_bps": "current_mc1_expected_bps"})
    bcf = bcf.rename(columns={"final_score": "bcf_final_score", "mc1_expected_bps": "bcf_mc1_expected_bps"})
    joined = current.merge(bcf, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    return joined.merge(policy, on="candidate_id", how="left", validate="one_to_one")


def _metrics(frame: pd.DataFrame, label: str, period: str, out: Path) -> dict[str, object]:
    # Parent replay adapter is the current controlled portfolio contract.
    return parent._portfolio_metrics(frame, label, period, out)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.parquet")) if path.is_dir() else [path]:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def run(*, p2_root: Path, funnel_root: Path, policy_path: Path, live_current: Path, live_bcf: Path, out: Path, arms: Sequence[str]) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    policy_columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    policy = pd.read_parquet(policy_path, columns=policy_columns)
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    _progress(out, stage="policy_loaded", rows=int(len(policy)))
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy labels duplicate candidate IDs")
    features = _candidate_feature_names(arms)
    family_predictions: dict[str, pd.DataFrame] = {}
    audits: list[pd.DataFrame] = []
    for family in ("current", "bcf"):
        _progress(out, stage="mc1_family_start", family=family)
        target_free = _load_family(p2_root, funnel_root, family, arms)
        leaked = PROHIBITED_META_LABEL_COLUMNS.intersection(target_free.columns)
        if leaked:
            raise AssertionError(f"{family}: outcome leaked into MC1 feature panel: {sorted(leaked)}")
        panel = target_free.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        prediction, audit = _predictions(panel, features, family, out)
        family_predictions[family] = prediction
        audits.append(audit)
        _progress(out, stage="mc1_family_done", family=family, rows=int(len(prediction)))
    current, bcf = family_predictions["current"], family_predictions["bcf"]
    _progress(out, stage="combine_start")
    combined = _combine(current, bcf)
    _progress(out, stage="baseline_start")
    baseline = _live_baseline(live_current, live_bcf, policy)
    _progress(out, stage="baseline_done", rows=int(len(baseline)))
    baseline_ids = pd.Index(baseline["candidate_id"].astype(str).unique())
    # A delta is meaningful only when both systems are replayed on exactly the
    # same frozen candidate identities.  The old score family contains a
    # small number of identities without an O3 score, so filter the control as
    # well as the challenger to this intersection.
    experimental_ids = pd.Index(combined["candidate_id"].astype(str).unique())
    common_ids = baseline_ids.intersection(experimental_ids, sort=False)
    matched = combined.loc[combined["candidate_id"].astype(str).isin(common_ids)].copy()
    baseline = baseline.loc[baseline["candidate_id"].astype(str).isin(common_ids)].copy()
    if matched.empty:
        raise AssertionError("experimental stack has no identities shared with the current-live control")
    results = []
    for period, (start, end) in EVALUATION_PERIODS.items():
        _progress(out, stage="portfolio_period_start", period=period)
        part = combined.loc[combined["__decision_ts__"].ge(start) & combined["__decision_ts__"].lt(end)].copy()
        # The parent adapter enforces dual MC1 admission and the single global
        # constrained portfolio.  Its priority remains BCF-family mapped EV.
        results.append(_metrics(part, "orthogonal_meta_full_coverage_only", period, out))
        matched_part = matched.loc[matched["__decision_ts__"].ge(start) & matched["__decision_ts__"].lt(end)].copy()
        results.append(_metrics(matched_part, "orthogonal_meta_matched_stack", period, out))
        control = baseline.loc[baseline["__decision_ts__"].ge(start) & baseline["__decision_ts__"].lt(end)].copy()
        results.append(_metrics(control, "current_live_baseline", period, out))
        _progress(out, stage="portfolio_period_done", period=period)
    metrics = pd.DataFrame(results)
    metrics.to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
    left = metrics.loc[metrics["arm"].eq("current_live_baseline")].set_index("period")
    right = metrics.loc[metrics["arm"].eq("orthogonal_meta_matched_stack")].set_index("period")
    common = left.index.intersection(right.index)
    delta = pd.DataFrame({"period": common})
    for field in (
        "accepted_rows", "realised_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised",
        "worst_month_bps", "worst_week_bps", "max_drawdown",
    ):
        if field in left.columns and field in right.columns:
            delta[f"delta_{field}"] = right.loc[common, field].to_numpy(float) - left.loc[common, field].to_numpy(float)
    delta.to_parquet(out / "delta_vs_current_live_baseline.parquet", index=False, compression="zstd")
    audit = pd.concat(audits, ignore_index=True)
    audit.to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline challenger only; frozen live MC1/current stack unchanged",
        "arms": list(arms), "mc1_features": list(features),
        "comparison_population": {
            "baseline_source_ids": int(len(baseline_ids)), "common_ids": int(len(common_ids)),
            "experimental_full_rows": int(len(combined)), "experimental_matched_rows": int(len(matched)),
            "baseline_matched_rows": int(len(baseline)),
            "delta_definition": "orthogonal_meta_matched_stack minus current_live_baseline on exact common candidate IDs",
        },
        "sources": {"p2_root": str(p2_root), "funnel_root": str(funnel_root), "policy_path": str(policy_path), "live_current": str(live_current), "live_bcf": str(live_bcf)},
        "source_hashes": {"p2_root": _sha256(p2_root), "funnel_root": _sha256(funnel_root), "policy_path": _sha256(policy_path), "live_current": _sha256(live_current), "live_bcf": _sha256(live_bcf)},
        "causality": {
            "new_meta_inputs": "strict-OOF target-free five-head ranks/dispersion/deltas",
            "mc1_fit": "six-month chronological train with resolved policy labels before held month",
            "shift": "prior 21 calendar days of fully resolved labels only",
            "admission": "dual current and BCF experimental MC1 expected EV >=30 bps",
            "portfolio": "same parent global constrained replay; BCF experimental mapped EV priority",
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-root", type=Path, required=True)
    parser.add_argument("--funnel-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--live-current", type=Path, required=True)
    parser.add_argument("--live-bcf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", required=True, help="comma-separated retained OOF arms")
    args = parser.parse_args()
    run(
        p2_root=args.p2_root, funnel_root=args.funnel_root, policy_path=args.policy_path,
        live_current=args.live_current, live_bcf=args.live_bcf, out=args.out,
        arms=tuple(args.arms.split(",")),
    )


if __name__ == "__main__":
    main()
