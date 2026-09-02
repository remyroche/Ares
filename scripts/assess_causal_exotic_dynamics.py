#!/usr/bin/env python3
"""Strict-OOF 2025 assessment of causal exotic-dynamics feature families.

This is Stage 2 of the Kalman Features Utility research protocol.  It joins
only target-free dynamic states to the immutable paired score-family ledger,
then evaluates quality, raw/residual information, conditional information,
tail behaviour and shallow family-only probes.  It deliberately performs no
live, policy, or admission mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_exotic_dynamics import FEATURE_COLUMNS


BCF = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
CURRENT = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
DEFAULT_DYNAMIC = ROOT / "data_perp/artifacts/causal_exotic_dynamics_2025train_2026confirm_20260831_v3_expanded"
DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_exotic_dynamics_assessment_2025_20260831_v2_expanded"

CORE = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)
LABELS = ("policy_path_valid", "policy_net_bps", "policy_label_available_ts")
SEED = 1729


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: object) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _family(name: str) -> str:
    return name.split("_", 1)[0].upper()


def _load_score(path: Path, prefix: str) -> pd.DataFrame:
    columns = ["candidate_id", "__decision_ts__", "__symbol__", *CORE, *LABELS, "mc1_expected_bps"]
    frame = pd.read_parquet(path, columns=columns)
    frame["candidate_id"] = frame.candidate_id.astype(str)
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    frame["policy_label_available_ts"] = _utc(frame["policy_label_available_ts"])
    renamed = {name: f"{prefix}_{name}" for name in (*CORE, "mc1_expected_bps")}
    return frame.rename(columns=renamed)


def _load_panel(dynamic: Path, bcf: Path, current: Path) -> pd.DataFrame:
    target_free = pd.read_parquet(dynamic / "target_free_candidate_intersection.parquet")
    target_free.candidate_id = target_free.candidate_id.astype(str)
    target_free["__decision_ts__"] = _utc(target_free["__decision_ts__"])
    left, right = _load_score(bcf, "bcf"), _load_score(current, "current")
    policy = left.loc[:, ["candidate_id", *LABELS]].merge(
        right.loc[:, ["candidate_id", *LABELS]], on="candidate_id", suffixes=("_b", "_c"), validate="one_to_one",
    )
    for field in LABELS:
        a, b = policy[f"{field}_b"], policy[f"{field}_c"]
        if pd.api.types.is_numeric_dtype(a):
            same = np.isclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), equal_nan=True).all()
        else:
            same = a.fillna("__null__").astype(str).equals(b.fillna("__null__").astype(str))
        if not same:
            raise AssertionError(f"paired source policy labels differ: {field}")
    policy = policy.loc[:, ["candidate_id", *[f"{field}_b" for field in LABELS]]].rename(columns={f"{field}_b": field for field in LABELS})
    left = left.drop(columns=list(LABELS))
    right = right.drop(columns=list(LABELS) + ["__decision_ts__", "__symbol__"])
    score = left.merge(right, on="candidate_id", validate="one_to_one").merge(policy, on="candidate_id", validate="one_to_one")
    panel = target_free.merge(score, on=["candidate_id", "__decision_ts__", "__symbol__"], validate="one_to_one")
    # Read only feature columns and their status from the partitioned matrix.
    dynamic_table = ds.dataset(str(dynamic / "feature_parts"), format="parquet", partitioning="hive")
    states = dynamic_table.to_table(columns=["candidate_id", "dynamic_source_status", *FEATURE_COLUMNS]).to_pandas()
    states.candidate_id = states.candidate_id.astype(str)
    if states.candidate_id.duplicated().any():
        raise AssertionError("dynamic feature partitions duplicate target-free identity")
    panel = panel.merge(states, on="candidate_id", how="left", validate="one_to_one")
    if len(panel) != len(target_free) or panel.dynamic_source_status.isna().any():
        raise AssertionError("dynamic feature matrix does not exactly preserve target-free candidate identity")
    panel["m0_expected_bps"] = (
        pd.to_numeric(panel["bcf_mc1_expected_bps"], errors="raise")
        + pd.to_numeric(panel["current_mc1_expected_bps"], errors="raise")
    ) / 2.0
    panel["m0_min_expected_bps"] = panel[["bcf_mc1_expected_bps", "current_mc1_expected_bps"]].min(axis=1)
    panel["month"] = panel["__decision_ts__"].dt.strftime("%Y-%m")
    panel["valid_label"] = panel.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(panel.policy_net_bps, errors="coerce"))
    panel["mc1_residual"] = pd.to_numeric(panel.policy_net_bps, errors="coerce") - panel.m0_expected_bps
    return panel


def _spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"x": pd.to_numeric(left, errors="coerce"), "y": pd.to_numeric(right, errors="coerce")}).dropna()
    return float(pair.x.corr(pair.y, method="spearman")) if len(pair) >= 32 else float("nan")


def _conditional_mi_proxy(target: pd.Series, feature: pd.Series, control: pd.Series) -> float:
    """Discrete CMI I(Y;F|M0), with rank bins on a deterministic sample."""
    frame = pd.DataFrame({"y": target, "x": feature, "z": control}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 512:
        return float("nan")
    if len(frame) > 50_000:
        # Candidate IDs are not needed: a fixed row stride avoids outcome-driven sampling.
        frame = frame.iloc[np.linspace(0, len(frame) - 1, 50_000, dtype=int)]
    bins = 8
    codes = []
    for field in ("y", "x", "z"):
        rank = frame[field].rank(method="first", pct=True).to_numpy(float)
        codes.append(np.minimum(bins - 1, (rank * bins).astype(int)))
    y, x, z = codes
    total = float(len(frame))
    value = 0.0
    for zi in range(bins):
        select_z = z == zi
        nz = int(select_z.sum())
        if not nz:
            continue
        for yi in range(bins):
            for xi in range(bins):
                nxyz = int(np.sum(select_z & (y == yi) & (x == xi)))
                if not nxyz:
                    continue
                nyz = int(np.sum(select_z & (y == yi)))
                nxz = int(np.sum(select_z & (x == xi)))
                value += nxyz / total * np.log((nxyz * nz) / max(nyz * nxz, 1))
    return float(value)


def _quality_and_information(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled = panel.loc[panel.valid_label].copy()
    rows, monthly = [], []
    for field in FEATURE_COLUMNS:
        values = pd.to_numeric(panel[field], errors="coerce")
        valid = values.notna()
        labelled_values = pd.to_numeric(labelled[field], errors="coerce")
        by_month = labelled.assign(_feature=labelled_values).groupby("month", sort=True).apply(
            lambda part: pd.Series({
                "raw_ic": _spearman(part["_feature"], part["policy_net_bps"]),
                "residual_ic": _spearman(part["_feature"], part["mc1_residual"]),
            }), include_groups=False,
        ).reset_index()
        for _, item in by_month.iterrows():
            monthly.append({"feature_name": field, "month": item["month"], "raw_ic": item.raw_ic, "residual_ic": item.residual_ic})
        finite = values[valid]
        timestamp_std = panel.loc[valid, ["__decision_ts__", field]].groupby("__decision_ts__")[field].std().dropna()
        month_median = panel.loc[valid].assign(_feature=values[valid]).groupby("month")["_feature"].median().std()
        q01, q99 = (finite.quantile(.01), finite.quantile(.99)) if len(finite) else (np.nan, np.nan)
        tail = labelled.loc[labelled_values.notna(), ["policy_net_bps"]].copy()
        tail["feature"] = labelled_values[labelled_values.notna()]
        tail_rows = {}
        for q in (.50, .80, .90, .95, .98):
            threshold = tail.feature.quantile(q) if len(tail) else np.nan
            picked = tail.loc[tail.feature.ge(threshold)]
            tail_rows[f"tail_{int(q*100)}_net_bps"] = float(picked.policy_net_bps.mean()) if len(picked) else np.nan
            tail_rows[f"tail_{int(q*100)}_positive_rate"] = float((picked.policy_net_bps > 0.0).mean()) if len(picked) else np.nan
        rows.append({
            "feature_name": field, "family": _family(field), "finite_rows": int(valid.sum()),
            "coverage": float(valid.mean()), "global_std": float(finite.std()) if len(finite) else np.nan,
            "cross_sectional_std_mean": float(timestamp_std.mean()) if len(timestamp_std) else np.nan,
            "monthly_median_dispersion": float(month_median), "q01": q01, "q99": q99,
            "raw_ic": _spearman(labelled_values, labelled.policy_net_bps),
            "residual_ic": _spearman(labelled_values, labelled.mc1_residual),
            "conditional_mi_m0": _conditional_mi_proxy(labelled.mc1_residual, labelled_values, labelled.m0_expected_bps),
            "positive_month_fraction_raw": float((by_month.raw_ic > 0.0).mean()),
            "positive_month_fraction_residual": float((by_month.residual_ic > 0.0).mean()),
            **tail_rows,
        })
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def _nested_family_fields(train: pd.DataFrame, fields: list[str]) -> tuple[str, ...]:
    """Select a compact family contract using *only* a held fold's prior data.

    The raw score is residual information times sign stability across the
    prior calendar months.  It is deliberately simple and deterministic: the
    downstream shallow model is allowed to learn a nonlinear response, but a
    feature cannot enter that model merely because it worked in the held
    month.  Missing values remain model missingness rather than a filter.
    """
    rows: list[dict[str, object]] = []
    for field in fields:
        value = pd.to_numeric(train[field], errors="coerce")
        coverage = float(value.notna().mean())
        information = _spearman(value, train["mc1_residual"])
        monthly = train.assign(_feature=value).groupby("month", sort=True).apply(
            lambda part: _spearman(part["_feature"], part["mc1_residual"]),
            include_groups=False,
        ).dropna()
        sign_stability = float((monthly * information > 0.0).mean()) if len(monthly) and np.isfinite(information) else 0.0
        rows.append({
            "feature": field,
            "coverage": coverage,
            "information": information,
            "sign_stability": sign_stability,
            "selection_score": abs(information) * sign_stability if coverage >= .70 else -np.inf,
        })
    ranked = pd.DataFrame(rows).sort_values(["selection_score", "feature"], ascending=[False, True], kind="stable")
    # A modest cap prevents a single family with many correlated transforms
    # from obtaining more mapper authority than a compact family.
    chosen = ranked.loc[np.isfinite(ranked.selection_score), "feature"].head(min(8, max(3, len(fields) // 3))).tolist()
    if not chosen:
        # HGB can accommodate nulls, but a source that is wholly unavailable
        # cannot make a causal specialist claim.  The fold records this state
        # instead of silently using a future-derived fallback.
        return ()
    return tuple(chosen)


def _family_probe(panel: pd.DataFrame, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = [field for field in FEATURE_COLUMNS if _family(field) == family]
    labelled = panel.loc[panel.valid_label & panel[fields].notna().any(axis=1)].copy()
    held_months = pd.date_range("2025-07-01", "2025-12-01", freq="MS", tz="UTC")
    pieces, trace = [], []
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=3)
        train = labelled.loc[
            labelled.__decision_ts__.ge(start) & labelled.__decision_ts__.lt(held) & labelled.policy_label_available_ts.lt(held)
        ].copy()
        test = labelled.loc[labelled.__decision_ts__.ge(held) & labelled.__decision_ts__.lt(end)].copy()
        if len(train) < 2_000 or test.empty:
            trace.append({"family": family, "held_month": f"{held:%Y-%m}", "status": "insufficient_prior_resolved_support", "train_rows": len(train), "held_rows": len(test)})
            continue
        selected = _nested_family_fields(train, fields)
        if not selected:
            trace.append({"family": family, "held_month": f"{held:%Y-%m}", "status": "no_prior_eligible_feature", "train_rows": len(train), "held_rows": len(test), "selected_features": "[]"})
            continue
        medians = train.loc[:, list(selected)].apply(pd.to_numeric, errors="coerce").median()
        target = pd.to_numeric(train.mc1_residual, errors="coerce").clip(*train.mc1_residual.quantile([.02, .98]).to_numpy(float))
        model = HistGradientBoostingRegressor(max_depth=2, max_iter=80, learning_rate=.04, l2_regularization=20.0, min_samples_leaf=100, random_state=SEED)
        model.fit(train.loc[:, list(selected)].apply(pd.to_numeric, errors="coerce").fillna(medians), target)
        test["family_probe_residual_bps"] = model.predict(test.loc[:, list(selected)].apply(pd.to_numeric, errors="coerce").fillna(medians))
        test["family"] = family
        pieces.append(test.loc[:, ["candidate_id", "__decision_ts__", "month", "family", "policy_net_bps", "mc1_residual", "family_probe_residual_bps"]])
        trace.append({"family": family, "held_month": f"{held:%Y-%m}", "status": "scored", "train_rows": len(train), "held_rows": len(test), "selected_features": json.dumps(selected), "selected_feature_count": len(selected)})
    return (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()), pd.DataFrame(trace)


def _frozen_2025_feature_contract(traces: pd.DataFrame) -> pd.DataFrame:
    """Freeze only features repeatedly selected by prior-only 2025 folds.

    This deliberately uses the *selection trace*, not 2026 outcomes or a
    retrospective full-period importance ranking.  A feature needs to appear
    in at least half of a family's successful temporal folds; the stable
    contract is capped at eight fields per family.
    """
    rows: list[dict[str, object]] = []
    for family, group in traces.loc[traces.status.eq("scored")].groupby("family", sort=True):
        folds = int(len(group))
        counts: dict[str, int] = {}
        for text in group.selected_features:
            for field in json.loads(text):
                counts[field] = counts.get(field, 0) + 1
        threshold = max(1, int(np.ceil(folds / 2.0)))
        for rank, (field, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0])), start=1):
            rows.append({
                "family": family,
                "feature_name": field,
                "selected_folds": count,
                "scored_folds": folds,
                "selection_frequency": count / max(1, folds),
                "stable_selected": bool(count >= threshold and rank <= 8),
            })
    contract = pd.DataFrame(rows)
    if contract.empty or not contract.stable_selected.any():
        raise AssertionError("2025 temporal feature selection produced no stable family contract")
    return contract


def run(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    panel = _load_panel(args.dynamic.resolve(), args.bcf.resolve(), args.current.resolve())
    panel = panel.loc[panel.__decision_ts__.ge(pd.Timestamp("2025-04-01", tz="UTC")) & panel.__decision_ts__.lt(pd.Timestamp("2026-01-01", tz="UTC"))].copy()
    out.mkdir(parents=True, exist_ok=False)
    quality, monthly = _quality_and_information(panel)
    quality.to_parquet(out / "feature_quality_and_information_2025.parquet", index=False)
    monthly.to_parquet(out / "feature_monthly_information_2025.parquet", index=False)
    probes, traces = [], []
    for family in ("CP", "SP", "WV", "EN", "DS"):
        prediction, trace = _family_probe(panel, family)
        probes.append(prediction)
        traces.append(trace)
    probe_frame = pd.concat(probes, ignore_index=True)
    trace_frame = pd.concat(traces, ignore_index=True)
    probe_frame.to_parquet(out / "family_probe_oof_2025.parquet", index=False)
    trace_frame.to_parquet(out / "family_probe_trace_2025.parquet", index=False)
    _frozen_2025_feature_contract(trace_frame).to_parquet(out / "frozen_2025_family_feature_contract.parquet", index=False)
    # The matrix itself stays in the source artifact; this compact joined panel
    # lets later nested specialist/MC1 stages reproduce the exact selection.
    panel.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "month", "m0_expected_bps", "m0_min_expected_bps", "policy_net_bps", "policy_label_available_ts", "valid_label", "mc1_residual", "dynamic_source_status", *FEATURE_COLUMNS]].to_parquet(out / "selection_panel_2025.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-exotic-dynamics-assessment-2025-v1",
        "scope": "offline strict-OOF 2025 feature assessment only; no MC1/live/policy mutation",
        "period": "2025-04 through 2025-12",
        "baseline": "immutable paired BCF/current historical MC1 expected-bps outputs; residual = parent-policy net bps minus their mean",
        "causality": "feature source is target-free completed 15m state; labels are used only after resolved policy_label_available_ts; family probes choose compact fields from each prior three-month window before scoring the held month; the 2026 feature contract is frozen only from recurring 2025 selections",
        "dynamic_root": str(args.dynamic.resolve()),
        "dynamic_manifest_sha256": _sha256(args.dynamic.resolve() / "run_manifest.json"),
        "bcf_sha256": _sha256(args.bcf.resolve()), "current_sha256": _sha256(args.current.resolve()),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--bcf", type=Path, default=BCF)
    parser.add_argument("--current", type=Path, default=CURRENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
