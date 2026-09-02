#!/usr/bin/env python3
"""Strict-prequential MC1-equivalent challenger above short P0 -> O250/H6 -> C3 -> K0.

The active research architecture is deliberately frozen to P0 -> O -> C -> K0.
This runner is therefore *not* an architecture promotion.  It isolates whether a
small, fixed, causally trained expected-net mapper can improve the frozen C3/K0
score.  It consumes only outer-OOF O/C/K0 predictions and point-in-time P0
context.  Its optional agreement geometry comes from the five independently
trained C-target arms in Round 3A; it is never available to the base O or C3
training process.

All mapper fits use only earlier outer-OOF rows whose H12 policy label resolved
strictly before the held month.  Recent correctness is calculated sequentially
from the same prior-resolved ledger.  The held population remains target-free
until scoring completes; invalid paths are excluded only from model targets,
history, and realised-economics accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402


SCHEMA = "strict_r3_short_p0_o250_c3_k0_mc1_equivalent_v1"
ROUND3 = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_targets_20260821_v1"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_o250_c3_k0_mc1_equivalent_20260822_v1"
TARGETS = (
    "C0_t5_regret",
    "C1_policy_net_ordinal",
    "C2_capture_efficiency",
    "C3_normalized_regret",
    "C4_hybrid_quality",
)
PRIMARY_TARGET = "C3_normalized_regret"
CONTEXT_ROOTS = (
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2024_maydec_20260821_v1",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2025_h1_20260821_v1",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2026_janjul_20260821_v1",
)
CONTEXT_FIELDS = (
    "prequential_base_rank42",
    "prequential_base_anchor_bps",
    "prequential_base_score",
)
ADMISSION_BPS = 50.0
POLICY_CLIP_BPS = 500.0
MIN_TRAIN_ROWS = 1_000
MIN_TRAIN_MONTHS = 3
RECENT_DAYS = 21
RECENT_TRIM = 0.10
MAPPER_PARAMS = {
    "max_depth": 2,
    "max_iter": 80,
    "learning_rate": 0.04,
    "l2_regularization": 20.0,
    "min_samples_leaf": 100,
    "random_state": 1729,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _finite(values: pd.Series | Iterable[float] | np.ndarray) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _valid_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["rich_path_label_valid"].fillna(False).astype(bool)
        & ~frame["rich_path_target_invalid"].fillna(True).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & _finite(frame["policy_net_bps"]).notna()
        & frame["__label_available_at__"].notna()
    )


def _assert_same_values(raw: pd.DataFrame, column: str) -> None:
    values = raw.groupby("candidate_id", sort=False)[column]
    if pd.api.types.is_datetime64_any_dtype(raw[column]):
        first = values.transform("first")
        if not raw[column].eq(first).fillna(raw[column].isna() & first.isna()).all():
            raise AssertionError(f"Round-3 arms disagree on protected timestamp {column}")
        return
    if pd.api.types.is_bool_dtype(raw[column]) or raw[column].dropna().isin([True, False]).all():
        first = values.transform("first").astype("boolean")
        current = raw[column].astype("boolean")
        if not current.eq(first).fillna(current.isna() & first.isna()).all():
            raise AssertionError(f"Round-3 arms disagree on protected label field {column}")
        return
    if not pd.api.types.is_numeric_dtype(raw[column]):
        first = values.transform("first")
        if not raw[column].eq(first).fillna(raw[column].isna() & first.isna()).all():
            raise AssertionError(f"Round-3 arms disagree on protected field {column}")
        return
    current = _finite(raw[column]).to_numpy(float)
    first = _finite(values.transform("first")).to_numpy(float)
    if not np.isclose(current, first, rtol=0.0, atol=2e-5, equal_nan=True).all():
        raise AssertionError(f"Round-3 arms disagree on protected label field {column}")


def _load_context() -> tuple[pd.DataFrame, dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for root in CONTEXT_ROOTS:
        path = root / "short_p0_top1_hourly_population.parquet"
        manifest = root / "run_manifest.json"
        if not path.exists() or not manifest.exists():
            raise FileNotFoundError(f"missing target-free P0 context under {root}")
        pieces.append(pd.read_parquet(path, columns=["candidate_id", *CONTEXT_FIELDS]))
        hashes[str(root.resolve())] = _sha256(manifest)
    context = pd.concat(pieces, ignore_index=True)
    for field in CONTEXT_FIELDS:
        _assert_same_values(context, field)
    context = context.drop_duplicates("candidate_id", keep="first")
    if context["candidate_id"].duplicated().any():
        raise AssertionError("target-free P0 context identity is non-unique")
    return context, hashes


def _load_round3() -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for target in TARGETS:
        path = ROUND3 / f"{target}_outer_oof_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path)
        if not part["target"].astype(str).eq(target).all():
            raise AssertionError(f"target identity mismatch in {path}")
        if part["candidate_id"].duplicated().any():
            raise AssertionError(f"non-unique Round-3 candidate identities in {target}")
        for field in ("__ts__", "__decision_ts__", "__label_available_at__"):
            part[field] = _utc(part[field])
        parts.append(part)
        source_hashes[target] = _sha256(path)
    raw = pd.concat(parts, ignore_index=True)
    if raw.groupby("candidate_id", sort=False).size().nunique() != 1 or int(raw.groupby("candidate_id", sort=False).size().iloc[0]) != len(TARGETS):
        raise AssertionError("Round-3 target arms do not share identical candidate identities")
    protected = (
        "__ts__", "__decision_ts__", "__symbol__", "side_name", "__label_available_at__",
        "mfe_6h_bps", "policy_net_bps", "policy_regret_bps", "policy_gross_bps",
        "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", "held_month",
    )
    for field in protected:
        _assert_same_values(raw, field)
    primary = raw.loc[raw["target"].eq(PRIMARY_TARGET)].copy()
    if not primary["side_name"].astype(str).str.lower().eq("short").all():
        raise AssertionError("MC1-equivalent input is not side-local short")
    index = ["candidate_id"]
    k0 = raw.pivot(index=index, columns="target", values="K0_expected_policy_net_bps").reindex(columns=TARGETS)
    op = raw.pivot(index=index, columns="target", values="opportunity_probability").reindex(columns=TARGETS)
    conversion = raw.pivot(index=index, columns="target", values="conversion_score").reindex(columns=TARGETS)
    if k0.isna().all(axis=1).any() or op.isna().all(axis=1).any() or conversion.isna().all(axis=1).any():
        raise AssertionError("one Round-3 target stream has no target-free O/C/K0 score")
    k0.columns = [f"k0__{target}" for target in TARGETS]
    op.columns = [f"opp__{target}" for target in TARGETS]
    conversion.columns = [f"conversion__{target}" for target in TARGETS]
    primary = primary.merge(k0.reset_index(), on="candidate_id", how="left", validate="one_to_one")
    primary = primary.merge(op.reset_index(), on="candidate_id", how="left", validate="one_to_one")
    primary = primary.merge(conversion.reset_index(), on="candidate_id", how="left", validate="one_to_one")
    context, context_hashes = _load_context()
    primary = primary.merge(context, on="candidate_id", how="left", validate="one_to_one")
    if primary.loc[:, list(CONTEXT_FIELDS)].isna().all(axis=1).any():
        raise AssertionError("a current C3 row lacks all target-free P0 context")
    return _add_geometry(primary), {
        "round3_prediction_hashes": source_hashes,
        "round3_manifest_sha256": _sha256(ROUND3 / "run_manifest.json"),
        "context_manifest_hashes": context_hashes,
        "targetfree_candidate_rows": int(len(primary)),
    }


def _add_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    k0_fields = [f"k0__{target}" for target in TARGETS]
    opp_fields = [f"opp__{target}" for target in TARGETS]
    k0 = output.loc[:, k0_fields].apply(_finite).to_numpy(float)
    opp = output.loc[:, opp_fields].apply(_finite).to_numpy(float)
    output["k0_c3_bps"] = _finite(output["K0_expected_policy_net_bps"]).to_numpy(float)
    output["opportunity_probability_c3"] = _finite(output["opportunity_probability"]).to_numpy(float)
    output["conversion_score_c3"] = _finite(output["conversion_score"]).to_numpy(float)
    output["agreement_k0_mean_bps"] = np.nanmean(k0, axis=1)
    output["agreement_k0_median_bps"] = np.nanmedian(k0, axis=1)
    output["agreement_k0_std_bps"] = np.nanstd(k0, axis=1)
    output["agreement_k0_iqr_bps"] = np.nanquantile(k0, .75, axis=1) - np.nanquantile(k0, .25, axis=1)
    output["agreement_k0_positive_fraction"] = np.nanmean(k0 >= ADMISSION_BPS, axis=1)
    output["agreement_c3_minus_median_bps"] = output["k0_c3_bps"] - output["agreement_k0_median_bps"]
    output["agreement_opp_mean"] = np.nanmean(opp, axis=1)
    output["agreement_opp_std"] = np.nanstd(opp, axis=1)
    output["agreement_opp_ge_50_fraction"] = np.nanmean(opp >= .50, axis=1)
    # Fixed economic bins retain their score meaning at inference; this is not
    # a held-period percentile transformation.
    output["k0_score_band"] = np.digitize(
        output["k0_c3_bps"].to_numpy(float), [-200., -100., 0., 50., 100., 150., 200., 300.], right=False,
    ).astype(np.int8)
    return output


def _trimmed_mean(values: Iterable[float]) -> float:
    array = np.sort(np.asarray(list(values), dtype=float))
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    trim = int(math.floor(len(array) * RECENT_TRIM))
    kept = array[trim:len(array) - trim] if trim and len(array) > 2 * trim else array
    return float(kept.mean())


def _causal_correctness_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal, equal-day 21d K0 residual and correctness state."""
    output = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True).copy()
    valid = output.loc[_valid_label(output), ["__decision_ts__", "__label_available_at__", "k0_score_band", "k0_c3_bps", "policy_net_bps"]].copy()
    valid["residual_bps"] = _finite(valid["policy_net_bps"]).to_numpy(float) - _finite(valid["k0_c3_bps"]).to_numpy(float)
    valid["hit"] = (_finite(valid["policy_net_bps"]).to_numpy(float) > 0.0).astype(float)
    valid = valid.sort_values(["__label_available_at__", "__decision_ts__"], kind="stable").reset_index(drop=True)
    history: deque[tuple[pd.Timestamp, pd.Timestamp, int, float, float]] = deque()
    pointer = 0
    fields: dict[str, np.ndarray] = {
        "state_global_residual21_bps": np.zeros(len(output), dtype=np.float32),
        "state_band_residual21_bps": np.zeros(len(output), dtype=np.float32),
        "state_global_hit_rate21": np.full(len(output), .5, dtype=np.float32),
        "state_band_hit_rate21": np.full(len(output), .5, dtype=np.float32),
        "state_global_support_days21": np.zeros(len(output), dtype=np.int16),
        "state_band_support21": np.zeros(len(output), dtype=np.int16),
    }
    for decision, group in output.groupby("__decision_ts__", sort=True):
        while pointer < len(valid) and valid.loc[pointer, "__label_available_at__"] < decision:
            row = valid.loc[pointer]
            history.append((row["__decision_ts__"], row["__decision_ts__"].normalize(), int(row["k0_score_band"]), float(row["residual_bps"]), float(row["hit"])))
            pointer += 1
        cutoff = decision - pd.Timedelta(days=RECENT_DAYS)
        while history and history[0][0] < cutoff:
            history.popleft()
        daily: dict[pd.Timestamp, list[tuple[float, float]]] = defaultdict(list)
        by_band: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for _, day, band, residual, hit in history:
            daily[day].append((residual, hit))
            by_band[band].append((residual, hit))
        global_residual = _trimmed_mean(np.mean([row[0] for row in values]) for values in daily.values())
        global_hit = _trimmed_mean(np.mean([row[1] for row in values]) for values in daily.values())
        global_residual = 0.0 if not np.isfinite(global_residual) else global_residual
        global_hit = .5 if not np.isfinite(global_hit) else global_hit
        for row_index, row in group.iterrows():
            local = by_band.get(int(row["k0_score_band"]), [])
            support = len(local)
            local_residual = (sum(item[0] for item in local) + 10.0 * global_residual) / (support + 10.0)
            local_hit = (sum(item[1] for item in local) + 10.0 * global_hit) / (support + 10.0)
            fields["state_global_residual21_bps"][row_index] = global_residual
            fields["state_band_residual21_bps"][row_index] = local_residual
            fields["state_global_hit_rate21"][row_index] = global_hit
            fields["state_band_hit_rate21"][row_index] = local_hit
            fields["state_global_support_days21"][row_index] = min(len(daily), np.iinfo(np.int16).max)
            fields["state_band_support21"][row_index] = min(support, np.iinfo(np.int16).max)
    for field, values in fields.items():
        output[field] = values
    return output


SCORE_FEATURES = (
    "k0_c3_bps", "opportunity_probability_c3", "conversion_score_c3", *CONTEXT_FIELDS,
)
AGREEMENT_FEATURES = (
    "agreement_k0_mean_bps", "agreement_k0_std_bps", "agreement_k0_iqr_bps",
    "agreement_k0_positive_fraction", "agreement_c3_minus_median_bps",
    "agreement_opp_mean", "agreement_opp_std", "agreement_opp_ge_50_fraction",
)
CORRECTNESS_FEATURES = (
    "state_band_residual21_bps", "state_band_hit_rate21", "state_band_support21",
)
ARM_FEATURES: dict[str, tuple[str, ...]] = {
    "K0_native_train_p80": (),
    "K0_absolute_50bps": (),
    "MC0_score_geometry": SCORE_FEATURES,
    "MC1_c_target_agreement": (*SCORE_FEATURES, *AGREEMENT_FEATURES),
    "MC1_agreement_global_shift": (*SCORE_FEATURES, *AGREEMENT_FEATURES),
    "MC1_agreement_correctness": (*SCORE_FEATURES, *AGREEMENT_FEATURES, *CORRECTNESS_FEATURES),
}
# These arms are intentionally non-generative: native K0 must already admit a
# row, then the fixed MC estimate may only veto it.  The 50-bps floor is the
# same declared economic floor used by the direct mapper arms, not a tuned
# secondary threshold.
DEMOTION_ARMS = {
    "K0_native_p80_demote_MC0_50": "MC0_score_geometry_expected_net_bps",
    "K0_native_p80_demote_MC1_agreement_50": "MC1_c_target_agreement_expected_net_bps",
}
METRIC_ARMS = (*ARM_FEATURES, *DEMOTION_ARMS)


def _matrix(frame: pd.DataFrame, features: Sequence[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame.loc[:, list(features)].apply(_finite).astype(np.float32)
    if medians is None:
        medians = values.median(axis=0, skipna=True).fillna(0.0).astype(np.float32)
    return values.fillna(medians).astype(np.float32), medians


def _fit_predict(train: pd.DataFrame, held: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    x_train, medians = _matrix(train, features)
    x_held, _ = _matrix(held, features, medians)
    target = _finite(train["policy_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    return np.asarray(HistGradientBoostingRegressor(**MAPPER_PARAMS).fit(x_train, target).predict(x_held), dtype=float)


def _monthly_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    months = sorted(pd.to_datetime(frame["held_month"] + "-01", utc=True).drop_duplicates())
    for month in months:
        stop = month + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(stop)].copy()
        train = frame.loc[frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & _valid_label(frame)].copy()
        audit: dict[str, Any] = {
            "held_month": month.strftime("%Y-%m"), "held_targetfree_rows": int(len(held)),
            "outer_oof_train_rows": int(len(train)), "outer_oof_train_months": int(train["__decision_ts__"].dt.to_period("M").nunique()),
            "max_train_label_available_at": train["__label_available_at__"].max().isoformat() if len(train) else None,
        }
        if len(train) < MIN_TRAIN_ROWS or audit["outer_oof_train_months"] < MIN_TRAIN_MONTHS:
            audit.update({"status": "skipped", "reason": "insufficient strict-prequential mapper support"})
            audits.append(audit)
            continue
        if not train["__label_available_at__"].lt(month).all():
            raise AssertionError("mapper training includes unresolved held-month label")
        part = held.copy()
        part["K0_native_train_p80_expected_net_bps"] = _finite(part["k0_c3_bps"]).to_numpy(float)
        part["K0_absolute_50bps_expected_net_bps"] = _finite(part["k0_c3_bps"]).to_numpy(float)
        for arm, fields in ARM_FEATURES.items():
            if arm.startswith("K0_"):
                continue
            expected = _fit_predict(train, held, fields)
            if arm in {"MC1_agreement_global_shift", "MC1_agreement_correctness"}:
                expected += held["state_global_residual21_bps"].to_numpy(float)
            part[f"{arm}_expected_net_bps"] = expected.astype(np.float32)
        rows.append(part)
        audit.update({"status": "complete", "reason": "", "admission_rule": f"expected net >= {ADMISSION_BPS:.0f} bps"})
        audits.append(audit)
    if not rows:
        raise RuntimeError("MC1-equivalent mapper has no supported strict-OOS folds")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def _cvar(values: np.ndarray, fraction: float = .10) -> float:
    array = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    return float(array[:max(1, int(math.ceil(len(array) * fraction)))].mean()) if len(array) else float("nan")


def _metrics(part: pd.DataFrame, arm: str) -> dict[str, Any]:
    prediction = f"{arm}_expected_net_bps"
    if arm == "K0_native_train_p80":
        selected = part.loc[_finite(part["k0_c3_bps"]).ge(_finite(part["K0_train_p80_expected_policy_net_bps"]))].copy()
        admission = "frozen outer-train inner-OOF p80 K0 threshold"
    elif arm in DEMOTION_ARMS:
        prediction = DEMOTION_ARMS[arm]
        native = _finite(part["k0_c3_bps"]).ge(_finite(part["K0_train_p80_expected_policy_net_bps"]))
        selected = part.loc[native & _finite(part[prediction]).ge(ADMISSION_BPS)].copy()
        admission = f"native K0 p80 admission AND MC demotion score >= {ADMISSION_BPS:.0f} bps"
    else:
        selected = part.loc[_finite(part[prediction]).ge(ADMISSION_BPS)].copy()
        admission = f"expected net >= {ADMISSION_BPS:.0f} bps"
    known = selected.loc[_valid_label(selected)].copy()
    net = _finite(known["policy_net_bps"]).to_numpy(float)
    all_known = part.loc[_valid_label(part)].copy()
    ic = pd.Series(_finite(all_known[prediction]).to_numpy(float)).corr(pd.Series(_finite(all_known["policy_net_bps"]).to_numpy(float)), method="spearman") if len(all_known) >= 5 else float("nan")
    return {
        "arm": arm, "admission_rule": admission, "scored_candidates": int(len(part)),
        "selected_candidates": int(len(selected)), "outcome_known_candidates": int(len(known)),
        "outcome_coverage": float(len(known) / len(selected)) if len(selected) else float("nan"),
        "net_bps_per_trade": float(net.mean()) if len(net) else float("nan"), "total_net_bps": float(net.sum()) if len(net) else 0.0,
        "cvar10_bps": _cvar(net), "positive_fraction": float(np.mean(net > 0.0)) if len(net) else float("nan"),
        "all_valid_rank_ic": float(ic) if np.isfinite(ic) else float("nan"),
    }


def _tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly: list[dict[str, Any]] = []
    era: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for arm in METRIC_ARMS:
        for month, part in predictions.groupby("held_month", sort=True):
            item = _metrics(part, arm); item["period"] = str(month); monthly.append(item)
        for year, part in predictions.groupby(predictions["__decision_ts__"].dt.year, sort=True):
            item = _metrics(part, arm); item["period"] = str(year); era.append(item)
        target_eras = [row for row in era if row["arm"] == arm and row["period"] in {"2025", "2026"}]
        per_month = [row for row in monthly if row["arm"] == arm and str(row["period"]).startswith(("2025-", "2026-"))]
        total_known = sum(row["outcome_known_candidates"] for row in target_eras)
        mean_net = (sum(row["total_net_bps"] for row in target_eras) / total_known) if total_known else float("nan")
        summary.append({
            "arm": arm,
            "net_2025": next((row["net_bps_per_trade"] for row in target_eras if row["period"] == "2025"), float("nan")),
            "net_2026": next((row["net_bps_per_trade"] for row in target_eras if row["period"] == "2026"), float("nan")),
            "mean_net_bps_per_trade": mean_net,
            "total_net_bps": sum(row["total_net_bps"] for row in target_eras),
            "selected": total_known,
            "worst_month": min((row["net_bps_per_trade"] for row in per_month if np.isfinite(row["net_bps_per_trade"])), default=float("nan")),
            "mean_cvar10": float(np.mean([row["cvar10_bps"] for row in target_eras])) if target_eras else float("nan"),
        })
    summary_frame = pd.DataFrame(summary)
    control = float(summary_frame.loc[summary_frame["arm"].eq("K0_absolute_50bps"), "selected"].iloc[0])
    summary_frame["participation_vs_k0_abs50"] = summary_frame["selected"] / max(control, 1.0)
    summary_frame["passes_gate"] = summary_frame["net_2025"].ge(90.0) & summary_frame["net_2026"].ge(90.0) & summary_frame["participation_vs_k0_abs50"].ge(.70)
    return pd.DataFrame(monthly), pd.DataFrame(era), summary_frame.sort_values(["passes_gate", "mean_net_bps_per_trade", "worst_month", "total_net_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    local = frame.loc[:, list(columns)] if columns is not None else frame
    cols = [str(item) for item in local.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in local.itertuples(index=False, name=None))
    return "\n".join(lines)


def _write_report(out: Path, *, monthly: pd.DataFrame, era: pd.DataFrame, summary: pd.DataFrame, audit: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# Short P0 → O250/H6 → C3 → K0 MC1-equivalent challenger", "",
        "Research-only. The frozen architecture remains P0 → O → C → K0. This separate test has no canonical, live, or promotion authority.", "",
        "## Two-era selection", "", _table(summary), "",
        "## Era economics", "", _table(era, ["arm", "period", "selected_candidates", "net_bps_per_trade", "total_net_bps", "cvar10_bps", "all_valid_rank_ic"]), "",
        "## Monthly economics", "", _table(monthly, ["arm", "period", "selected_candidates", "net_bps_per_trade", "total_net_bps", "cvar10_bps", "all_valid_rank_ic"]), "",
        "## Contract", "",
        "- `MC0_score_geometry` is a depth-2 expected-policy-net mapper over frozen C3/K0/O/P0 geometry.",
        "- `MC1_c_target_agreement` adds target-free agreement/dispersion across the five independently strict-OOF C target arms.",
        "- The two remaining MC1 variants add either a causal equal-day 21-day global residual shift or score-band recent correctness; all outcomes resolve strictly before the decision timestamp.",
        "- Every mapper is fit only on outer-OOF, valid rows with `label_available_at < held_month_start`. All candidates are scored before invalid paths are excluded from economics.",
        "- `K0_native_train_p80` is the unchanged C3 control. `K0_absolute_50bps` isolates the fixed +50-bps threshold from mapper effects.",
        "- `K0_native_p80_demote_*` retains native K0 admission and lets MC0/MC1 only reject a candidate whose independently mapped EV is below the same +50-bps economic floor; it can never manufacture an admission.",
        "- The C-target agreement inputs are experimental and were produced by a target-selection run over the same 2025–2026 evidence. Results are model-selection evidence, not untouched confirmation.", "",
        "## Fold audit", "", _table(audit), "",
        "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_O250_C3_K0_MC1_EQUIVALENT_REPORT.md").write_text("\n".join(lines))


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    frame, provenance = _load_round3()
    frame = _causal_correctness_state(frame)
    predictions, audit = _monthly_predictions(frame)
    monthly, era, summary = _tables(predictions)
    out.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short",
        "scope": "research-only MC1-equivalent ablation above the frozen current P0→O250/H6→C3→K0 stack; no canonical or live change",
        "architecture": "frozen P0 → O250_H6 → C3_normalized_regret → K0; optional post-K0 challenger only",
        "primary_stack": {"opportunity": "mfe_6h_bps > 250", "conversion": PRIMARY_TARGET, "admission_control": "outer-train inner-OOF K0 p80"},
        "mapper": {"type": "HistGradientBoostingRegressor", **MAPPER_PARAMS, "arms": {name: list(fields) for name, fields in ARM_FEATURES.items()}, "non_generative_demotion_arms": DEMOTION_ARMS},
        "admission": {"mapper_and_k0_abs_control": f"expected policy net >= {ADMISSION_BPS:.0f} bps", "native_k0": "unchanged outer-train inner-OOF p80"},
        "causality": {"upstream": "each O/C/K0 arm is monthly outer strict-OOF", "mapper_fit": "valid outer-OOF rows with label_available_at < held month start", "correctness": f"prior-resolved only, strict < decision, equal-day {RECENT_DAYS}d/{RECENT_TRIM:.0%} trimmed", "forbidden": ["held outcome features", "held-period percentile ranks", "future MFE route", "canonical/live mutation"]},
        "upstream": provenance,
    }
    predictions.to_parquet(out / "mc1_equivalent_outer_oof_predictions.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "mc1_equivalent_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(out / "mc1_equivalent_era_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "mc1_equivalent_summary.parquet", index=False, compression="zstd")
    audit.to_parquet(out / "mc1_equivalent_fold_audit.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(out, monthly=monthly, era=era, summary=summary, audit=audit, manifest=manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
