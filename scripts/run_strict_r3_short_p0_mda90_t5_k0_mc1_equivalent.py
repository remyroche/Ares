#!/usr/bin/env python3
"""Strict-prequential MC1-equivalent mapper on short MDA90 × T5 × K0.

This is a deliberately narrow, research-only successor to the completed
short P0 opportunity/conversion experiment.  It does *not* retrain the
opportunity or conversion heads.  Instead it consumes their already strict-
OOF monthly outputs and asks the MC1 question one level higher:

    given the MDA90/T5/K0 expected-policy-EV and agreement among six frozen
    opportunity heads, what is the expected policy net of this candidate?

The mapper is always trained on earlier monthly OOS rows whose exact policy
label was resolved before the held month.  Recent-correctness state is built
row by row from prior-resolved OOS rows only.  Hence neither realised MFE nor
the current/held month's policy outcome can affect a candidate score.

Variants are intentionally small and fixed to the long MC1_d2 model geometry:

* ``MC0_score_only``: MDA90/T5/K0 and P0 base context;
* ``MC1_agreement``: MC0 plus cross-opportunity-head agreement;
* ``MC1_agreement_global_shift``: MC1 plus the exact long-MC1-style causal
  21-day global residual adjustment;
* ``MC1_agreement_correctness``: MC1 plus causal score-band correctness and
  the long-MC1-style 21-day equal-day-trimmed global residual shift.

All variants are research-only and have no short live, canonical, or portfolio
authority.  They are evaluated with an absolute expected-policy-EV >= +50 bps
admission rule so they can be compared directly with the intended MC1 use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
SIDE = "short"
SCHEMA = "strict_r3_short_p0_mda90_t5_k0_mc1_equivalent_v1"
SOURCE = ROOT / "data_perp/artifacts/strict_r3_short_p0_two_stage_opportunity_conversion_2024may_2026jul_20260821_v2"
SOURCE_ROOTS = (
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2024_maydec_20260821_v1",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2025_h1_20260821_v1",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2026_janjul_20260821_v1",
)
BASE_ARM = "O_MDA90_binary"
AGREEMENT_ARMS = (
    "O_F41_binary",
    "O_F90_binary",
    "O_F115_binary",
    "O_MDA30_binary",
    "O_MDA60_binary",
    "O_MDA90_binary",
)
EXPECTED_K0 = "K0_analytic_mixture_expected_policy_net_bps"
K0_THRESHOLD = "K0_analytic_mixture_train_p80_expected_policy_net_bps"
ADMISSION_BPS = 50.0
POLICY_CLIP_BPS = 500.0
MIN_TRAIN_ROWS = 1_000
MIN_TRAIN_MONTHS = 3
RECENT_DAYS = 21
RECENT_TRIM = 0.10
MC1_PARAMS = {
    "max_depth": 2,
    "max_iter": 80,
    "learning_rate": 0.04,
    "l2_regularization": 20.0,
    "min_samples_leaf": 100,
    "random_state": 1729,
}
CONTEXT_FIELDS = (
    "prequential_base_rank42",
    "prequential_base_anchor_bps",
    "prequential_base_score",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp | pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_datetime(value, utc=True, errors="raise")
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _finite(values: pd.Series | Iterable[float] | np.ndarray) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _valid_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["rich_path_label_valid"].fillna(False).astype(bool)
        & ~frame["rich_path_target_invalid"].fillna(True).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & _finite(frame["policy_net_bps"]).notna()
        & frame["__label_available_at__"].notna()
    )


def _assert_numeric_duplicates(frame: pd.DataFrame, *, identity: str, columns: Sequence[str]) -> pd.DataFrame:
    repeated = frame[identity].duplicated(keep=False)
    if not repeated.any():
        return frame.copy()
    duplicate = frame.loc[repeated].sort_values(identity, kind="stable")
    for column in columns:
        values = _finite(duplicate[column]).to_numpy(float)
        first = _finite(duplicate.groupby(identity, sort=False)[column].transform("first")).to_numpy(float)
        if not np.isclose(values, first, rtol=0.0, atol=2e-5, equal_nan=True).all():
            candidate = str(duplicate.loc[~np.isclose(values, first, rtol=0.0, atol=2e-5, equal_nan=True), identity].iloc[0])
            raise AssertionError(f"target-free context sources disagree for {candidate}/{column}")
    return pd.concat([frame.loc[~repeated], duplicate.drop_duplicates(identity, keep="first")], ignore_index=True)


def _load_context(roots: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    columns = ["candidate_id", "side_name", *CONTEXT_FIELDS]
    for root in roots:
        path = root / "short_p0_top1_hourly_population.parquet"
        manifest = root / "run_manifest.json"
        if not path.exists() or not manifest.exists():
            raise FileNotFoundError(f"missing immutable short P0 source under {root}")
        part = pd.read_parquet(path, columns=columns)
        if not part["side_name"].astype(str).str.lower().eq(SIDE).all():
            raise AssertionError(f"non-short P0 context source: {root}")
        pieces.append(part)
        hashes[str(root.resolve())] = _sha256(manifest)
    output = _assert_numeric_duplicates(pd.concat(pieces, ignore_index=True), identity="candidate_id", columns=CONTEXT_FIELDS)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("target-free P0 context identity is non-unique")
    return output.loc[:, ["candidate_id", *CONTEXT_FIELDS]], hashes


def _rank_01(values: pd.Series) -> np.ndarray:
    """Deterministic rank coordinate within a fixed source arm, never outcomes."""
    source = _finite(values)
    output = np.full(len(source), 0.5, dtype=float)
    finite = source.notna().to_numpy(bool)
    if finite.any():
        output[finite] = source.loc[finite].rank(method="average", pct=True).to_numpy(float)
    return output


def _load_wide(source: Path, context_roots: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = source / "two_stage_outer_oof_predictions.parquet"
    manifest = source / "run_manifest.json"
    if not path.exists() or not manifest.exists():
        raise FileNotFoundError(f"not an O×C source artifact: {source}")
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "__label_available_at__",
        "mfe_12h_bps", "policy_net_bps", "policy_regret_bps", "rich_path_label_valid",
        "rich_path_target_invalid", "policy_path_valid", "opportunity_probability", "conversion_cdf",
        EXPECTED_K0, K0_THRESHOLD, "arm", "held_month",
    ]
    raw = pd.read_parquet(path, columns=columns)
    raw = raw.loc[raw["arm"].astype(str).isin(AGREEMENT_ARMS)].copy()
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        raw[column] = _utc(raw[column])
    if not raw["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError("O×C source is not side-local short")
    if raw.duplicated(["candidate_id", "arm"]).any():
        raise AssertionError("O×C source duplicates candidate/arm identities")
    base = raw.loc[raw["arm"].eq(BASE_ARM)].copy()
    if len(base) * len(AGREEMENT_ARMS) != len(raw):
        raise AssertionError("all agreement heads must score each MDA90 candidate")
    index = ["candidate_id", "held_month"]
    k0 = raw.pivot(index=index, columns="arm", values=EXPECTED_K0).reindex(columns=AGREEMENT_ARMS)
    op = raw.pivot(index=index, columns="arm", values="opportunity_probability").reindex(columns=AGREEMENT_ARMS)
    if k0.isna().all(axis=1).any() or op.isna().all(axis=1).any():
        raise AssertionError("a target-free agreement head has no usable output")
    k0.columns = [f"k0__{name}" for name in AGREEMENT_ARMS]
    op.columns = [f"opp__{name}" for name in AGREEMENT_ARMS]
    base = base.merge(k0.reset_index(), on=index, how="left", validate="one_to_one")
    base = base.merge(op.reset_index(), on=index, how="left", validate="one_to_one")
    context, context_hashes = _load_context(context_roots)
    base = base.merge(context, on="candidate_id", how="left", validate="one_to_one")
    if base.loc[:, list(CONTEXT_FIELDS)].isna().all(axis=1).any():
        raise AssertionError("a target-free MDA90 candidate lacks all P0 score context")
    if not base["__label_available_at__"].dropna().eq(
        base.loc[base["__label_available_at__"].notna(), "__decision_ts__"] + pd.Timedelta(hours=12)
    ).all():
        raise AssertionError("MC1 source labels must resolve at decision plus 12 hours")
    return _add_static_agreement(base), {
        "source_manifest_sha256": _sha256(manifest),
        "context_manifest_hashes": context_hashes,
        "outer_source_rows": int(len(base)),
    }


def _add_static_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    k0_fields = [f"k0__{name}" for name in AGREEMENT_ARMS]
    opp_fields = [f"opp__{name}" for name in AGREEMENT_ARMS]
    k0 = output.loc[:, k0_fields].apply(_finite).to_numpy(float)
    opp = output.loc[:, opp_fields].apply(_finite).to_numpy(float)
    output["k0_mda90_bps"] = _finite(output[EXPECTED_K0]).to_numpy(float)
    output["opportunity_probability_mda90"] = _finite(output["opportunity_probability"]).to_numpy(float)
    output["conversion_cdf_mda90"] = _finite(output["conversion_cdf"]).to_numpy(float)
    output["agreement_k0_mean_bps"] = np.nanmean(k0, axis=1)
    output["agreement_k0_median_bps"] = np.nanmedian(k0, axis=1)
    output["agreement_k0_std_bps"] = np.nanstd(k0, axis=1)
    output["agreement_k0_iqr_bps"] = np.nanquantile(k0, .75, axis=1) - np.nanquantile(k0, .25, axis=1)
    output["agreement_k0_positive_fraction"] = np.nanmean(k0 >= ADMISSION_BPS, axis=1)
    output["agreement_k0_mda90_minus_median_bps"] = output["k0_mda90_bps"] - output["agreement_k0_median_bps"]
    output["agreement_opp_mean"] = np.nanmean(opp, axis=1)
    output["agreement_opp_std"] = np.nanstd(opp, axis=1)
    output["agreement_opp_ge_50_fraction"] = np.nanmean(opp >= .50, axis=1)
    # Numeric fixed edges, unlike a held-period quantile, retain the same
    # score-band meaning at inference.  They condition recent correctness only.
    output["k0_score_band"] = np.digitize(
        output["k0_mda90_bps"].to_numpy(float), [-200., -100., 0., 50., 100., 150., 200., 300.], right=False,
    ).astype(np.int8)
    return output


def _trimmed_mean(values: Iterable[float], trim: float = RECENT_TRIM) -> float:
    array = np.sort(np.asarray(list(values), dtype=float))
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    count = int(math.floor(len(array) * trim))
    retained = array[count:len(array) - count] if count and len(array) > 2 * count else array
    return float(retained.mean())


def _causal_correctness_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach 21d state using labels strictly resolved *before* each decision.

    The input must be one already-OOS MDA90 prediction per candidate.  Rows
    missing an exact policy outcome are never placed in the history.  Equal-day
    aggregation means one high-activity shock day has one day's influence.
    """
    output = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
    valid = output.loc[_valid_label(output), [
        "__decision_ts__", "__label_available_at__", "k0_score_band", "k0_mda90_bps", "policy_net_bps",
    ]].copy()
    valid["residual"] = _finite(valid["policy_net_bps"]).to_numpy(float) - _finite(valid["k0_mda90_bps"]).to_numpy(float)
    valid["hit"] = (_finite(valid["policy_net_bps"]).to_numpy(float) > 0.0).astype(float)
    valid = valid.sort_values(["__label_available_at__", "__decision_ts__"], kind="stable").reset_index(drop=True)
    history: deque[tuple[pd.Timestamp, pd.Timestamp, int, float, float]] = deque()
    pointer = 0
    fields = {
        "state_global_residual21_bps": np.zeros(len(output), dtype=float),
        "state_band_residual21_bps": np.zeros(len(output), dtype=float),
        "state_global_hit_rate21": np.full(len(output), .5, dtype=float),
        "state_band_hit_rate21": np.full(len(output), .5, dtype=float),
        "state_global_support_days21": np.zeros(len(output), dtype=np.int16),
        "state_band_support21": np.zeros(len(output), dtype=np.int16),
    }
    position = 0
    for decision, group in output.groupby("__decision_ts__", sort=True):
        # Strict < is intentional: labels resolving exactly at a decision are
        # not usable for that decision, matching the outer O×C contract.
        while pointer < len(valid) and valid.loc[pointer, "__label_available_at__"] < decision:
            row = valid.loc[pointer]
            history.append((row["__decision_ts__"], row["__decision_ts__"].normalize(), int(row["k0_score_band"]), float(row["residual"]), float(row["hit"])))
            pointer += 1
        cutoff = decision - pd.Timedelta(days=RECENT_DAYS)
        while history and history[0][0] < cutoff:
            history.popleft()
        daily: dict[pd.Timestamp, list[tuple[float, float]]] = defaultdict(list)
        for _, day, _, residual, hit in history:
            daily[day].append((residual, hit))
        daily_residual = [float(np.mean([item[0] for item in items])) for items in daily.values()]
        daily_hit = [float(np.mean([item[1] for item in items])) for items in daily.values()]
        global_residual = _trimmed_mean(daily_residual)
        global_hit = _trimmed_mean(daily_hit)
        global_residual = 0.0 if not np.isfinite(global_residual) else global_residual
        global_hit = .5 if not np.isfinite(global_hit) else global_hit
        for row_index, row in group.iterrows():
            band = int(row["k0_score_band"])
            local = [(residual, hit) for _, _, prior_band, residual, hit in history if prior_band == band]
            local_count = len(local)
            # Ten-observation partial pooling makes an empty/rare band fall
            # back to its causal global state, never a later outcome.
            local_residual = (sum(item[0] for item in local) + 10.0 * global_residual) / (local_count + 10.0)
            local_hit = (sum(item[1] for item in local) + 10.0 * global_hit) / (local_count + 10.0)
            fields["state_global_residual21_bps"][position] = global_residual
            fields["state_band_residual21_bps"][position] = local_residual
            fields["state_global_hit_rate21"][position] = global_hit
            fields["state_band_hit_rate21"][position] = local_hit
            fields["state_global_support_days21"][position] = min(len(daily), np.iinfo(np.int16).max)
            fields["state_band_support21"][position] = min(local_count, np.iinfo(np.int16).max)
            position += 1
    if position != len(output):
        raise AssertionError("causal correctness-state traversal lost identities")
    for field, values in fields.items():
        output[field] = values
    return output


SCORE_FEATURES = (
    "k0_mda90_bps",
    "opportunity_probability_mda90",
    "conversion_cdf_mda90",
    *CONTEXT_FIELDS,
)
AGREEMENT_FEATURES = (
    "agreement_k0_mean_bps",
    "agreement_k0_std_bps",
    "agreement_k0_iqr_bps",
    "agreement_k0_positive_fraction",
    "agreement_k0_mda90_minus_median_bps",
    "agreement_opp_mean",
    "agreement_opp_std",
    "agreement_opp_ge_50_fraction",
)
CORRECTNESS_FEATURES = (
    "state_band_residual21_bps",
    "state_band_hit_rate21",
    "state_band_support21",
)
ARM_FEATURES = {
    "K0_native_train_p80": (),
    "K0_absolute_50bps": (),
    "MC0_score_only": SCORE_FEATURES,
    "MC1_agreement": (*SCORE_FEATURES, *AGREEMENT_FEATURES),
    "MC1_agreement_global_shift": (*SCORE_FEATURES, *AGREEMENT_FEATURES),
    "MC1_agreement_correctness": (*SCORE_FEATURES, *AGREEMENT_FEATURES, *CORRECTNESS_FEATURES),
}


def _matrix(frame: pd.DataFrame, features: Sequence[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame.loc[:, list(features)].apply(_finite).astype(np.float32)
    if medians is None:
        medians = values.median(axis=0, skipna=True).fillna(0.0).astype(np.float32)
    return values.fillna(medians).astype(np.float32), medians


def _fit_predict(train: pd.DataFrame, held: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    x_train, medians = _matrix(train, features)
    x_held, _ = _matrix(held, features, medians)
    target = _finite(train["policy_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    model = HistGradientBoostingRegressor(**MC1_PARAMS).fit(x_train, target)
    return np.asarray(model.predict(x_held), dtype=float)


def _monthly_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    months = sorted(pd.to_datetime(frame["held_month"] + "-01", utc=True).drop_duplicates())
    for month in months:
        stop = month + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(stop)].copy()
        train = frame.loc[
            frame["__decision_ts__"].lt(month)
            & frame["__label_available_at__"].lt(month)
            & _valid_label(frame)
        ].copy()
        status = "complete"
        reason = ""
        if len(train) < MIN_TRAIN_ROWS or train["__decision_ts__"].dt.to_period("M").nunique() < MIN_TRAIN_MONTHS:
            status = "skipped"
            reason = "insufficient strict-prequential mapper training support"
        audit = {
            "held_month": month.strftime("%Y-%m"), "status": status, "reason": reason,
            "held_targetfree_rows": int(len(held)), "outer_oof_train_rows": int(len(train)),
            "outer_oof_train_months": int(train["__decision_ts__"].dt.to_period("M").nunique()),
            "max_train_label_available_at": train["__label_available_at__"].max().isoformat() if len(train) else None,
        }
        if status == "skipped":
            audits.append(audit)
            continue
        if not bool(train["__label_available_at__"].lt(month).all()):
            raise AssertionError("MC1 mapper fit received a non-prequential label")
        part = held.copy()
        part["K0_native_train_p80_expected_net_bps"] = _finite(part["k0_mda90_bps"]).to_numpy(float)
        part["K0_absolute_50bps_expected_net_bps"] = _finite(part["k0_mda90_bps"]).to_numpy(float)
        for arm, features in ARM_FEATURES.items():
            if arm.startswith("K0_"):
                continue
            prediction = _fit_predict(train, held, features)
            # Preserve the frozen long MC1 architecture: a static shallow
            # mapper owns structural calibration; only the full equivalent
            # receives the causal equal-day 21d residual level adjustment.
            if arm in {"MC1_agreement_global_shift", "MC1_agreement_correctness"}:
                prediction += held["state_global_residual21_bps"].to_numpy(float)
            part[f"{arm}_expected_net_bps"] = prediction
        rows.append(part)
        audit.update({"status": "complete", "score_feature_rows": int(len(train)), "admission_rule": f"expected net >= {ADMISSION_BPS:.0f} bps"})
        audits.append(audit)
    if not rows:
        raise RuntimeError("MC1-equivalent mapper produced no supported strict-OOS folds")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def _tail_cvar(values: np.ndarray, fraction: float = .10) -> float:
    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not len(finite):
        return float("nan")
    return float(finite[:max(1, int(math.ceil(len(finite) * fraction)))].mean())


def _metrics(part: pd.DataFrame, arm: str) -> dict[str, Any]:
    if arm == "K0_native_train_p80":
        selected = part.loc[_finite(part["k0_mda90_bps"]).ge(_finite(part[K0_THRESHOLD]))].copy()
        threshold = "outer-train inner-OOF p80"
        prediction = "k0_mda90_bps"
    else:
        prediction = f"{arm}_expected_net_bps"
        selected = part.loc[_finite(part[prediction]).ge(ADMISSION_BPS)].copy()
        threshold = f"absolute {ADMISSION_BPS:.0f} bps"
    valid = selected.loc[_valid_label(selected)].copy()
    net = _finite(valid["policy_net_bps"]).to_numpy(float)
    all_valid = part.loc[_valid_label(part)].copy()
    score = _finite(all_valid[prediction]).to_numpy(float)
    outcome = _finite(all_valid["policy_net_bps"]).to_numpy(float)
    rank_ic = pd.Series(score).corr(pd.Series(outcome), method="spearman") if len(all_valid) >= 5 else float("nan")
    return {
        "arm": arm,
        "admission_rule": threshold,
        "scored_candidates": int(len(part)),
        "selected_candidates": int(len(selected)),
        "outcome_known_candidates": int(len(valid)),
        "outcome_coverage": float(len(valid) / len(selected)) if len(selected) else float("nan"),
        "net_bps_per_trade": float(np.mean(net)) if len(net) else float("nan"),
        "total_net_bps": float(np.sum(net)) if len(net) else 0.0,
        "cvar10_bps": _tail_cvar(net),
        "positive_fraction": float(np.mean(net > 0.0)) if len(net) else float("nan"),
        "all_valid_rank_ic": float(rank_ic) if np.isfinite(rank_ic) else float("nan"),
    }


def _metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly: list[dict[str, Any]] = []
    era: list[dict[str, Any]] = []
    for arm in ARM_FEATURES:
        for month, part in predictions.groupby("held_month", sort=True):
            row = _metrics(part, arm)
            row["period"] = str(month)
            monthly.append(row)
        for year, part in predictions.groupby(predictions["__decision_ts__"].dt.year, sort=True):
            row = _metrics(part, arm)
            row["period"] = str(year)
            month_rows = [item for item in monthly if item["arm"] == arm and str(item["period"]).startswith(str(year) + "-")]
            row["months"] = len(month_rows)
            row["positive_months"] = int(sum(item["net_bps_per_trade"] > 0.0 for item in month_rows if np.isfinite(item["net_bps_per_trade"])))
            row["worst_month_net_bps_per_trade"] = float(min((item["net_bps_per_trade"] for item in month_rows if np.isfinite(item["net_bps_per_trade"])), default=np.nan))
            era.append(row)
        row = _metrics(predictions, arm)
        row["period"] = "all_supported"
        row["months"] = int(predictions["held_month"].nunique())
        month_rows = [item for item in monthly if item["arm"] == arm]
        row["positive_months"] = int(sum(item["net_bps_per_trade"] > 0.0 for item in month_rows if np.isfinite(item["net_bps_per_trade"])))
        row["worst_month_net_bps_per_trade"] = float(min((item["net_bps_per_trade"] for item in month_rows if np.isfinite(item["net_bps_per_trade"])), default=np.nan))
        era.append(row)
    return pd.DataFrame(monthly), pd.DataFrame(era)


def _table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    local = frame.loc[:, [column for column in columns if column in frame]].copy()
    if local.empty:
        return "_No supported rows._"
    try:
        return local.to_markdown(index=False)
    except ImportError:
        header = " | ".join(local.columns)
        return "\n".join([header, " | ".join(["---"] * len(local.columns)), *[" | ".join(map(str, row)) for row in local.itertuples(index=False, name=None)]])


def _write_report(out: Path, *, era: pd.DataFrame, monthly: pd.DataFrame, audit: pd.DataFrame, manifest: dict[str, Any]) -> None:
    report = [
        "# Short MDA90 × T5 × K0 MC1-equivalent mapper",
        "",
        "Research-only. This is a strictly-prequential calibration/reliability test above frozen short MDA90 opportunity, T5 conversion, and K0. It does not alter short live authority or any canonical stack.",
        "",
        "## Supported strict-OOS economics",
        "",
        _table(era, ["arm", "period", "months", "selected_candidates", "net_bps_per_trade", "total_net_bps", "positive_months", "worst_month_net_bps_per_trade", "cvar10_bps", "all_valid_rank_ic"]),
        "",
        "## Monthly economics",
        "",
        _table(monthly, ["arm", "period", "selected_candidates", "net_bps_per_trade", "total_net_bps", "cvar10_bps", "all_valid_rank_ic"]),
        "",
        "## Causal contract",
        "",
        "- Upstream MDA90/T5/K0 outputs are the completed monthly strict-OOS two-stage ledger.",
        "- Every MC mapper fit uses only earlier OOS rows with `label_available_at < held_month_start`.",
        "- Agreement inputs are target-free K0/opportunity outputs from the six frozen binary opportunity heads.",
        "- Recent correctness uses only exact policy outcomes resolved strictly before each decision; invalid/incomplete paths never enter history.",
        "- `MC1_agreement_global_shift` and `MC1_agreement_correctness` receive the fixed long-MC1-style equal-day 21-day trimmed residual adjustment after the depth-2 mapper.",
        "- MC variants use expected net >= +50 bps. `K0_native_train_p80` is retained unchanged as the original control; `K0_absolute_50bps` isolates the threshold difference.",
        "",
        "## Fold audit",
        "",
        _table(audit, ["held_month", "status", "held_targetfree_rows", "outer_oof_train_rows", "outer_oof_train_months", "reason"]),
        "",
        "```json",
        json.dumps({key: manifest[key] for key in ("schema", "side", "scope", "upstream", "model", "admission", "causality")}, indent=2),
        "```",
        "",
    ]
    (out / "SHORT_P0_MDA90_T5_K0_MC1_EQUIVALENT_REPORT.md").write_text("\n".join(report))


def run(*, source: Path, context_roots: Sequence[Path], out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    frame, provenance = _load_wide(source, context_roots)
    frame = _causal_correctness_state(frame)
    predictions, audit = _monthly_predictions(frame)
    monthly, era = _metric_tables(predictions)
    out.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "side": SIDE,
        "scope": "research-only short MC1-equivalent atop frozen MDA90 opportunity + T5 conversion + K0; no short live/canonical authority",
        "upstream": {
            "artifact": str(source),
            "base_arm": BASE_ARM,
            "conversion": "frozen T5 low-regret conversion",
            "combiner": "frozen K0 analytic mixture",
            "agreement_arms": list(AGREEMENT_ARMS),
            **provenance,
        },
        "model": {"type": "HistGradientBoostingRegressor", **MC1_PARAMS, "variants": {name: list(fields) for name, fields in ARM_FEATURES.items() if fields}},
        "admission": {"mc_variants": f"expected policy net >= {ADMISSION_BPS:.0f} bps", "native_k0_control": "original outer-training inner-OOF p80 threshold"},
        "causality": {
            "upstream": "each MDA90/T5/K0 source score is outer strict-OOS",
            "mapper_fit": "rows have label_available_at < held month start",
            "correctness": f"prior-resolved only, strict < decision, equal-day {RECENT_DAYS}d with {RECENT_TRIM:.0%} tail trimming",
            "invalid_paths": "scored target-free but excluded from mapper target, correctness history, and economic metrics",
            "forbidden": ["held outcomes", "held-percentile admission", "MFE route", "live/canonical mutation"],
        },
    }
    predictions.to_parquet(out / "mc1_equivalent_outer_oof_predictions.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "mc1_equivalent_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(out / "mc1_equivalent_era_metrics.parquet", index=False, compression="zstd")
    audit.to_parquet(out / "mc1_equivalent_fold_audit.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(out, era=era, monthly=monthly, audit=audit, manifest=manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--context-root", type=Path, action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(source=args.source, context_roots=tuple(args.context_root or SOURCE_ROOTS), out=args.out)


if __name__ == "__main__":
    main()
