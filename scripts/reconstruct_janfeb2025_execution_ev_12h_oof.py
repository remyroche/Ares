#!/usr/bin/env python3
"""Reconstruct an exact-1m, two-layer execution-EV OOF panel for Jan-Feb 2025.

The archived incumbent score starts in March 2025, but the causal source-label
ledgers and immutable one-minute execution paths begin in January.  This runner
therefore builds a *new* diagnostic score rather than pretending to reproduce
the later old55/current incumbent:

1. retain only raw numeric point-in-time inputs from the source ledgers;
2. recompute the current side-parent 12-hour policy target on exact 1m paths;
3. generate a weekly side-local base OOF soft-positive score;
4. train a side-local execution-EV model only on those prior base OOF scores;
5. emit one common-unit direct-EV score for pooled global top-k diagnosis.

The first seven days warm the base model and the next seven days create the
first inner OOF meta-training rows.  Consequently strict two-layer EV OOF starts
on 2025-01-15; all of February is forward OOS.  Every fold purges unresolved
12-hour outcomes.  Feature selection, imputation and fitted models are local to
the permitted prior rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as pat
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (  # noqa: E402
    PartitionedOHLCVStore,
    read_kraken_execution_1m,
)
from extreme_price_movements.execution_ev_labels import (  # noqa: E402
    reason_names,
    simulate_execution_ev_12h,
)
from scripts.backfill_historical_execution_ev_12h_oof import (  # noqa: E402
    _current_geometry,
)


SCHEMA = "janfeb2025_execution_ev_exact1m_two_layer_oof_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
PATH_COLUMNS = ("open", "high", "low", "close")
TARGET = "execution_net_ev_12h"
BASE_TARGET = "execution_soft_positive_12h"
BASE_SCORE = "historical_base_soft_oof"
DIRECT_SCORE = "historical_direct_ev_oof"
HORIZON_MINUTES = 720
DECISION_DELAY_MINUTES = 60
BASE_WARMUP_DAYS = 7
META_WARMUP_DAYS = 14
FOLD_DAYS = 7
MAX_SELECTED_FEATURES = 40
MAX_FIT_ROWS = 100_000
MIN_BASE_TRAIN_ROWS = 10_000
MIN_META_TRAIN_ROWS = 10_000
SOFT_LABEL_SCALE_RETURN = 0.01
SEED = 202501
META_FEATURES = (
    BASE_SCORE,
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_score_z_within_timestamp",
    "base_score_rank_pct_within_timestamp",
    "candidate_group_size",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = {str(k): _safe(v) for k, v in payload.items() if k != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _utc(values: Iterable[Any] | pd.Series, *, name: str) -> pd.Series:
    result = pd.Series(pd.to_datetime(values, utc=True, errors="coerce"))
    if result.isna().any():
        raise ValueError(f"{name} contains invalid UTC timestamps")
    return result


def normalize_symbol(value: Any) -> str:
    symbol = str(value).strip().replace("/", "_")
    if not symbol:
        raise ValueError("blank symbol")
    return symbol


def eligible_raw_features(
    paths: Iterable[Path], *, minimum_features: int = 20
) -> list[str]:
    """Return the conservative numeric PIT intersection across all ledgers."""

    common: set[str] | None = None
    numeric_by_path: list[set[str]] = []
    for path in paths:
        schema = pq.read_schema(path)
        numeric = {
            field.name
            for field in schema
            if not field.name.startswith("__")
            and field.name not in {"side", "side_name", "timeframe", "candidate_id"}
            and (
                pat.is_floating(field.type)
                or pat.is_integer(field.type)
                or pat.is_boolean(field.type)
            )
        }
        numeric_by_path.append(numeric)
        common = numeric if common is None else common & numeric
    selected = sorted(common or ())
    if len(selected) < minimum_features:
        raise ValueError(f"too few common raw PIT inputs: {len(selected)}")
    return selected


def requested_months(start_month: str, end_month: str) -> list[pd.Period]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end month precedes start month")
    return list(pd.period_range(start, end, freq="M"))


def source_paths(
    labels_root: Path,
    *,
    start_month: str = "2025-01",
    end_month: str = "2025-02",
) -> list[Path]:
    months = requested_months(start_month, end_month)
    paths = [
        labels_root
        / f"train_global_{side}_5_{month.year}_{month.month:02d}.parquet"
        for side in ("long", "short")
        for month in months
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing requested source ledgers: {missing}")
    return paths


def load_candidates(
    labels_root: Path,
    *,
    start_month: str = "2025-01",
    end_month: str = "2025-02",
    symbol_allowlist: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[Path]]:
    paths = source_paths(
        labels_root, start_month=start_month, end_month=end_month
    )
    features = eligible_raw_features(paths)
    use = [*IDENTITY, "__decision_ts__", *features]
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=use,
            filters=(
                [("__symbol__", "in", sorted(symbol_allowlist))]
                if symbol_allowlist
                else None
            ),
        )
        frame["__source_file__"] = str(path)
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    result["__ts__"] = _utc(result["__ts__"], name="__ts__").to_numpy()
    result["execution_decision_utc"] = _utc(
        result.pop("__decision_ts__"), name="__decision_ts__"
    ).to_numpy()
    expected = result["__ts__"] + pd.Timedelta(minutes=DECISION_DELAY_MINUTES)
    if not result["execution_decision_utc"].equals(expected):
        raise ValueError("source decision timestamp is not signal + 60 minutes")
    result["__symbol__"] = result["__symbol__"].map(normalize_symbol)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    if not result["side_name"].isin(("long", "short")).all():
        raise ValueError("source sides are not canonical long/short")
    if result.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("source ledgers contain duplicate exact identities")
    for column in features:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            np.float32
        )
    # Consolidate the blocks created by the narrow dtype normalization before
    # appending execution-contract columns.
    result = result.copy()
    result["execution_label_end_utc"] = result[
        "execution_decision_utc"
    ] + pd.Timedelta(minutes=HORIZON_MINUTES)
    result["candidate_month"] = result["__ts__"].dt.strftime("%Y-%m")
    result = result.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    return result, features, paths


def _atr_fraction_at_decision(
    store: PartitionedOHLCVStore, symbol: str, decisions: pd.Series
) -> np.ndarray:
    start = decisions.min() - pd.Timedelta(hours=20)
    end = decisions.max() + pd.Timedelta(hours=1)
    bars = store.load(
        symbol,
        columns=["high", "low", "close"],
        start_ts=start,
        end_ts=end,
    )
    if bars.empty:
        return np.full(len(decisions), np.nan, dtype=np.float32)
    bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
    bars.index = pd.to_datetime(bars.index, utc=True)
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    prior = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prior).abs(), (low - prior).abs()], axis=1
    ).max(axis=1)
    atr_fraction = true_range.rolling(14, min_periods=14).mean() / close
    return (
        atr_fraction.reindex(pd.DatetimeIndex(decisions))
        .to_numpy(dtype=np.float32)
    )


def _minute_grid(
    data_root: Path, symbol: str, decisions: pd.Series
) -> tuple[np.ndarray, pd.Timestamp]:
    start = pd.Timestamp(decisions.min())
    end = pd.Timestamp(decisions.max()) + pd.Timedelta(minutes=HORIZON_MINUTES)
    bars = read_kraken_execution_1m(data_root, symbol, start=start, end=end)
    if bars.empty:
        return np.empty((0, len(PATH_COLUMNS)), dtype=np.float32), start
    if "ts" in bars.columns:
        bars = bars.set_index("ts")
    bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars = bars.loc[~bars.index.isna()]
    bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
    grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
    values = (
        bars.reindex(grid)
        .loc[:, list(PATH_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    return values, start


def materialize_exact_labels(
    candidates: pd.DataFrame,
    *,
    data_root: Path,
    hourly_root: Path,
    policy: Path,
    batch_rows: int,
    coverage_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    long_geometry, short_geometry, fee = _current_geometry(policy)
    hourly_store = PartitionedOHLCVStore(str(hourly_root), timeframe="1h")
    label_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    groups = candidates.groupby("__symbol__", sort=True).groups
    for number, (symbol, indices) in enumerate(groups.items(), start=1):
        rows = candidates.loc[list(indices)].copy().reset_index(drop=True)
        decisions = rows["execution_decision_utc"]
        atr = _atr_fraction_at_decision(hourly_store, str(symbol), decisions)
        values, start = _minute_grid(data_root, str(symbol), decisions)
        offsets = (
            ((decisions - start) / pd.Timedelta(minutes=1))
            .astype(np.int64)
            .to_numpy()
        )
        path_complete = np.zeros(len(rows), dtype=bool)
        if len(values):
            ends = offsets + HORIZON_MINUTES
            in_range = (offsets >= 0) & (ends <= len(values))
            good = (
                np.isfinite(values).all(axis=1)
                & (values > 0.0).all(axis=1)
                & (values[:, 1] >= values[:, 2])
            )
            prefix = np.concatenate(([0], np.cumsum(good, dtype=np.int64)))
            path_complete[in_range] = (
                prefix[ends[in_range]] - prefix[offsets[in_range]]
                == HORIZON_MINUTES
            )
        atr_complete = np.isfinite(atr) & (atr > 0)
        complete = path_complete & atr_complete
        coverage = rows.loc[:, [*IDENTITY, "candidate_month"]].copy()
        coverage["complete_exact_1m_path"] = path_complete
        coverage["complete_causal_atr"] = atr_complete
        coverage["complete_exact_label"] = complete
        coverage_parts.append(coverage)
        if coverage_only or not complete.any():
            continue
        valid_positions = np.flatnonzero(complete)
        for begin in range(0, len(valid_positions), batch_rows):
            positions = valid_positions[begin : begin + batch_rows]
            local = rows.iloc[positions].copy().reset_index(drop=True)
            local_offsets = offsets[positions]
            arrays = tuple(
                np.stack(
                    [
                        values[offset : offset + HORIZON_MINUTES, column]
                        for offset in local_offsets
                    ]
                )
                for column in range(len(PATH_COLUMNS))
            )
            gross, net, reason, exit_bar, mfe, mae = simulate_execution_ev_12h(
                *arrays,
                np.where(local["side_name"].eq("long"), 1.0, -1.0).astype(
                    np.float64
                ),
                atr[positions].astype(np.float64),
                np.full(len(local), fee, dtype=np.float64),
                long_geometry.vector(),
                short_geometry.vector(),
                1,
            )
            local["atr_fraction_14h"] = atr[positions]
            local["execution_gross_ev_12h"] = gross.astype(np.float32)
            local[TARGET] = net.astype(np.float32)
            local["execution_cost_return"] = np.float32(fee)
            local["execution_exit_reason"] = reason_names(reason)
            local["execution_exit_minute"] = (exit_bar + 1).astype(np.int16)
            local["execution_mfe_return_12h"] = mfe.astype(np.float32)
            local["execution_mae_return_12h"] = mae.astype(np.float32)
            label_parts.append(local)
        if number == 1 or number % 25 == 0 or number == len(groups):
            print(
                f"[janfeb-exact1m] {number}/{len(groups)} {symbol} "
                f"complete={int(complete.sum())}/{len(rows)}",
                flush=True,
            )
    coverage = pd.concat(coverage_parts, ignore_index=True)
    geometry = {
        "round_trip_fee_return": fee,
        "long": long_geometry.vector().tolist(),
        "short": short_geometry.vector().tolist(),
    }
    if coverage_only:
        return pd.DataFrame(), coverage, geometry
    if not label_parts:
        raise ValueError("no exact one-minute labels materialized")
    labels = pd.concat(label_parts, ignore_index=True)
    labels = labels.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if labels.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("materialized labels contain duplicate identities")
    if not np.allclose(
        labels["execution_gross_ev_12h"] - labels["execution_cost_return"],
        labels[TARGET],
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("gross-cost reconciliation failed")
    clipped = np.clip(
        labels[TARGET].to_numpy(np.float64) / SOFT_LABEL_SCALE_RETURN, -30.0, 30.0
    )
    labels[BASE_TARGET] = (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)
    return labels, coverage, geometry


def load_external_deployed_policy_labels(
    candidates: pd.DataFrame,
    *,
    labels_path: Path | Sequence[Path],
    manifest_path: Path | Sequence[Path],
    policy_path: Path,
    spread_baseline_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach an immutable canonical-simulator label panel to raw PIT rows.

    This keeps the original fee-only reconstruction reproducible while allowing
    a distinct current-spread counterfactual target.  It fails closed unless
    the external artifact proves simulator, policy, spread and universal
    side-parent-fallback parity.
    """

    label_paths = (
        [labels_path] if isinstance(labels_path, Path) else list(labels_path)
    )
    manifest_paths = (
        [manifest_path] if isinstance(manifest_path, Path) else list(manifest_path)
    )
    if not label_paths or len(label_paths) != len(manifest_paths):
        raise ValueError("external label and manifest panel counts differ")
    expected_simulator = (
        "extreme_price_movements.simple_policy_optimiser.simulate_and_score"
    )
    panel_contracts: list[dict[str, Any]] = []
    manifests: list[Mapping[str, Any]] = []
    for label_panel, manifest_panel in zip(label_paths, manifest_paths):
        manifest = json.loads(manifest_panel.read_text(encoding="utf-8"))
        manifests.append(manifest)
        accounting = manifest.get("accounting", {})
        source = manifest.get("source", {})
        geometry = manifest.get("geometry", {})
        exit_contract = manifest.get("exit_policy_contract", {})
        checks = {
            "simulator": accounting.get("simulator") == expected_simulator,
            "policy_sha256": source.get("policy_sha256") == _sha256(policy_path),
            "spread_sha256": accounting.get("spread_baseline_sha256")
            == _sha256(spread_baseline_path),
            "side_parent_fallback": (
                float(geometry.get("fallback_rate", -1.0)) == 1.0
                and int(geometry.get("side_archetype_rows", -1)) == 0
            ),
            "horizon_minutes": int(exit_contract.get("horizon_minutes", -1))
            == HORIZON_MINUTES,
        }
        failed = sorted(key for key, value in checks.items() if not value)
        if failed:
            raise ValueError(
                f"external deployed-policy label contract failed for "
                f"{label_panel}: {failed}"
            )
        panel_contracts.append(
            {
                "labels": str(label_panel),
                "labels_sha256": _sha256(label_panel),
                "manifest": str(manifest_panel),
                "manifest_sha256": _sha256(manifest_panel),
                "checks": checks,
                "source_geometry": geometry,
            }
        )
    required = [
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        TARGET,
        "execution_cost_return",
        "execution_exit_reason",
        "execution_exit_hour",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
    ]
    external = pd.concat(
        [pd.read_parquet(path, columns=required) for path in label_paths],
        ignore_index=True,
    )
    external["__ts__"] = _utc(external["__ts__"], name="labels.__ts__").to_numpy()
    external["side_name"] = external["side_name"].astype(str).str.lower()
    external["candidate_id"] = external["candidate_id"].astype(str)
    external["__symbol__"] = external["__symbol__"].map(normalize_symbol)
    if external.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("external deployed-policy labels contain duplicate identities")
    for column in ("execution_decision_utc", "execution_label_end_utc"):
        external[column] = _utc(external[column], name=f"labels.{column}").to_numpy()
    expected_decision = external["__ts__"] + pd.Timedelta(
        minutes=DECISION_DELAY_MINUTES
    )
    expected_end = expected_decision + pd.Timedelta(minutes=HORIZON_MINUTES)
    if (
        not external["execution_decision_utc"].equals(expected_decision)
        or not external["execution_label_end_utc"].equals(expected_end)
    ):
        raise ValueError("external deployed-policy label timestamps violate 12h contract")
    numeric = [
        "execution_gross_ev_12h",
        TARGET,
        "execution_cost_return",
        "execution_exit_hour",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
    ]
    for column in numeric:
        external[column] = pd.to_numeric(external[column], errors="coerce")
    if not np.isfinite(external[numeric].to_numpy(np.float64)).all():
        raise ValueError("external deployed-policy labels contain nonfinite economics")
    if not np.allclose(
        external["execution_gross_ev_12h"] - external["execution_cost_return"],
        external[TARGET],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("external deployed-policy gross-cost reconciliation failed")

    candidate_work = candidates.copy()
    candidate_work["__symbol__"] = candidate_work["__symbol__"].map(normalize_symbol)
    identity = candidate_work.loc[:, list(IDENTITY) + ["candidate_month"]].copy()
    candidate_inputs = candidate_work.drop(
        columns=["execution_decision_utc", "execution_label_end_utc"],
        errors="ignore",
    )
    joined = candidate_inputs.merge(
        external, on=list(IDENTITY), how="inner", validate="one_to_one"
    )
    if joined.empty:
        raise ValueError("external deployed-policy labels do not join source candidates")
    joined["atr_fraction_14h"] = np.nan
    joined["execution_exit_minute"] = np.rint(
        joined["execution_exit_hour"].to_numpy(np.float64) * 60.0
    ).astype(np.int16)
    clipped = np.clip(
        joined[TARGET].to_numpy(np.float64) / SOFT_LABEL_SCALE_RETURN, -30.0, 30.0
    )
    joined[BASE_TARGET] = (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)

    label_keys = external.loc[:, list(IDENTITY)].assign(__has_label__=True)
    coverage = identity.merge(
        label_keys, on=list(IDENTITY), how="left", validate="one_to_one"
    )
    has_label = coverage.pop("__has_label__").eq(True)
    coverage["complete_exact_1m_path"] = has_label
    coverage["complete_causal_atr"] = has_label
    coverage["complete_exact_label"] = has_label
    contract = {
        "mode": "external_current_spread_counterfactual",
        "panels": panel_contracts,
        "economic_interpretation": (
            "current frozen per-asset spread counterfactual on historical paths; "
            "not contemporaneous historical execution-cost evidence"
        ),
    }
    return (
        joined.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True),
        coverage,
        contract,
    )


def deterministic_fit_sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    positions = np.linspace(0, len(frame) - 1, maximum, dtype=np.int64)
    return frame.iloc[positions].copy()


def select_features(
    train: pd.DataFrame, features: list[str], *, maximum: int = MAX_SELECTED_FEATURES
) -> tuple[list[str], dict[str, float]]:
    sample = deterministic_fit_sample(
        train.sort_values(["__ts__", "__symbol__"], kind="stable"), MAX_FIT_ROWS
    )
    y = pd.to_numeric(sample[BASE_TARGET], errors="coerce")
    scores: dict[str, float] = {}
    for column in features:
        x = pd.to_numeric(sample[column], errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 100 or x.loc[valid].nunique(dropna=True) < 2:
            scores[column] = 0.0
            continue
        value = x.loc[valid].corr(y.loc[valid], method="spearman")
        scores[column] = abs(float(value)) if np.isfinite(value) else 0.0
    selected = sorted(features, key=lambda item: (-scores[item], item))[:maximum]
    if not selected or max(scores.values(), default=0.0) <= 0.0:
        raise ValueError("fold-local feature selection found no usable raw inputs")
    return selected, scores


def _fit_matrix(
    train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    medians = (
        train.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .median()
        .fillna(0.0)
    )
    x_train = (
        train.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .to_numpy(np.float32)
    )
    x_eval = (
        evaluation.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .to_numpy(np.float32)
    )
    return x_train, x_eval, {str(k): float(v) for k, v in medians.items()}


def weekly_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    values: list[pd.Timestamp] = []
    current = start
    while current < end:
        values.append(current)
        current += pd.Timedelta(days=FOLD_DAYS)
    return values


def _model(*, direct: bool) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=100 if direct else 80,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1e-3,
        random_state=SEED + int(direct),
    )


def add_candidate_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby(["__ts__", "side_name"], sort=False)[BASE_SCORE]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    cutoff = grouped.transform(lambda values: values.quantile(0.70))
    result["base_margin_to_cutoff"] = result[BASE_SCORE] - cutoff
    result["base_margin_to_cutoff_z"] = result["base_margin_to_cutoff"] / std
    result["base_score_z_within_timestamp"] = (result[BASE_SCORE] - mean) / std
    result["base_score_rank_pct_within_timestamp"] = grouped.rank(
        method="average", pct=True
    )
    result["candidate_group_size"] = grouped.transform("size").astype(np.float32)
    for column in META_FEATURES:
        result[column] = (
            pd.to_numeric(result[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    return result


def generate_base_oof(
    labels: pd.DataFrame, raw_features: list[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    overall_start = labels["__ts__"].min().floor("D")
    overall_end = labels["__ts__"].max().ceil("D") + pd.Timedelta(days=1)
    first_eval = overall_start + pd.Timedelta(days=BASE_WARMUP_DAYS)
    for side in ("long", "short"):
        side_rows = labels.loc[labels["side_name"].eq(side)].copy()
        for fold_start in weekly_boundaries(first_eval, overall_end):
            fold_end = min(fold_start + pd.Timedelta(days=FOLD_DAYS), overall_end)
            train = side_rows.loc[
                side_rows["execution_label_end_utc"] <= fold_start
            ].copy()
            evaluation = side_rows.loc[
                (side_rows["__ts__"] >= fold_start)
                & (side_rows["__ts__"] < fold_end)
            ].copy()
            row = {
                "side": side,
                "fold_start_utc": fold_start,
                "fold_end_utc": fold_end,
                "train_rows": int(len(train)),
                "eval_rows": int(len(evaluation)),
                "max_train_label_end_utc": (
                    train["execution_label_end_utc"].max() if len(train) else None
                ),
                "status": "trained",
            }
            if len(train) < MIN_BASE_TRAIN_ROWS or evaluation.empty:
                row["status"] = "insufficient_prior_history"
                audit.append(row)
                continue
            selected, importance = select_features(train, raw_features)
            fit = deterministic_fit_sample(
                train.sort_values(["__ts__", "__symbol__"], kind="stable"),
                MAX_FIT_ROWS,
            )
            x_train, x_eval, medians = _fit_matrix(fit, evaluation, selected)
            model = _model(direct=False)
            model.fit(x_train, fit[BASE_TARGET].to_numpy(np.float32))
            evaluation[BASE_SCORE] = model.predict(x_eval).astype(np.float32)
            evaluation["base_oof_fold_start_utc"] = fold_start
            evaluation["base_oof_train_cutoff_utc"] = fold_start
            outputs.append(evaluation)
            row.update(
                {
                    "fit_rows": int(len(fit)),
                    "selected_features": selected,
                    "selection_abs_spearman": {
                        key: importance[key] for key in selected
                    },
                    "median_fill": medians,
                }
            )
            audit.append(row)
    if not outputs:
        raise ValueError("no base OOF folds trained")
    result = pd.concat(outputs, ignore_index=True)
    return add_candidate_context(result), audit


def generate_direct_ev_oof(
    base_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    overall_start = base_oof["__ts__"].min().floor("D") - pd.Timedelta(
        days=BASE_WARMUP_DAYS
    )
    overall_end = base_oof["__ts__"].max().ceil("D") + pd.Timedelta(days=1)
    first_eval = overall_start + pd.Timedelta(days=META_WARMUP_DAYS)
    for side in ("long", "short"):
        side_rows = base_oof.loc[base_oof["side_name"].eq(side)].copy()
        for fold_start in weekly_boundaries(first_eval, overall_end):
            fold_end = min(fold_start + pd.Timedelta(days=FOLD_DAYS), overall_end)
            train = side_rows.loc[
                side_rows["execution_label_end_utc"] <= fold_start
            ].copy()
            evaluation = side_rows.loc[
                (side_rows["__ts__"] >= fold_start)
                & (side_rows["__ts__"] < fold_end)
            ].copy()
            row = {
                "side": side,
                "fold_start_utc": fold_start,
                "fold_end_utc": fold_end,
                "train_rows": int(len(train)),
                "eval_rows": int(len(evaluation)),
                "max_train_label_end_utc": (
                    train["execution_label_end_utc"].max() if len(train) else None
                ),
                "status": "trained",
            }
            if len(train) < MIN_META_TRAIN_ROWS or evaluation.empty:
                row["status"] = "insufficient_prior_base_oof_history"
                audit.append(row)
                continue
            fit = deterministic_fit_sample(
                train.sort_values(["__ts__", "__symbol__"], kind="stable"),
                MAX_FIT_ROWS,
            )
            x_train, x_eval, medians = _fit_matrix(
                fit, evaluation, list(META_FEATURES)
            )
            model = _model(direct=True)
            model.fit(x_train, fit[TARGET].to_numpy(np.float32))
            evaluation[DIRECT_SCORE] = model.predict(x_eval).astype(np.float32)
            evaluation["direct_oof_fold_start_utc"] = fold_start
            evaluation["direct_oof_train_cutoff_utc"] = fold_start
            outputs.append(evaluation)
            row.update({"fit_rows": int(len(fit)), "median_fill": medians})
            audit.append(row)
    if not outputs:
        raise ValueError("no strict two-layer execution-EV OOF folds trained")
    result = pd.concat(outputs, ignore_index=True)
    return (
        result.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True),
        audit,
    )


def topk_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    def select(group: pd.DataFrame) -> pd.DataFrame:
        n = max(1, int(np.ceil(0.10 * len(group))))
        return group.nlargest(n, DIRECT_SCORE)

    def summarize(selected: pd.DataFrame, *, candidate_rows: int) -> dict[str, Any]:
        return {
            "candidate_rows": int(candidate_rows),
            "selected_rows": int(len(selected)),
            "mean_net_ev_bps": float(selected[TARGET].mean() * 1e4),
            "sum_net_return": float(selected[TARGET].sum()),
            "positive_rate": float((selected[TARGET] > 0.0).mean()),
            "long_rows": int(selected["side_name"].eq("long").sum()),
            "short_rows": int(selected["side_name"].eq("short").sum()),
        }

    global_selected = select(frame)
    return {
        "global": summarize(global_selected, candidate_rows=len(frame)),
        "global_book_by_month": {
            str(key): summarize(group, candidate_rows=int(frame["candidate_month"].eq(key).sum()))
            for key, group in global_selected.groupby("candidate_month", sort=True)
        },
        "global_book_by_week": {
            str(key): summarize(
                group,
                candidate_rows=int(
                    frame["__ts__"]
                    .dt.tz_localize(None)
                    .dt.to_period("W-SUN")
                    .astype(str)
                    .eq(key)
                    .sum()
                ),
            )
            for key, group in global_selected.assign(
                __week__=global_selected["__ts__"]
                .dt.tz_localize(None)
                .dt.to_period("W-SUN")
                .astype(str)
            ).groupby("__week__", sort=True)
        },
        "global_book_by_side": {
            str(key): summarize(
                group, candidate_rows=int(frame["side_name"].eq(key).sum())
            )
            for key, group in global_selected.groupby("side_name", sort=True)
        },
        "diagnostic_month_local_pooled_top10": {
            str(key): summarize(select(group), candidate_rows=len(group))
            for key, group in frame.groupby("candidate_month", sort=True)
        },
    }


def coverage_by_side_month(coverage: pd.DataFrame) -> pd.DataFrame:
    table = (
        coverage.groupby(["candidate_month", "side_name"], sort=True)
        .agg(
            candidate_rows=("candidate_id", "size"),
            complete_exact_1m_paths=("complete_exact_1m_path", "sum"),
            complete_causal_atr=("complete_causal_atr", "sum"),
            complete_exact_labels=("complete_exact_label", "sum"),
        )
        .reset_index()
    )
    table["exact_1m_path_coverage"] = (
        table["complete_exact_1m_paths"] / table["candidate_rows"].clip(lower=1)
    )
    table["causal_atr_coverage"] = (
        table["complete_causal_atr"] / table["candidate_rows"].clip(lower=1)
    )
    table["exact_label_coverage"] = (
        table["complete_exact_labels"] / table["candidate_rows"].clip(lower=1)
    )
    return table


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    source_universe: dict[str, Any] = {"mode": "all_source_symbols"}
    symbols: set[str] | None = None
    if args.symbol_allowlist is not None:
        symbols = {
            line.strip()
            for line in args.symbol_allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not symbols:
            raise ValueError("symbol allowlist is empty")
        source_universe = {
            "mode": "frozen_symbol_allowlist",
            "path": str(args.symbol_allowlist),
            "sha256": _sha256(args.symbol_allowlist),
            "symbols": sorted(symbols),
            "symbol_count": len(symbols),
        }
    candidates, raw_features, sources = load_candidates(
        args.labels_root,
        start_month=args.start_month,
        end_month=args.end_month,
        symbol_allowlist=symbols,
    )
    if candidates.empty:
        raise ValueError("symbol allowlist removes every source candidate")
    if args.external_labels is not None:
        if args.external_label_manifest is None or args.spread_baseline is None:
            raise ValueError(
                "--external-labels requires --external-label-manifest and "
                "--spread-baseline"
            )
        if args.coverage_only:
            raise ValueError(
                "--coverage-only is not supported with immutable external labels"
            )
        labels, coverage, geometry = load_external_deployed_policy_labels(
            candidates,
            labels_path=args.external_labels,
            manifest_path=args.external_label_manifest,
            policy_path=args.policy,
            spread_baseline_path=args.spread_baseline,
        )
    else:
        labels, coverage, geometry = materialize_exact_labels(
            candidates,
            data_root=args.data_root,
            hourly_root=args.hourly_root,
            policy=args.policy,
            batch_rows=args.batch_rows,
            coverage_only=args.coverage_only,
        )
    coverage_path = args.output_dir / "coverage_by_side_month.csv"
    coverage_table = coverage_by_side_month(coverage)
    coverage_table.to_csv(coverage_path, index=False)
    coverage_gate_column = (
        "exact_label_coverage"
        if args.external_labels is not None
        else "exact_1m_path_coverage"
    )
    coverage_gate_minimum = (
        float(args.minimum_external_label_coverage)
        if args.external_labels is not None
        else float(args.minimum_path_coverage)
    )
    coverage_gate_rows = coverage_table
    intentionally_unlabeled = pd.DataFrame(columns=coverage_table.columns)
    if args.external_labels is not None and args.allow_external_label_gaps:
        intentionally_unlabeled = coverage_table.loc[
            coverage_table["complete_exact_labels"].eq(0)
        ].copy()
        coverage_gate_rows = coverage_table.loc[
            coverage_table["complete_exact_labels"].gt(0)
        ]
    failed_path_coverage = coverage_gate_rows.loc[
        coverage_gate_rows[coverage_gate_column] < coverage_gate_minimum
    ]
    preflight = {
        "schema": "historical_execution_ev_exact1m_coverage_preflight_v1",
        "requested_start_month": args.start_month,
        "requested_end_month": args.end_month,
        "candidate_rows": int(len(candidates)),
        "raw_feature_count": int(len(raw_features)),
        "minimum_path_coverage": float(args.minimum_path_coverage),
        "coverage_gate_column": coverage_gate_column,
        "coverage_gate_minimum": coverage_gate_minimum,
        "all_side_months_pass_path_gate": bool(failed_path_coverage.empty),
        "failed_side_months": failed_path_coverage.to_dict(orient="records"),
        "allow_external_label_gaps": bool(args.allow_external_label_gaps),
        "intentionally_unlabeled_side_months": intentionally_unlabeled.to_dict(
            orient="records"
        ),
        "coverage_only": bool(args.coverage_only),
        "coverage": coverage_table.to_dict(orient="records"),
        "source_ledgers": [str(path) for path in sources],
    }
    preflight_path = args.output_dir / "coverage_preflight.json"
    _write_json(preflight_path, preflight)
    if not failed_path_coverage.empty:
        raise ValueError(
            "candidate-level exact-1m path coverage gate failed; inspect "
            f"{preflight_path}"
        )
    if args.coverage_only:
        return {
            "coverage": coverage_path,
            "preflight": preflight_path,
        }
    base_oof, base_audit = generate_base_oof(labels, raw_features)
    direct_oof, direct_audit = generate_direct_ev_oof(base_oof)

    label_columns = [
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "candidate_month",
        "atr_fraction_14h",
        "execution_gross_ev_12h",
        TARGET,
        "execution_cost_return",
        "execution_exit_reason",
        "execution_exit_minute",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        BASE_TARGET,
    ]
    labels_path = args.output_dir / "exact_1m_execution_ev_12h_labels.parquet"
    oof_path = args.output_dir / "two_layer_direct_ev_strict_oof.parquet"
    base_audit_path = args.output_dir / "base_fold_audit.json"
    direct_audit_path = args.output_dir / "direct_ev_fold_audit.json"
    labels.loc[:, label_columns].to_parquet(
        labels_path, index=False, compression="zstd"
    )
    keep_oof = [
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "candidate_month",
        TARGET,
        BASE_TARGET,
        *META_FEATURES,
        DIRECT_SCORE,
        "base_oof_fold_start_utc",
        "base_oof_train_cutoff_utc",
        "direct_oof_fold_start_utc",
        "direct_oof_train_cutoff_utc",
    ]
    direct_oof.loc[:, list(dict.fromkeys(keep_oof))].to_parquet(
        oof_path, index=False, compression="zstd"
    )
    _write_json(base_audit_path, {"folds": base_audit})
    _write_json(direct_audit_path, {"folds": direct_audit})
    artifacts = {
        path.name: _sha256(path)
        for path in (
            labels_path,
            oof_path,
            coverage_path,
            preflight_path,
            base_audit_path,
            direct_audit_path,
        )
    }
    summary: dict[str, Any] = {
        "schema": (
            "historical_execution_ev_current_spread_counterfactual_two_layer_oof_v1"
            if args.external_labels is not None
            else SCHEMA
        ),
        "rows": {
            "source_candidates": int(len(candidates)),
            "exact_1m_labels": int(len(labels)),
            "base_oof": int(len(base_oof)),
            "strict_two_layer_direct_ev_oof": int(len(direct_oof)),
        },
        "period": {
            "requested_start_month": args.start_month,
            "requested_end_month": args.end_month,
            "source_start": candidates["__ts__"].min(),
            "source_end": candidates["__ts__"].max(),
            "strict_two_layer_oof_start": direct_oof["__ts__"].min(),
            "strict_two_layer_oof_end": direct_oof["__ts__"].max(),
        },
        "source": {
            "ledgers": {
                str(path): {"rows": pq.ParquetFile(path).metadata.num_rows, "sha256": _sha256(path)}
                for path in sources
            },
            "contract": "causal source candidate identities plus raw numeric PIT inputs only",
            "excluded": "all __-prefixed target, archetype, reliability, meta-selection and outcome columns",
            "incumbent_parity": "not claimed; old55/current fitted state is unavailable before March",
            "universe": source_universe,
        },
        "raw_features": raw_features,
        "target": {
            "name": TARGET,
            "simulator": (
                "simple_policy_optimiser.simulate_and_score"
                if args.external_labels is not None
                else "simulate_execution_ev_12h"
            ),
            "bar_minutes": 1,
            "horizon_minutes": HORIZON_MINUTES,
            "signal_to_decision_minutes": DECISION_DELAY_MINUTES,
            "atr": (
                "external canonical path-input eligibility lineage"
                if args.external_labels is not None
                else "14 completed hourly true-range observations at decision"
            ),
            "cost": (
                "deployed fee plus current frozen per-asset spread counterfactual"
                if args.external_labels is not None
                else "current side-parent round-trip fee exactly once"
            ),
            "geometry": geometry,
            "policy": str(args.policy),
            "policy_sha256": _sha256(args.policy),
        },
        "base": {
            "target": BASE_TARGET,
            "definition": f"sigmoid({TARGET}/{SOFT_LABEL_SCALE_RETURN})",
            "side_local": True,
            "walk_forward": f"{FOLD_DAYS}d expanding",
            "warmup_days": BASE_WARMUP_DAYS,
            "feature_selection": f"fold-local top {MAX_SELECTED_FEATURES} absolute Spearman",
            "fit_row_cap": MAX_FIT_ROWS,
        },
        "execution_ev": {
            "target": TARGET,
            "side_local": True,
            "inputs": list(META_FEATURES),
            "base_score_provenance": "inner weekly expanding OOF only",
            "walk_forward": f"{FOLD_DAYS}d expanding",
            "warmup_days": META_WARMUP_DAYS,
            "first_strict_oof": direct_oof["__ts__"].min(),
            "scored_period_status": (
                "strict expanding OOF after the reported warm-up; every fit "
                "uses only prior resolved labels"
            ),
        },
        "validation": {
            "purge": "train execution_label_end_utc <= evaluation fold start",
            "embargo": "no future rows are used; 12h unresolved-path purge is fail-closed",
            "ranking": "one pooled global top 10% after side-local common-unit direct EV",
            "promotion_status": "diagnostic reconstruction; not incumbent parity or promotion evidence",
        },
        "coverage": coverage_table.to_dict(orient="records"),
        "metrics": topk_metrics(direct_oof),
        "artifacts": artifacts,
    }
    summary["manifest_sha256"] = _canonical_hash(summary)
    summary_path = args.output_dir / "summary.json"
    _write_json(summary_path, summary)
    return {
        "labels": labels_path,
        "oof": oof_path,
        "coverage": coverage_path,
        "summary": summary_path,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--labels-root",
        type=Path,
        default=ROOT
        / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels",
    )
    result.add_argument("--data-root", type=Path, default=ROOT / "data_perp")
    result.add_argument(
        "--hourly-root",
        type=Path,
        default=ROOT / "data_perp/exchanges/krakenfutures",
    )
    result.add_argument(
        "--policy",
        type=Path,
        default=ROOT
        / "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1/production_staging/best_policy_params.json",
    )
    result.add_argument("--start-month", default="2025-01")
    result.add_argument("--end-month", default="2025-02")
    result.add_argument(
        "--coverage-only",
        action="store_true",
        help="audit candidate-level 720-minute path and ATR coverage without fitting",
    )
    result.add_argument(
        "--minimum-path-coverage",
        type=float,
        default=0.99,
        help="fail closed when any side-month exact-1m path coverage is lower",
    )
    result.add_argument(
        "--minimum-external-label-coverage",
        type=float,
        default=0.70,
        help=(
            "minimum source-candidate coverage for a pre-materialized canonical "
            "label panel; distinct from exact-path coverage inside that artifact"
        ),
    )
    result.add_argument(
        "--allow-external-label-gaps",
        action="store_true",
        help=(
            "permit explicitly reported zero-label side-month gaps between immutable "
            "external panels; partially covered months must still pass the gate"
        ),
    )
    result.add_argument("--batch-rows", type=int, default=1000)
    result.add_argument(
        "--external-labels",
        type=Path,
        nargs="+",
        help=(
            "immutable labels from materialize_execution_ev_policy_labels.py; "
            "preserves the original fee-only runner when omitted"
        ),
    )
    result.add_argument("--external-label-manifest", type=Path, nargs="+")
    result.add_argument("--spread-baseline", type=Path)
    result.add_argument("--symbol-allowlist", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.batch_rows < 1:
        raise ValueError("--batch-rows must be positive")
    if not 0.0 < arguments.minimum_path_coverage <= 1.0:
        raise ValueError("--minimum-path-coverage must lie in (0, 1]")
    if not 0.0 < arguments.minimum_external_label_coverage <= 1.0:
        raise ValueError("--minimum-external-label-coverage must lie in (0, 1]")
    print(json.dumps({key: str(value) for key, value in run(arguments).items()}, indent=2))
