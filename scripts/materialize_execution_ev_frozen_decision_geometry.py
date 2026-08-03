#!/usr/bin/env python3
"""Materialize the frozen, outcome-free side-parent decision geometry.

This is deliberately a *pre-label* adapter.  The Pack-B final-refit top-40
context supplies the exact candidate identity and decision time; canonical
hourly OHLCV supplies only data observed at the signal.  It recreates the
deployed July side-parent barrier rule:

``barrier[t] = max(WilderATR14(high, low, close)[t - 1] / close[t - 1], .005)``.

The current signal ATR is retained as an audit field, but never substituted for
the lagged value.  There is no outcome input, policy-archetype inference,
as-of join, fill, or fallback geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from scripts.materialize_packb_auxiliary_targets import wilder_atr_fraction  # noqa: E402


SCHEMA = "execution_ev_frozen_decision_geometry_v1"
POPULATION_SCHEMA = "packb_final_refits_forward_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SIDES = ("long", "short")
ATR_PERIOD = 14
MIN_BARRIER_PCT = np.float32(0.005)
DEFAULT_CONTEXT_ROOT = ROOT / (
    "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v1/packb"
)
DEFAULT_POLICY = ROOT / (
    "data_perp/artifacts/"
    "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/"
    "simple_policy_optimiser/deployment/best_policy_params_perps.json"
)
DEFAULT_OHLCV_ROOT = ROOT / "data_perp/exchanges/krakenfutures"
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v1/"
    "frozen_decision_geometry"
)


class FrozenDecisionGeometryError(RuntimeError):
    """Raised when exact decision-time geometry cannot be established."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise FrozenDecisionGeometryError(f"{name} has null or invalid UTC timestamps")
    return parsed


def _manifest_bound_context(
    context_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only an exact, outcome-free Pack-B final-refit top-40 stream."""

    if not context_path.is_file() or not manifest_path.is_file():
        raise FrozenDecisionGeometryError(
            "manifest-bound Pack-B final-refit context is required"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != POPULATION_SCHEMA
        or manifest.get("status") != "frozen_final_refit_preentry_context_not_oos_metrics"
        or manifest.get("contract", {}).get("outcomes_used") is not False
        or manifest.get("output", {}).get("sha256") != _sha256(context_path)
    ):
        raise FrozenDecisionGeometryError(
            "Pack-B context is not an exact outcome-free final-refit binding"
        )
    frame = pd.read_parquet(context_path)
    required = {
        *IDENTITY,
        "selected_top40",
        "prediction_source",
        "execution_decision_utc",
        "feature_available_at",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FrozenDecisionGeometryError(f"Pack-B context missing required columns: {missing}")
    frame = frame.copy()
    frame["__ts__"] = _utc(frame["__ts__"], name="context.__ts__")
    frame["execution_decision_utc"] = _utc(
        frame["execution_decision_utc"], name="context.execution_decision_utc"
    )
    frame["feature_available_at"] = _utc(
        frame["feature_available_at"], name="context.feature_available_at"
    )
    for column in ("__symbol__", "side_name", "candidate_id"):
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise FrozenDecisionGeometryError(f"context.{column} has blank identities")
    frame["side_name"] = frame["side_name"].str.lower()
    expected_decision = frame["__ts__"] + pd.Timedelta(hours=1)
    if (
        frame.duplicated(list(IDENTITY), keep=False).any()
        or frame["candidate_id"].duplicated().any()
        or set(frame["side_name"]) != set(SIDES)
        or not frame["selected_top40"].astype(bool).all()
        or set(frame["prediction_source"].astype(str)) != {"frozen_final_refit"}
        or not frame["execution_decision_utc"].eq(expected_decision).all()
        or (frame["feature_available_at"] > frame["execution_decision_utc"]).any()
    ):
        raise FrozenDecisionGeometryError(
            "Pack-B context identity, top40, prediction-source, or timing contract changed"
        )
    return frame.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True), manifest


def _canonical_side_parents(policy_path: Path) -> dict[str, dict[str, Any]]:
    """Require the two explicit canonical parents, never a heuristic proxy."""

    if not policy_path.is_file():
        raise FrozenDecisionGeometryError("frozen production policy JSON is required")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    contract = policy.get("exit_geometry_contract")
    if not isinstance(contract, Mapping) or str(contract.get("replay_timeframe")) != "1m":
        raise FrozenDecisionGeometryError("policy must bind exact 1m exit geometry")
    if int(contract.get("horizon_minutes", 0)) < 720:
        raise FrozenDecisionGeometryError("policy horizon must cover the 12h label timeout")
    strategies = policy.get("strategies")
    if not isinstance(strategies, list):
        raise FrozenDecisionGeometryError("policy has no strategy list")
    parents: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        exact = [
            dict(strategy)
            for strategy in strategies
            if isinstance(strategy, Mapping)
            and strategy.get("selected", True)
            and str(strategy.get("side", "")).lower() == side
            and str(strategy.get("exit_geometry_scope", "")) == "side_parent"
            and str(strategy.get("canonical_strategy_id", "")) == f"{side}__parent"
            and str(strategy.get("strategy_id", "")) == f"{side}__parent"
        ]
        if len(exact) != 1:
            raise FrozenDecisionGeometryError(
                f"policy must contain exactly one selected canonical {side}__parent strategy"
            )
        parents[side] = exact[0]
    return parents


def _store_parts(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    safe = symbol.replace("/", "_")
    directory = root / "ohlcv" / f"symbol={safe}"
    return [
        path
        for year in range(int(start.year), int(end.year) + 1)
        for path in sorted((directory / f"year={year}").glob("*.parquet"))
    ]


def _signal_and_lagged_atr(
    population: pd.DataFrame,
    *,
    ohlcv_root: Path,
    warmup_days: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str], dict[str, Any]]:
    """Return exact current and prior-hour ATR fractions for every identity.

    The signal bar and its literal previous hourly bar are required exactly.
    Historical source gaps inside the warmup remain explicit diagnostics: no
    candle is introduced, filled, or as-of joined.  Wilder's recurrence is
    evaluated on the chronological observed candle stream, so the first true
    range after a source outage spans from the prior observed close as it does
    in the canonical historical pipeline.
    """

    if warmup_days < 90:
        raise FrozenDecisionGeometryError("the frozen geometry contract requires >=90d ATR warmup")
    population = population.reset_index(drop=True)
    store = PartitionedOHLCVStore(str(ohlcv_root), timeframe="1h")
    current = np.full(len(population), np.nan, dtype=np.float32)
    lagged = np.full(len(population), np.nan, dtype=np.float32)
    source_parts: dict[str, str] = {}
    coverage_by_symbol: dict[str, dict[str, Any]] = {}
    hour = pd.Timedelta(hours=1)
    for symbol, indices in population.groupby("__symbol__", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        signals = pd.DatetimeIndex(population.loc[positions, "__ts__"])
        earliest = pd.Timestamp(signals.min())
        latest = pd.Timestamp(signals.max())
        start = earliest - pd.Timedelta(days=int(warmup_days)) - hour
        bars = store.load(str(symbol), columns=["high", "low", "close"], start_ts=start, end_ts=latest)
        parts = _store_parts(ohlcv_root, str(symbol), start, latest)
        if not parts or bars.empty:
            raise FrozenDecisionGeometryError(f"{symbol}: canonical hourly OHLCV/provenance is missing")
        for part in parts:
            source_parts[str(part.relative_to(ohlcv_root))] = _sha256(part)
        bars = bars.loc[:, ["high", "low", "close"]].copy()
        bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
        bars = bars.loc[~bars.index.isna()].sort_index()
        if bars.index.duplicated().any():
            raise FrozenDecisionGeometryError(f"{symbol}: duplicate hourly OHLC timestamps")
        if bars.index.min() > start:
            raise FrozenDecisionGeometryError(
                f"{symbol}: canonical hourly OHLCV does not reach the required {warmup_days}d warmup boundary"
            )
        expected = pd.date_range(start, latest, freq="1h", tz="UTC")
        observed_expected = bars.index[(bars.index >= start) & (bars.index <= latest)]
        missing_history = expected.difference(observed_expected)
        numeric = bars.apply(pd.to_numeric, errors="coerce")
        raw = numeric.to_numpy(dtype=np.float64)
        if (
            not np.isfinite(raw).all()
            or (raw <= 0.0).any()
            or (raw[:, 0] < raw[:, 1]).any()
        ):
            raise FrozenDecisionGeometryError(f"{symbol}: nonfinite or invalid canonical hourly OHLCV")
        atr = wilder_atr_fraction(
            numeric["high"].to_numpy(dtype=np.float64),
            numeric["low"].to_numpy(dtype=np.float64),
            numeric["close"].to_numpy(dtype=np.float64),
            period=ATR_PERIOD,
        )
        lookup = pd.Series(atr, index=bars.index)
        exact_current = lookup.reindex(signals)
        exact_lagged = lookup.reindex(signals - hour)
        values = (exact_current.to_numpy(dtype=np.float32), exact_lagged.to_numpy(dtype=np.float32))
        if any((not np.isfinite(value).all()) or (value <= 0.0).any() for value in values):
            raise FrozenDecisionGeometryError(
                f"{symbol}: current and prior hourly signal bars/ATR must exist exactly and be finite"
            )
        current[positions], lagged[positions] = values
        coverage_by_symbol[str(symbol)] = {
            "warmup_start": start,
            "last_signal": latest,
            "expected_hourly_rows": int(len(expected)),
            "observed_hourly_rows": int(len(observed_expected)),
            "historical_gap_rows": int(len(missing_history)),
            "historical_gap_first": missing_history.min() if len(missing_history) else None,
            "historical_gap_last": missing_history.max() if len(missing_history) else None,
            "exact_signal_rows": int(len(signals)),
            "exact_prior_signal_rows": int(len(signals)),
        }
    return current, lagged, source_parts, coverage_by_symbol


def build_geometry(
    population: pd.DataFrame,
    *,
    policy_path: Path,
    ohlcv_root: Path,
    warmup_days: int = 90,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one outcome-free side-parent geometry row per Pack-B candidate."""

    parents = _canonical_side_parents(policy_path)
    current, lagged, source_parts, coverage_by_symbol = _signal_and_lagged_atr(
        population, ohlcv_root=ohlcv_root, warmup_days=warmup_days
    )
    barrier = np.maximum(lagged, MIN_BARRIER_PCT).astype(np.float32)
    if not np.isfinite(barrier).all() or (barrier < MIN_BARRIER_PCT).any():
        raise FrozenDecisionGeometryError("lagged ATR barrier projection is nonfinite")
    output = population.loc[:, list(IDENTITY)].copy()
    output["__barrier_pct__"] = barrier
    output["policy_archetype"] = "side_parent"
    output["geometry_available_at"] = population["__ts__"].to_numpy()
    output["__signal_atr_fraction__"] = current
    output["__lagged_signal_atr_fraction__"] = lagged
    output["canonical_parent_strategy_id"] = output["side_name"].map(
        {side: str(parents[side]["canonical_strategy_id"]) for side in SIDES}
    )
    if output.isna().any().any() or output.duplicated(list(IDENTITY)).any():
        raise FrozenDecisionGeometryError("geometry projection does not preserve exact finite identity coverage")
    output = output.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    audit = {
        "source_parts": source_parts,
        "coverage_by_symbol": coverage_by_symbol,
        "parents": {
            side: {
                "canonical_strategy_id": parents[side]["canonical_strategy_id"],
                "strategy_id": parents[side]["strategy_id"],
                "strategy_sha256": hashlib.sha256(
                    json.dumps(_safe(parents[side]), sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for side in SIDES
        },
    }
    return output, audit


def materialize(
    *,
    context_path: Path,
    context_manifest_path: Path,
    policy_path: Path,
    ohlcv_root: Path,
    output_dir: Path,
    warmup_days: int = 90,
) -> dict[str, Path]:
    """Materialize immutable geometry for the exact Pack-B top-40 population."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    population, context_manifest = _manifest_bound_context(context_path, context_manifest_path)
    geometry, audit = build_geometry(
        population,
        policy_path=policy_path,
        ohlcv_root=ohlcv_root,
        warmup_days=warmup_days,
    )
    decision = population.loc[:, list(IDENTITY) + ["execution_decision_utc"]].merge(
        geometry.loc[:, list(IDENTITY) + ["geometry_available_at"]],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    if len(decision) != len(population) or (decision["geometry_available_at"] > decision["execution_decision_utc"]).any():
        raise FrozenDecisionGeometryError("geometry availability must be at or before every decision")
    output_dir.mkdir(parents=True, exist_ok=False)
    geometry_path = output_dir / "frozen_decision_geometry.parquet"
    geometry.to_parquet(geometry_path, index=False, compression="zstd")
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": SCHEMA,
        "status": "materialized_outcome_free_side_parent_geometry",
        "outcomes_used": False,
        "promotion_status": "retrospective_research_input_not_a_promoted_policy_change",
        "geometry_contract": {
            "policy_archetype": "side_parent",
            "parent_strategy_validation": "one selected canonical <side>__parent strategy per side",
            "barrier_formula": "max(raw-price Wilder ATR(14) / close at exact signal t-1, 0.005)",
            "signal_atr_formula": "raw-price Wilder ATR(14) / close at exact signal t",
            "atr_period": ATR_PERIOD,
            "atr_warmup_days": int(warmup_days),
            "join": "exact __symbol__ + signal timestamp and exact prior hourly timestamp; no as-of, forward-fill, backfill, interpolation, or neutral fill",
            "historical_gap_handling": "retain source gaps as audited gaps; compute Wilder recurrence on observed chronological canonical bars, while requiring exact signal and t-1 bars",
            "geometry_available_at": "signal timestamp (therefore <= signal + 1h decision)",
        },
        "inputs": {
            "packb_context": {
                "path": str(context_path),
                "sha256": _sha256(context_path),
                "manifest_path": str(context_manifest_path),
                "manifest_sha256": _sha256(context_manifest_path),
                "schema": context_manifest["schema"],
            },
            "policy": {"path": str(policy_path), "sha256": _sha256(policy_path)},
            "ohlcv": {
                "root": str(ohlcv_root),
                "source_parts": audit["source_parts"],
                "coverage_by_symbol": audit["coverage_by_symbol"],
            },
        },
        "policy_parents": audit["parents"],
        "output": {"path": str(geometry_path), "sha256": _sha256(geometry_path), "rows": int(len(geometry))},
        "rows": {
            "total": int(len(geometry)),
            "long": int(geometry["side_name"].eq("long").sum()),
            "short": int(geometry["side_name"].eq("short").sum()),
        },
    }
    _write_json(manifest_path, manifest)
    return {"geometry": geometry_path, "manifest": manifest_path}


def historical_lagged_barrier_parity(
    *,
    labels_path: Path,
    max_unique_symbol_hours: int | None = None,
) -> dict[str, Any]:
    """Audit a historical label archive against the literal lagged-ATR rule.

    This is intentionally read-only and has no policy or raw-market-data
    dependency: the historical ledger is the authoritative record of its own
    signal-time ATR.  It verifies the literal deployed invariant on *contiguous
    symbol-hours* only, never carrying an ATR across a missing candidate hour.
    """

    if not labels_path.is_file():
        raise FrozenDecisionGeometryError("historical label artifact is required for parity audit")
    labels = pd.read_parquet(
        labels_path,
        columns=[
            "__ts__",
            "__symbol__",
            "__barrier_pct__",
            "__path_auxiliary_atr_fraction__",
        ],
    )
    labels["__ts__"] = _utc(labels["__ts__"], name="historical_labels.__ts__")
    labels["__symbol__"] = labels["__symbol__"].astype("string").str.strip()
    labels["__barrier_pct__"] = pd.to_numeric(labels["__barrier_pct__"], errors="coerce")
    labels["__path_auxiliary_atr_fraction__"] = pd.to_numeric(
        labels["__path_auxiliary_atr_fraction__"], errors="coerce"
    )
    duplicates = labels.groupby(["__ts__", "__symbol__"], dropna=False).agg(
        barrier=("__barrier_pct__", "nunique"),
        atr=("__path_auxiliary_atr_fraction__", "nunique"),
    )
    if duplicates["barrier"].gt(1).any() or duplicates["atr"].gt(1).any():
        raise FrozenDecisionGeometryError(
            "historical archive assigns conflicting barrier/ATR values to a symbol-hour"
        )
    unique = labels.drop_duplicates(["__ts__", "__symbol__"], keep="first").copy()
    unique = unique.sort_values(["__symbol__", "__ts__"], kind="stable")
    if max_unique_symbol_hours is not None:
        if max_unique_symbol_hours < 1:
            raise FrozenDecisionGeometryError("max_unique_symbol_hours must be positive")
        unique = unique.head(int(max_unique_symbol_hours)).copy()
    unique["__prior_ts__"] = unique.groupby("__symbol__", sort=False)["__ts__"].shift(1)
    unique["__prior_atr__"] = unique.groupby("__symbol__", sort=False)[
        "__path_auxiliary_atr_fraction__"
    ].shift(1)
    contiguous = unique["__ts__"].sub(unique["__prior_ts__"]).eq(pd.Timedelta(hours=1))
    contiguous_rows = unique.loc[contiguous].copy()
    expected = np.maximum(
        contiguous_rows["__prior_atr__"].to_numpy(dtype=np.float32), MIN_BARRIER_PCT
    ).astype(np.float32)
    observed = contiguous_rows["__barrier_pct__"].to_numpy(dtype=np.float32)
    unequal = ~np.isclose(observed, expected, rtol=0.0, atol=0.0, equal_nan=False)
    if unequal.any():
        sample = contiguous_rows.loc[
            unequal,
            ["__ts__", "__symbol__", "__barrier_pct__", "__prior_atr__"],
        ].head(5)
        raise FrozenDecisionGeometryError(
            "historical barrier parity failed for "
            f"{int(unequal.sum())}/{len(contiguous_rows)} contiguous symbol-hours: "
            f"{sample.to_dict(orient='records')}"
        )
    return {
        "rows": int(len(contiguous_rows)),
        "mismatches": 0,
        "formula": "max(__path_auxiliary_atr_fraction__[t-1], 0.005) on exact contiguous symbol-hours",
        "labels_sha256": _sha256(labels_path),
    }


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("materialize", help="materialize frozen decision geometry")
    build.add_argument("--context", type=Path, default=DEFAULT_CONTEXT_ROOT / "packb_forward_context.parquet")
    build.add_argument("--context-manifest", type=Path, default=DEFAULT_CONTEXT_ROOT / "manifest.json")
    build.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY)
    build.add_argument("--ohlcv-root", type=Path, default=DEFAULT_OHLCV_ROOT)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--warmup-days", type=int, default=90)
    parity = sub.add_parser("historical-parity", help="audit the historical lagged-barrier invariant")
    parity.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data_perp/artifacts/path_archetype_labels_july20_20260726_v1/path_archetype_labels.parquet",
    )
    parity.add_argument("--max-unique-symbol-hours", type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = _parser()
    if args.command == "materialize":
        result = materialize(
            context_path=args.context,
            context_manifest_path=args.context_manifest,
            policy_path=args.policy_json,
            ohlcv_root=args.ohlcv_root,
            output_dir=args.output_dir,
            warmup_days=args.warmup_days,
        )
    else:
        result = historical_lagged_barrier_parity(
            labels_path=args.labels,
            max_unique_symbol_hours=args.max_unique_symbol_hours,
        )
    print(json.dumps(_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
