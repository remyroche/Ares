#!/usr/bin/env python3
"""Materialize causal ATR and frozen decision geometry for a retrospective cohort.

This is intentionally narrower than the historical Pack-B auxiliary-target
runner: it consumes a *scored final-refit* top-40 population and a separately
frozen, outcome-free geometry source.  It does not create labels, inspect any
future path, infer missing policy geometry, or fall back to a parent exit
without an explicit ``side_parent`` instruction in the geometry source.
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


SCHEMA = "execution_ev_retrospective_causal_geometry_v1"
GEOMETRY_SCHEMA = "execution_ev_frozen_decision_geometry_v1"
POPULATION_SCHEMA = "packb_final_refits_forward_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SIDES = ("long", "short")
FORBIDDEN_OUTCOME_COLUMNS = {
    "execution_label_end_utc",
    "execution_label_available_at",
    "execution_net_ev_12h",
    "execution_gross_ev_12h",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
}


class RetrospectiveGeometryError(RuntimeError):
    """Raised when causal geometry/ATR cannot be proven for every candidate."""


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
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise RetrospectiveGeometryError(f"{name} has null or invalid UTC timestamps")
    return parsed


def _canonical_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise RetrospectiveGeometryError(f"{source} missing identity columns: {missing}")
    output = frame.copy()
    output["__ts__"] = _utc(output["__ts__"], name=f"{source}.__ts__")
    for column in ("__symbol__", "side_name", "candidate_id"):
        output[column] = output[column].astype("string").str.strip()
        if output[column].isna().any() or output[column].eq("").any():
            raise RetrospectiveGeometryError(f"{source}.{column} has blank identities")
    output["side_name"] = output["side_name"].str.lower()
    if not output["side_name"].isin(SIDES).all():
        raise RetrospectiveGeometryError(f"{source}.side_name must be canonical long/short")
    if output.duplicated(list(IDENTITY), keep=False).any():
        raise RetrospectiveGeometryError(f"{source} has duplicate exact identities")
    if output["candidate_id"].duplicated().any():
        raise RetrospectiveGeometryError(f"{source} has duplicate candidate IDs")
    return output


def _load_bound_manifest(
    artifact: Path,
    manifest_path: Path,
    *,
    expected_schema: str,
    source: str,
    require_outcome_free: bool,
) -> dict[str, Any]:
    if not artifact.is_file() or not manifest_path.is_file():
        raise RetrospectiveGeometryError(f"{source} artifact and manifest must both exist")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != expected_schema:
        raise RetrospectiveGeometryError(
            f"{source} manifest schema must be {expected_schema!r}"
        )
    candidates = (
        payload.get("source_artifact_sha256"),
        payload.get("output", {}).get("sha256"),
        payload.get("outputs", {}).get("path", {}).get("sha256"),
    )
    if _sha256(artifact) not in {value for value in candidates if isinstance(value, str)}:
        raise RetrospectiveGeometryError(f"{source} manifest does not bind artifact hash")
    if require_outcome_free and payload.get("outcomes_used") is not False:
        raise RetrospectiveGeometryError(f"{source} manifest must explicitly declare outcomes_used=false")
    return payload


def _policy_archetypes(policy_path: Path) -> dict[str, set[str]]:
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    contract = payload.get("exit_geometry_contract", {})
    if contract.get("replay_timeframe") != "1m":
        raise RetrospectiveGeometryError("policy replay timeframe must be exact 1m")
    strategies = payload.get("strategies")
    if not isinstance(strategies, list):
        raise RetrospectiveGeometryError("policy has no strategy list")
    result = {side: set() for side in SIDES}
    for strategy in strategies:
        if not isinstance(strategy, Mapping) or not strategy.get("selected", True):
            continue
        if strategy.get("exit_geometry_scope") != "side_archetype":
            continue
        side = str(strategy.get("side", "")).lower()
        archetype = str(strategy.get("policy_archetype", "")).strip()
        if side in result and archetype:
            result[side].add(archetype)
    if not all(result.values()):
        raise RetrospectiveGeometryError("policy has no selected local archetype for one side")
    return result


def _normalise_geometry_archetype(value: object, *, side: str) -> tuple[str, str, str]:
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "nat", "<na>"}:
        raise RetrospectiveGeometryError("geometry source contains blank policy archetype")
    if raw in {"side_parent", "__side_parent__"}:
        return "side_parent", f"policy_archetype_{raw}", "explicit_side_parent"
    prefix = "policy_archetype_"
    raw = raw[len(prefix) :] if raw.startswith(prefix) else raw
    if not raw.startswith(f"{side}__"):
        raise RetrospectiveGeometryError(
            f"geometry policy archetype {raw!r} does not match {side} side"
        )
    return raw, f"{prefix}{raw}", "frozen_local_archetype"


def _store_parts(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    safe = symbol.replace("/", "_")
    directory = root / "ohlcv" / f"symbol={safe}"
    years = range(int(start.year), int(end.year) + 1)
    return [
        path
        for year in years
        for path in sorted((directory / f"year={year}").glob("*.parquet"))
    ]


def _signal_atr(
    population: pd.DataFrame,
    *,
    ohlcv_root: Path,
    warmup_days: int,
) -> tuple[np.ndarray, dict[str, str]]:
    store = PartitionedOHLCVStore(str(ohlcv_root), timeframe="1h")
    values = np.full(len(population), np.nan, dtype=np.float32)
    source_parts: dict[str, str] = {}
    for symbol, indices in population.groupby("__symbol__", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        signals = population.loc[positions, "__ts__"]
        start = pd.Timestamp(signals.min()) - pd.Timedelta(days=int(warmup_days))
        end = pd.Timestamp(signals.max())
        bars = store.load(str(symbol), columns=["high", "low", "close"], start_ts=start, end_ts=end)
        parts = _store_parts(ohlcv_root, str(symbol), start, end)
        if not parts or bars.empty:
            raise RetrospectiveGeometryError(f"{symbol}: missing canonical hourly OHLCV provenance")
        for part in parts:
            source_parts[str(part.relative_to(ohlcv_root))] = _sha256(part)
        bars = bars.loc[:, ["high", "low", "close"]].copy()
        bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
        bars = bars.loc[~bars.index.isna()].sort_index()
        if bars.index.duplicated().any():
            raise RetrospectiveGeometryError(f"{symbol}: duplicate hourly OHLC timestamps")
        numeric = bars.apply(pd.to_numeric, errors="coerce")
        raw = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all() or (raw <= 0.0).any() or (raw[:, 0] < raw[:, 1]).any():
            raise RetrospectiveGeometryError(f"{symbol}: nonfinite or invalid hourly OHLCV")
        if bars.index.min() > start:
            raise RetrospectiveGeometryError(
                f"{symbol}: required {warmup_days}d ATR warmup is unavailable"
            )
        atr = wilder_atr_fraction(
            numeric["high"].to_numpy(dtype=np.float64),
            numeric["low"].to_numpy(dtype=np.float64),
            numeric["close"].to_numpy(dtype=np.float64),
        )
        lookup = pd.Series(atr, index=bars.index)
        aligned = lookup.reindex(pd.DatetimeIndex(signals))
        if aligned.isna().any() or not np.isfinite(aligned.to_numpy(dtype=float)).all() or (aligned <= 0.0).any():
            raise RetrospectiveGeometryError(
                f"{symbol}: signal-time ATR requires an exact finite hourly timestamp"
            )
        values[positions] = aligned.to_numpy(dtype=np.float32)
    return values, source_parts


def materialize(
    *,
    population_path: Path,
    population_manifest_path: Path,
    geometry_path: Path,
    geometry_manifest_path: Path,
    policy_path: Path,
    ohlcv_root: Path,
    output_dir: Path,
    warmup_days: int = 90,
    decision_delay_hours: int = 1,
    horizon_hours: int = 12,
) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if warmup_days < 1 or decision_delay_hours != 1 or horizon_hours != 12:
        raise RetrospectiveGeometryError("contract requires 90d+ warmup, +1h decision, and 12h horizon")
    population_manifest = _load_bound_manifest(
        population_path,
        population_manifest_path,
        expected_schema=POPULATION_SCHEMA,
        source="scored Pack-B population",
        require_outcome_free=False,
    )
    geometry_manifest = _load_bound_manifest(
        geometry_path,
        geometry_manifest_path,
        expected_schema=GEOMETRY_SCHEMA,
        source="frozen decision geometry",
        require_outcome_free=True,
    )
    population = _canonical_identity(pd.read_parquet(population_path), source="population")
    forbidden = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(population.columns))
    if forbidden:
        raise RetrospectiveGeometryError(
            f"population must be pre-outcome; forbidden columns: {forbidden}"
        )
    required_population = {"execution_decision_utc", "feature_available_at"}
    missing = sorted(required_population.difference(population.columns))
    if missing:
        raise RetrospectiveGeometryError(f"population missing required fields: {missing}")
    population["execution_decision_utc"] = _utc(
        population["execution_decision_utc"], name="population.execution_decision_utc"
    )
    population["feature_available_at"] = _utc(
        population["feature_available_at"], name="population.feature_available_at"
    )
    expected_decision = population["__ts__"] + pd.Timedelta(hours=decision_delay_hours)
    if not population["execution_decision_utc"].eq(expected_decision).all():
        raise RetrospectiveGeometryError("population decision timestamps are not signal +1h")
    if (population["feature_available_at"] > population["execution_decision_utc"]).any():
        raise RetrospectiveGeometryError("population feature availability occurs after decision")
    if set(population["side_name"]) != set(SIDES):
        raise RetrospectiveGeometryError("population must contain both long and short sides")

    geometry = _canonical_identity(pd.read_parquet(geometry_path), source="geometry")
    forbidden = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(geometry.columns))
    if forbidden:
        raise RetrospectiveGeometryError(
            f"geometry must be decision-time only; forbidden columns: {forbidden}"
        )
    required_geometry = {"__barrier_pct__", "policy_archetype", "geometry_available_at"}
    missing = sorted(required_geometry.difference(geometry.columns))
    if missing:
        raise RetrospectiveGeometryError(f"geometry missing required fields: {missing}")
    geometry["geometry_available_at"] = _utc(
        geometry["geometry_available_at"], name="geometry.geometry_available_at"
    )
    coverage = population.loc[:, list(IDENTITY)].merge(
        geometry.loc[:, list(IDENTITY)], on=list(IDENTITY), how="outer", indicator=True
    )
    if not coverage["_merge"].eq("both").all():
        raise RetrospectiveGeometryError(
            "population and frozen geometry must have exact one-to-one identity coverage"
        )
    geometry = population.loc[:, list(IDENTITY) + ["execution_decision_utc"]].merge(
        geometry, on=list(IDENTITY), how="inner", validate="one_to_one"
    )
    if (geometry["geometry_available_at"] > geometry["execution_decision_utc"]).any():
        raise RetrospectiveGeometryError("geometry availability occurs after decision")
    barrier = pd.to_numeric(geometry["__barrier_pct__"], errors="coerce")
    if not np.isfinite(barrier.to_numpy(dtype=float)).all() or (barrier <= 0.0).any() or (barrier >= 1.0).any():
        raise RetrospectiveGeometryError("geometry barriers must be finite in (0, 1)")
    allowed = _policy_archetypes(policy_path)
    context_rows: list[dict[str, Any]] = []
    for row in geometry.to_dict(orient="records"):
        raw, resolved, source = _normalise_geometry_archetype(
            row["policy_archetype"], side=str(row["side_name"])
        )
        if source == "frozen_local_archetype" and resolved not in allowed[str(row["side_name"])]:
            raise RetrospectiveGeometryError(
                f"geometry archetype {resolved!r} is not selected by the frozen policy"
            )
        context_rows.append(
            {
                "__ts__": row["__ts__"],
                "__symbol__": row["__symbol__"],
                "side_name": row["side_name"],
                "candidate_id": row["candidate_id"],
                "policy_archetype": raw,
                "resolved_policy_archetype": resolved,
                "execution_geometry_source": source,
                "geometry_available_at": row["geometry_available_at"],
            }
        )
    atr, source_parts = _signal_atr(
        population, ohlcv_root=ohlcv_root, warmup_days=warmup_days
    )
    if not np.isfinite(atr).all() or (atr <= 0.0).any():
        raise RetrospectiveGeometryError("signal-time ATR contains nonfinite values")
    policy_context = pd.DataFrame(context_rows).sort_values(list(IDENTITY), kind="stable")
    path_targets = population.loc[:, list(IDENTITY)].copy()
    path_targets["__barrier_pct__"] = barrier.to_numpy(dtype=np.float32)
    path_targets["__path_auxiliary_atr_fraction__"] = atr
    path_targets["__path_auxiliary_atr_available_at__"] = population["__ts__"].to_numpy()
    path_targets = path_targets.sort_values(list(IDENTITY), kind="stable")
    if len(policy_context) != len(population) or len(path_targets) != len(population):
        raise RetrospectiveGeometryError("projection did not preserve every population row")

    output_dir.mkdir(parents=True, exist_ok=False)
    context_path = output_dir / "policy_context.parquet"
    targets_path = output_dir / "path_targets.parquet"
    policy_context.to_parquet(context_path, index=False, compression="zstd")
    path_targets.to_parquet(targets_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "materialized_retrospective_non_promotable_12h_timeout_ablation",
        "outcomes_used": False,
        "promotion_status": "non_promotable_retrospective_only",
        "timing": {
            "signal_timestamp": "__ts__",
            "decision": "__ts__ + 1h",
            "atr_available_at": "__ts__",
            "label_horizon_hours": 12,
            "policy_horizon_interpretation": "12h timeout-only ablation; frozen policy geometry is unchanged",
        },
        "atr_contract": {
            "formula": "raw-price Wilder ATR(14) / signal-bar close",
            "join": "exact __symbol__ + __ts__; no as-of, forward-fill, backfill, or neutral fill",
            "warmup_days": warmup_days,
            "ohlcv_root": str(ohlcv_root),
            "source_parts": source_parts,
        },
        "geometry_contract": {
            "source": "separate frozen decision-time geometry artifact",
            "outcomes_used": False,
            "availability": "geometry_available_at <= execution decision",
            "parent_fallback": "permitted only when geometry source explicitly specifies side_parent",
        },
        "inputs": {
            "population": {"path": str(population_path), "sha256": _sha256(population_path), "manifest_sha256": _sha256(population_manifest_path)},
            "geometry": {"path": str(geometry_path), "sha256": _sha256(geometry_path), "manifest_sha256": _sha256(geometry_manifest_path)},
            "policy": {"path": str(policy_path), "sha256": _sha256(policy_path)},
        },
        "outputs": {
            "policy_context": {"path": str(context_path), "sha256": _sha256(context_path), "rows": int(len(policy_context))},
            "path_targets": {"path": str(targets_path), "sha256": _sha256(targets_path), "rows": int(len(path_targets))},
        },
        "rows": {"population": int(len(population)), "long": int(population["side_name"].eq("long").sum()), "short": int(population["side_name"].eq("short").sum())},
        "source_manifests": {
            "population_schema": population_manifest["schema"],
            "geometry_schema": geometry_manifest["schema"],
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {"policy_context": context_path, "path_targets": targets_path, "manifest": output_dir / "manifest.json"}


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--ohlcv-root", type=Path, default=Path("data_perp/exchanges/krakenfutures"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-days", type=int, default=90)
    parser.add_argument("--decision-delay-hours", type=int, default=1)
    parser.add_argument("--horizon-hours", type=int, default=12)
    return parser.parse_args(argv)


def main() -> None:
    args = _parser()
    result = materialize(
        population_path=args.population,
        population_manifest_path=args.population_manifest,
        geometry_path=args.geometry,
        geometry_manifest_path=args.geometry_manifest,
        policy_path=args.policy_json,
        ohlcv_root=args.ohlcv_root,
        output_dir=args.output_dir,
        warmup_days=args.warmup_days,
        decision_delay_hours=args.decision_delay_hours,
        horizon_hours=args.horizon_hours,
    )
    print(json.dumps(_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
