#!/usr/bin/env python3
"""Stage missing 1m windows and materialize deployed-policy execution-EV labels.

The stage command is read-only with respect to the canonical candle store.  It
emits only downloader-compatible windows that are not already complete.  The
materialize command uses the immutable store and the canonical
``simple_policy_optimiser.simulate_and_score`` implementation.  Portfolio
concurrency is intentionally disabled: these are candidate-local execution
labels for a downstream admission model, not a portfolio replay.
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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (  # noqa: E402
    canonical_kraken_execution_1m_root,
    read_kraken_execution_1m,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _attach_policy_archetype_column,
    _policy_spread_baseline_audit,
    simulate_and_score,
)

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
PATH_COLUMNS = ("open", "high", "low", "close")
LABEL_SCHEMA = "execution_ev_deployed_policy_1m_labels_v1"
PREDICTION_ROLE = "execution_ev_12h_labels"
CANONICAL_SIMULATOR = (
    "extreme_price_movements.simple_policy_optimiser.simulate_and_score"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    canonical = {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if key != "prediction_role_manifest_sha256"
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _historical_source_lineage(
    manifest_path: Path | None,
    *,
    candidates_path: Path,
    context_path: Path,
    path_targets_path: Path,
    policy_path: Path,
) -> dict[str, Any] | None:
    """Verify and carry an optional research-only historical input contract."""

    if manifest_path is None:
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "historical_backcast_exact1m_label_inputs_v1":
        raise ValueError("historical source-lineage manifest schema is invalid")
    expected = {
        "candidates": candidates_path,
        "context": context_path,
        "path_targets": path_targets_path,
    }
    for key, path in expected.items():
        if (payload.get("outputs", {}).get(key, {}).get("sha256")) != _sha256(path):
            raise ValueError(
                f"historical source-lineage manifest does not bind {key}"
            )
    if (payload.get("policy_json", {}).get("sha256")) != _sha256(policy_path):
        raise ValueError("historical source-lineage manifest does not bind policy")
    if (
        payload.get("oof_status") != "not_oof"
        or bool(payload.get("execution_parity_claim"))
        or bool(payload.get("promotion_eligible"))
    ):
        raise ValueError("historical source-lineage flags are unsafe")
    return {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
        },
        "evidence_scope": payload.get("evidence_scope"),
        "lineage": payload.get("lineage"),
        "oof_status": "not_oof",
        "execution_parity_claim": False,
        "promotion_eligible": False,
        "economics": payload.get("economics"),
        "historical_l2_spread_available": bool(
            payload.get("historical_l2_spread_available")
        ),
        "atr_contract": payload.get("atr_contract"),
        "decision_to_path": payload.get("decision_to_path"),
    }


def _utc(series: pd.Series, *, column: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{column} contains null or invalid UTC timestamps")
    return result


def _load_candidates(
    candidates_path: Path,
    context_path: Path,
    path_targets_path: Path,
    *,
    decision_delay_minutes: int,
    allow_subset: bool = False,
) -> pd.DataFrame:
    candidates = pd.read_parquet(candidates_path, columns=list(IDENTITY))
    context_schema = set(pq.read_schema(context_path).names)
    context_column = next(
        (
            column
            for column in (
                "policy_archetype",
                "archetype_policy_key",
                "local_side_archetype",
            )
            if column in context_schema
        ),
        None,
    )
    if context_column is None:
        raise ValueError("context has no observable policy-archetype column")
    context = pd.read_parquet(context_path, columns=[*IDENTITY, context_column]).rename(
        columns={context_column: "__raw_policy_archetype__"}
    )
    targets = pd.read_parquet(
        path_targets_path,
        columns=[*IDENTITY, "__barrier_pct__", "__path_auxiliary_atr_fraction__"],
    )
    for name, frame in (
        ("candidates", candidates),
        ("context", context),
        ("path targets", targets),
    ):
        if frame.duplicated(list(IDENTITY), keep=False).any():
            raise ValueError(f"{name} has duplicate exact candidate identities")
        frame["__ts__"] = _utc(frame["__ts__"], column=f"{name}.__ts__")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if not frame["side_name"].isin(("long", "short")).all():
            raise ValueError(f"{name}.side_name must be canonical long/short")
    merged = candidates.merge(
        context, on=list(IDENTITY), how="left", validate="one_to_one"
    ).merge(targets, on=list(IDENTITY), how="left", validate="one_to_one")
    required = (
        "__raw_policy_archetype__",
        "__barrier_pct__",
        "__path_auxiliary_atr_fraction__",
    )
    incomplete = merged[list(required)].isna().any(axis=1)
    if incomplete.any() and allow_subset:
        merged = merged.loc[~incomplete].copy()
    elif incomplete.any():
        missing = {
            column: int(merged[column].isna().sum())
            for column in required
            if merged[column].isna().any()
        }
        raise ValueError(
            f"candidate/context/path-target exact join is incomplete: {missing}"
        )
    for column in ("__barrier_pct__", "__path_auxiliary_atr_fraction__"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        values = merged[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError(f"{column} must be finite and strictly positive")
    merged["__decision_ts__"] = merged["__ts__"] + pd.Timedelta(
        minutes=int(decision_delay_minutes)
    )
    return merged.reset_index(drop=True)


def _load_symbol_bars(
    data_root: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    bars = read_kraken_execution_1m(data_root, symbol, start=start, end=end)
    if bars.empty:
        return pd.DataFrame(columns=list(PATH_COLUMNS))
    if "ts" in bars.columns:
        bars = bars.set_index("ts")
    bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars = bars.loc[~bars.index.isna()]
    bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
    bars = bars.loc[(bars.index >= start) & (bars.index < end), list(PATH_COLUMNS)]
    for column in PATH_COLUMNS:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars


def _window_completeness(
    bars: pd.DataFrame,
    decisions: pd.Series,
    horizon_minutes: int,
) -> np.ndarray:
    if decisions.empty or bars.empty:
        return np.zeros(len(decisions), dtype=bool)
    start = min(pd.Timestamp(decisions.min()), pd.Timestamp(bars.index.min())).floor(
        "min"
    )
    end = max(
        pd.Timestamp(decisions.max()) + pd.Timedelta(minutes=horizon_minutes),
        pd.Timestamp(bars.index.max()) + pd.Timedelta(minutes=1),
    ).ceil("min")
    grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
    valid = (
        bars.reindex(grid)
        .loc[:, list(PATH_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
    )
    values = valid.to_numpy(dtype=np.float64)
    good = (
        np.isfinite(values).all(axis=1)
        & (values > 0.0).all(axis=1)
        & (values[:, 1] >= values[:, 2])
    )
    prefix = np.concatenate(([0], np.cumsum(good, dtype=np.int64)))
    offsets = (
        ((_utc(decisions, column="decision") - start) / pd.Timedelta(minutes=1))
        .astype(np.int64)
        .to_numpy()
    )
    ends = offsets + int(horizon_minutes)
    in_range = (offsets >= 0) & (ends <= len(good))
    output = np.zeros(len(decisions), dtype=bool)
    output[in_range] = prefix[ends[in_range]] - prefix[offsets[in_range]] == int(
        horizon_minutes
    )
    return output


def _coverage_rows(
    candidates: pd.DataFrame,
    *,
    data_root: Path,
    horizon_minutes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete = np.zeros(len(candidates), dtype=bool)
    for symbol, indices in candidates.groupby("__symbol__", sort=True).groups.items():
        pos = np.asarray(list(indices), dtype=np.int64)
        decisions = candidates.loc[pos, "__decision_ts__"]
        start = decisions.min()
        end = decisions.max() + pd.Timedelta(minutes=horizon_minutes)
        bars = _load_symbol_bars(data_root, str(symbol), start, end)
        complete[pos] = _window_completeness(bars, decisions, horizon_minutes)
    audit = candidates.loc[:, list(IDENTITY) + ["__decision_ts__"]].copy()
    audit["complete"] = complete
    audit["month"] = audit["__ts__"].dt.strftime("%Y-%m")

    def summary(group: pd.DataFrame) -> dict[str, Any]:
        rows = int(len(group))
        covered = int(group["complete"].sum())
        return {
            "rows": rows,
            "complete": covered,
            "missing": rows - covered,
            "coverage": covered / max(rows, 1),
        }

    payload = {
        "overall": summary(audit),
        "by_side": {
            str(key): summary(group)
            for key, group in audit.groupby("side_name", sort=True)
        },
        "by_month": {
            str(key): summary(group) for key, group in audit.groupby("month", sort=True)
        },
    }
    return audit, payload


def _policy_contract(
    policy_path: Path,
    *,
    horizon_minutes_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    contract = policy.get("exit_geometry_contract")
    strategies = policy.get("strategies")
    if not isinstance(contract, Mapping) or not isinstance(strategies, list):
        raise ValueError("policy JSON lacks exit_geometry_contract or strategies")
    obsolete_minute_decay = sorted(
        {
            str(key)
            for strategy in strategies
            if isinstance(strategy, Mapping)
            for key, value in strategy.items()
            if "decay" in str(key).lower()
            and "minute" in str(key).lower()
            and value is not None
        }
    )
    if obsolete_minute_decay:
        raise ValueError(
            "policy contains obsolete non-null minute-decay fields that are "
            f"not converted to exact 1m bars: {obsolete_minute_decay}"
        )
    if str(contract.get("replay_timeframe")) != "1m":
        raise ValueError("policy replay timeframe must be exact 1m")
    source_horizon = int(contract.get("horizon_minutes", 0))
    if source_horizon <= 0:
        raise ValueError("policy horizon_minutes must be positive")
    horizon = (
        int(horizon_minutes_override)
        if horizon_minutes_override is not None
        else source_horizon
    )
    if horizon <= 0 or horizon > source_horizon:
        raise ValueError(
            "horizon override must be positive and cannot exceed the signed policy horizon"
        )
    expected_pathway = str(contract.get("policy_pathway_id", ""))
    if not expected_pathway:
        raise ValueError("policy pathway ID is missing")
    exit_contract = {
        "replay_timeframe": "1m",
        "horizon_minutes": horizon,
        "geometry_scope": "side_x_policy_archetype_with_side_parent_fallback",
        "policy_pathway_id": expected_pathway,
        "trailing_activation_curve": str(contract.get("trailing_activation_curve", "")),
        "simulator": CANONICAL_SIMULATOR,
        "source_policy_sha256": _sha256(policy_path),
    }
    if horizon != source_horizon:
        exit_contract.update(
            {
                "source_policy_horizon_minutes": source_horizon,
                "horizon_override": "timeout_only_ablation; exit geometry unchanged",
            }
        )
    return policy, exit_contract


def _strategy_maps(
    policy: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    parents: dict[str, dict[str, Any]] = {}
    locals_: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in policy["strategies"]:
        if not isinstance(raw, Mapping) or not raw.get("selected", True):
            continue
        strategy = dict(raw)
        side = str(strategy.get("side", "")).lower()
        scope = str(strategy.get("exit_geometry_scope", ""))
        if side not in {"long", "short"}:
            continue
        if scope == "side_parent":
            current = parents.get(side)
            canonical = str(strategy.get("canonical_strategy_id", ""))
            if current is None or canonical == f"{side}__parent":
                parents[side] = strategy
        elif scope == "side_archetype":
            archetype = str(strategy.get("policy_archetype", ""))
            if archetype:
                locals_[(side, archetype)] = strategy
    if set(parents) != {"long", "short"}:
        raise ValueError("policy must provide canonical long and short parent geometry")
    return parents, locals_


def _resolved_geometry(
    candidates: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parents, locals_ = _strategy_maps(policy)
    work = candidates.copy()
    normalized = np.empty(len(work), dtype=object)
    for side, indices in work.groupby("side_name", sort=True).groups.items():
        pos = np.asarray(list(indices), dtype=np.int64)
        local = pd.DataFrame(
            {
                "policy_archetype": work.loc[pos, "__raw_policy_archetype__"],
                "side": np.where(side == "long", 1.0, -1.0),
            }
        )
        normalized[pos] = _attach_policy_archetype_column(
            local, strategy_id=f"{side}_s52_meta_threshold_handoff"
        )["policy_archetype"].to_numpy()
    work["policy_archetype"] = normalized.astype(str)
    geometry_key: list[str] = []
    geometry_source: list[str] = []
    for side, archetype in zip(
        work["side_name"].astype(str), work["policy_archetype"].astype(str)
    ):
        local = locals_.get((side, archetype))
        if local is not None:
            geometry_key.append(str(local.get("canonical_strategy_id")))
            geometry_source.append("side_archetype")
        else:
            geometry_key.append(str(parents[side].get("canonical_strategy_id")))
            geometry_source.append("side_parent_fallback")
    work["execution_geometry_key"] = geometry_key
    work["execution_geometry_source"] = geometry_source
    counts = work["execution_geometry_source"].value_counts().to_dict()
    return work, {
        "rows": int(len(work)),
        "side_archetype_rows": int(counts.get("side_archetype", 0)),
        "side_parent_fallback_rows": int(counts.get("side_parent_fallback", 0)),
        "side_archetype_rate": float(
            counts.get("side_archetype", 0) / max(len(work), 1)
        ),
        "fallback_rate": float(
            counts.get("side_parent_fallback", 0) / max(len(work), 1)
        ),
        "observable_archetypes": sorted(work["policy_archetype"].unique().tolist()),
    }


def _simulation_kwargs(
    strategy: Mapping[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    params = dict(strategy)
    cost_pct = float(params.pop("cost_pct_per_side"))
    size_power = float(params.pop("size_power", 1.0))
    for key in (
        "max_concurrent_trades",
        "max_concurrent_per_asset",
        "max_new_entries_per_bar",
    ):
        params.pop(key, None)
    return cost_pct, size_power, params


def _simulate_batch(
    rows: pd.DataFrame,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    strategy: Mapping[str, Any],
) -> pd.DataFrame:
    cost_pct, size_power, params = _simulation_kwargs(strategy)
    simulator_rows = pd.DataFrame(
        {
            "timestamp": rows["__decision_ts__"].to_numpy(),
            "symbol": rows["__symbol__"].astype(str).to_numpy(),
            "side_name": rows["side_name"].astype(str).to_numpy(),
            "side": np.where(rows["side_name"].eq("long"), 1.0, -1.0),
            "rank_pct": np.ones(len(rows), dtype=np.float32),
            "barrier_pct": rows["__barrier_pct__"].to_numpy(dtype=np.float32),
            "policy_archetype": rows["policy_archetype"].astype(str).to_numpy(),
        }
    )
    metrics = simulate_and_score(
        simulator_rows,
        *arrays,
        cost_pct=cost_pct,
        size_power=size_power,
        max_concurrent_trades=1_000_000_000,
        max_concurrent_per_asset=1_000_000_000,
        max_new_entries_per_bar=1_000_000_000,
        **params,
    )
    selected = np.asarray(metrics["selected_mask"], dtype=bool)
    if selected.shape != (len(rows),) or not selected.all():
        raise ValueError(
            "candidate-local policy replay unexpectedly dropped valid rows"
        )
    gross = np.asarray(metrics["gross_returns"], dtype=np.float64)
    net = np.asarray(metrics["net_returns"], dtype=np.float64)
    entry = np.asarray(metrics["entry_prices"], dtype=np.float64)
    side = np.where(rows["side_name"].eq("long"), 1.0, -1.0)
    highs = arrays[1].astype(np.float64, copy=False)
    lows = arrays[2].astype(np.float64, copy=False)
    mfe = np.where(
        side > 0,
        np.max(highs / entry[:, None] - 1.0, axis=1),
        np.max(1.0 - lows / entry[:, None], axis=1),
    )
    mae = np.where(
        side > 0,
        np.max(1.0 - lows / entry[:, None], axis=1),
        np.max(highs / entry[:, None] - 1.0, axis=1),
    )
    return pd.DataFrame(
        {
            "execution_gross_ev_12h": gross,
            "execution_cost_return": gross - net,
            "execution_net_ev_12h": net,
            "execution_exit_reason": list(metrics["exit_reason"]),
            "execution_exit_hour": np.asarray(metrics["exit_bars"], dtype=np.float64)
            / 60.0,
            "execution_mfe_return_12h": mfe,
            "execution_mae_return_12h": mae,
            "execution_entry_price": entry,
            "execution_exit_price": np.asarray(
                metrics["exit_prices"], dtype=np.float64
            ),
            "execution_expected_spread_bps": np.asarray(
                metrics["expected_spread_bps"], dtype=np.float64
            ),
            "execution_entry_half_spread_bps": np.asarray(
                metrics["entry_half_spread_bps"], dtype=np.float64
            ),
            "execution_exit_half_spread_bps": np.asarray(
                metrics["exit_spread_cost_bps"], dtype=np.float64
            ),
        }
    )


def stage(args: argparse.Namespace) -> dict[str, Path]:
    if args.output.exists() or args.manifest.exists() or args.coverage_csv.exists():
        raise ValueError("refusing to overwrite stage outputs")
    policy, exit_contract = _policy_contract(
        args.policy_json,
        horizon_minutes_override=getattr(args, "horizon_minutes_override", None),
    )
    candidate_input_rows = int(pq.ParquetFile(args.candidates).metadata.num_rows)
    candidates = _load_candidates(
        args.candidates,
        args.context,
        args.path_targets,
        decision_delay_minutes=args.decision_delay_minutes,
        allow_subset=bool(args.allow_subset),
    )
    candidates, geometry = _resolved_geometry(candidates, policy)
    audit, coverage = _coverage_rows(
        candidates,
        data_root=args.data_root,
        horizon_minutes=int(exit_contract["horizon_minutes"]),
    )
    missing = audit.loc[~audit["complete"]].copy()
    staged = pd.DataFrame(
        {
            "timestamp": missing["__decision_ts__"],
            "symbol": missing["__symbol__"],
            **{column: missing[column] for column in IDENTITY},
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged.to_parquet(args.output, index=False, compression="zstd")
    audit.to_csv(args.coverage_csv, index=False)
    payload = {
        "schema": "execution_ev_policy_1m_backfill_stage_v1",
        "source": {
            "candidates": str(args.candidates),
            "candidates_sha256": _sha256(args.candidates),
            "context": str(args.context),
            "context_sha256": _sha256(args.context),
            "path_targets": str(args.path_targets),
            "path_targets_sha256": _sha256(args.path_targets),
            "policy": str(args.policy_json),
            "policy_sha256": _sha256(args.policy_json),
            "exact_join": {
                "input_candidate_rows": candidate_input_rows,
                "admitted_rows": int(len(candidates)),
                "dropped_incomplete_context_or_path_rows": int(
                    candidate_input_rows - len(candidates)
                ),
                "subset_allowed": bool(args.allow_subset),
                "imputation": "none",
            },
        },
        "store": str(canonical_kraken_execution_1m_root(args.data_root)),
        "exit_policy_contract": exit_contract,
        "geometry": geometry,
        "coverage": coverage,
        "output": {"path": str(args.output), "sha256": _sha256(args.output)},
        "coverage_csv": {
            "path": str(args.coverage_csv),
            "sha256": _sha256(args.coverage_csv),
        },
    }
    _write_json(args.manifest, payload)
    return {
        "staging": args.output,
        "manifest": args.manifest,
        "coverage": args.coverage_csv,
    }


def materialize(args: argparse.Namespace) -> dict[str, Path]:
    if args.output.exists() or args.manifest.exists():
        raise ValueError("refusing to overwrite materialization outputs")
    spread_audit = _policy_spread_baseline_audit()
    spread_source = Path(str(spread_audit.get("source", "")))
    if (
        not spread_audit.get("loaded")
        or spread_source.resolve() != args.spread_baseline.resolve()
    ):
        raise ValueError(
            "deployed spread baseline is not loaded from --spread-baseline; "
            "set EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH before launch"
        )
    policy, exit_contract = _policy_contract(
        args.policy_json,
        horizon_minutes_override=getattr(args, "horizon_minutes_override", None),
    )
    historical_lineage = _historical_source_lineage(
        getattr(args, "source_lineage_manifest", None),
        candidates_path=args.candidates,
        context_path=args.context,
        path_targets_path=args.path_targets,
        policy_path=args.policy_json,
    )
    if historical_lineage is not None and int(exit_contract["horizon_minutes"]) != 720:
        raise ValueError(
            "historical exact-path lineage requires the signed 720-minute replay"
        )
    horizon = int(exit_contract["horizon_minutes"])
    candidate_input_rows = int(pq.ParquetFile(args.candidates).metadata.num_rows)
    candidates = _load_candidates(
        args.candidates,
        args.context,
        args.path_targets,
        decision_delay_minutes=args.decision_delay_minutes,
        allow_subset=bool(args.allow_subset),
    )
    candidates, geometry = _resolved_geometry(candidates, policy)
    audit, coverage = _coverage_rows(
        candidates, data_root=args.data_root, horizon_minutes=horizon
    )
    missing = audit.loc[~audit["complete"]].copy()
    args.missing_csv.parent.mkdir(parents=True, exist_ok=True)
    missing_tmp = args.missing_csv.with_name(
        f".{args.missing_csv.name}.{os.getpid()}.tmp"
    )
    missing.to_csv(missing_tmp, index=False)
    os.replace(missing_tmp, args.missing_csv)
    if not missing.empty and not args.allow_subset:
        raise ValueError(
            f"{len(missing)} candidates still lack exact 1m paths; "
            "run targeted backfill or pass --allow-subset"
        )
    complete = candidates.loc[audit["complete"].to_numpy()].copy()
    strategy_by_key = {
        str(strategy.get("canonical_strategy_id")): strategy
        for strategy in policy["strategies"]
        if isinstance(strategy, Mapping) and strategy.get("selected", True)
    }
    parts: list[pd.DataFrame] = []
    for symbol, indices in complete.groupby("__symbol__", sort=True).groups.items():
        symbol_rows = complete.loc[list(indices)].copy().reset_index(drop=True)
        start = symbol_rows["__decision_ts__"].min()
        end = symbol_rows["__decision_ts__"].max() + pd.Timedelta(minutes=horizon)
        bars = _load_symbol_bars(args.data_root, str(symbol), start, end)
        grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
        values = (
            bars.reindex(grid).loc[:, list(PATH_COLUMNS)].to_numpy(dtype=np.float32)
        )
        offsets = (
            ((symbol_rows["__decision_ts__"] - start) / pd.Timedelta(minutes=1))
            .astype(np.int64)
            .to_numpy()
        )
        for begin in range(0, len(symbol_rows), int(args.batch_rows)):
            stop = min(begin + int(args.batch_rows), len(symbol_rows))
            batch = symbol_rows.iloc[begin:stop].copy().reset_index(drop=True)
            batch_offsets = offsets[begin:stop]
            matrices = tuple(
                np.stack(
                    [
                        values[offset : offset + horizon, column]
                        for offset in batch_offsets
                    ]
                )
                for column in range(len(PATH_COLUMNS))
            )
            for geometry_key, local_indices in batch.groupby(
                "execution_geometry_key", sort=True
            ).groups.items():
                pos = np.asarray(list(local_indices), dtype=np.int64)
                strategy = strategy_by_key.get(str(geometry_key))
                if strategy is None:
                    raise ValueError(
                        f"resolved policy strategy is missing: {geometry_key}"
                    )
                simulated = _simulate_batch(
                    batch.iloc[pos].reset_index(drop=True),
                    tuple(matrix[pos] for matrix in matrices),
                    strategy,
                )
                source_rows = batch.iloc[pos].reset_index(drop=True)
                label_end = source_rows["__decision_ts__"] + pd.Timedelta(
                    minutes=horizon
                )
                labels = pd.concat(
                    [
                        source_rows.loc[
                            :,
                            [
                                *IDENTITY,
                                "__decision_ts__",
                                "policy_archetype",
                                "execution_geometry_key",
                                "execution_geometry_source",
                            ],
                        ].rename(columns={"__decision_ts__": "execution_decision_utc"}),
                        simulated,
                    ],
                    axis=1,
                )
                labels["execution_label_end_utc"] = label_end.to_numpy()
                labels["execution_label_available_at"] = label_end.to_numpy()
                parts.append(labels)
    if not parts:
        raise ValueError("no complete policy labels were materialized")
    labels = pd.concat(parts, ignore_index=True)
    labels = labels.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if labels.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("materialized labels contain duplicate candidate identities")
    if not np.allclose(
        labels["execution_gross_ev_12h"] - labels["execution_cost_return"],
        labels["execution_net_ev_12h"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("gross - cost does not equal net execution EV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    labels.to_parquet(output_tmp, index=False, compression="zstd")
    output_sha256 = _sha256(output_tmp)
    manifest: dict[str, Any] = {
        "schema": LABEL_SCHEMA,
        "prediction_role": PREDICTION_ROLE,
        "source_artifact_sha256": output_sha256,
        "output": {
            "path": str(args.output),
            "sha256": output_sha256,
            "rows": int(len(labels)),
        },
        "source": {
            "candidates": str(args.candidates),
            "candidates_sha256": _sha256(args.candidates),
            "context": str(args.context),
            "context_sha256": _sha256(args.context),
            "path_targets": str(args.path_targets),
            "path_targets_sha256": _sha256(args.path_targets),
            "policy": str(args.policy_json),
            "policy_sha256": _sha256(args.policy_json),
            "exact_join": {
                "input_candidate_rows": candidate_input_rows,
                "admitted_rows": int(len(candidates)),
                "dropped_incomplete_context_or_path_rows": int(
                    candidate_input_rows - len(candidates)
                ),
                "subset_allowed": bool(args.allow_subset),
                "imputation": "none",
            },
        },
        "historical_lineage": historical_lineage,
        "exit_policy_contract": exit_contract,
        "targets": {
            "execution_net_ev_12h": {
                "horizon_hours": horizon / 60.0,
                "exit_policy_contract": exit_contract,
            }
        },
        "timing": {
            "signal_to_decision_minutes": int(args.decision_delay_minutes),
            "cadence_minutes": 1,
            "horizon_minutes": horizon,
            "label_available_at": "decision + full replay horizon",
        },
        "accounting": {
            "simulator": CANONICAL_SIMULATOR,
            "candidate_local_exit_replay": True,
            "portfolio_concurrency_applied": False,
            "fee": "strategy cost_pct_per_side applied on entry and exit",
            "spread": "deployed side-aware executable entry and exit spread handling",
            "gross_return": "relative to executable entry and spread-aware exit fill",
            "cost_return": "fee return; spread drag is embedded in gross return",
            "net_return": "gross return minus fee return",
            "spread_baseline": str(args.spread_baseline),
            "spread_baseline_sha256": _sha256(args.spread_baseline),
            "spread_audit": spread_audit,
        },
        "geometry": geometry,
        "coverage": coverage,
        "missing": {
            "path": str(args.missing_csv),
            "sha256": _sha256(args.missing_csv),
            "rows": int(len(missing)),
            "subset": bool(len(missing)),
        },
        "store": {
            "root": str(canonical_kraken_execution_1m_root(args.data_root)),
            "contract": "canonical_kraken_execution_1m_immutable_read_only_v1",
        },
    }
    manifest["prediction_role_manifest_sha256"] = _canonical_hash(manifest)
    manifest_tmp = args.manifest.with_name(f".{args.manifest.name}.{os.getpid()}.tmp")
    _write_json(manifest_tmp, manifest)
    os.replace(output_tmp, args.output)
    os.replace(manifest_tmp, args.manifest)
    return {
        "labels": args.output,
        "manifest": args.manifest,
        "missing": args.missing_csv,
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--path-targets", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument(
        "--source-lineage-manifest",
        type=Path,
        default=None,
        help=(
            "Optional signed historical label-input manifest; when supplied, "
            "its research-only lineage and exact input hashes are verified and "
            "propagated to the policy-label output."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--decision-delay-minutes", type=int, default=60)
    parser.add_argument(
        "--horizon-minutes-override",
        type=int,
        help=(
            "Shorter timeout-only ablation horizon; the signed exit geometry "
            "is unchanged and the override is recorded in the output contract."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage")
    _common(stage_parser)
    stage_parser.add_argument("--output", type=Path, required=True)
    stage_parser.add_argument("--manifest", type=Path, required=True)
    stage_parser.add_argument("--coverage-csv", type=Path, required=True)
    stage_parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "permit an explicitly audited exact subset when context or path inputs "
            "do not cover every candidate; attrition is recorded in the manifest"
        ),
    )
    materialize_parser = commands.add_parser("materialize")
    _common(materialize_parser)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--manifest", type=Path, required=True)
    materialize_parser.add_argument("--missing-csv", type=Path, required=True)
    materialize_parser.add_argument("--spread-baseline", type=Path, required=True)
    materialize_parser.add_argument("--batch-rows", type=int, default=500)
    materialize_parser.add_argument("--allow-subset", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.decision_delay_minutes < 0:
        raise ValueError("decision-delay-minutes must be non-negative")
    if args.command == "stage":
        paths = stage(args)
    else:
        if args.batch_rows < 1:
            raise ValueError("batch-rows must be positive")
        paths = materialize(args)
    print(json.dumps(_json_safe(paths), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
