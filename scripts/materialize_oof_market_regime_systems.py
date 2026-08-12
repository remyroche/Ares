#!/usr/bin/env python3
"""Materialize train-only hourly causal market-regime systems and candidate joins.

The input is the existing segmented causal multiview hourly panel.  Regime
models are fitted once per chronological block on rows strictly before the
block, then filtered forward over that block.  The optional candidate output is
a backward as-of join only; no candidate row is used to fit the hourly systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from itertools import permutations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_market_regime_systems import (  # noqa: E402
    CONTINUOUS_CONTEXT_FEATURE_KEYS,
    CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
    CausalContinuousContextConfig,
    CausalRelationshipBreakConfig,
    CausalMarketRegimeConfig,
    DEFAULT_GEOMETRY_SPECS,
    PHASE_NAMES,
    build_causal_continuous_context_features,
    build_causal_relationship_break_features,
    fit_causal_market_regime_systems,
    relationship_break_feature_names,
)
from extreme_price_movements.regime_oof_stack import (  # noqa: E402
    IDENTITY_COLUMNS,
    PROVENANCE_COLUMNS,
    TRANSITION_PROVENANCE_COLUMNS,
    asof_join_regime_timeline,
    derive_soft_state_fields,
    validate_candidate_identity,
    validate_combined_regime_transition_outputs,
)


SCHEMA = "oof_causal_market_regime_systems_v1"
DEFAULT_PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/oof_causal_market_regime_systems_20260803_v1"
HORIZON_TOKENS = ("_6h", "_12h", "_3h", "_1h", "_24h")
# State discovery should describe the market's current geometry/stress level.
# Fast deltas/accelerations are intentionally lower priority: the forward
# posterior dynamics already turns movement between those levels into the
# onset/active/settling transition surface.  If a sparse historical store has
# no level field for a view we retain its available acceleration proxy rather
# than inventing a substitute; the manifest records the frozen inputs.
STATE_DISCOVERY_ACCELERATION_TOKENS = (
    "delta", "change", "accel", "acceleration", "derivative", "slope",
)
FORBIDDEN = (
    "target", "label", "outcome", "future", "post_entry", "postentry",
    "realized", "realised", "pnl", "net_ev", "gross_ev", "mfe", "mae",
    "barrier", "timeout", "exit", "time_to", "policy", "score", "rank",
    "candidate_count",
)
VIEW_TOKENS: dict[str, tuple[str, ...]] = {
    "trend_volatility": ("trend", "ema", "momentum", "return", "ret", "vol", "atr", "range", "compression", "chop", "efficiency", "coherence", "entropy"),
    "breadth_dependence": ("breadth", "dispersion", "correlation", "corr", "dependence", "eigen", "effective_rank", "synchron", "cross_section"),
    "leverage_flow": ("fund", "funding", "open_interest", "oi_", "basis", "premium", "carry", "liquidation", "deleverag", "crowding", "flow"),
    "liquidity": ("liquid", "spread", "depth", "amihud", "amivest", "volume", "turnover", "orderbook", "ob_", "impact"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _blocks(timestamp: pd.Series, *, start: pd.Timestamp, frequency: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    values = pd.to_datetime(timestamp, utc=True, errors="raise")
    if frequency == "month":
        begins = sorted(pd.Timestamp(item.start_time, tz="UTC") for item in values.dt.to_period("M").unique())
        return [(item, item + pd.offsets.MonthBegin(1)) for item in begins if item + pd.offsets.MonthBegin(1) > start]
    if frequency == "quarter":
        begins = sorted(pd.Timestamp(item.start_time, tz="UTC") for item in values.dt.to_period("Q").unique())
        return [(item, item + pd.offsets.QuarterBegin(startingMonth=1)) for item in begins if item + pd.offsets.QuarterBegin(startingMonth=1) > start]
    raise ValueError("frequency must be 'month' or 'quarter'")


def _horizon_rank(name: str) -> tuple[int, int, int, str]:
    lower = name.lower()
    horizon = next((index for index, token in enumerate(HORIZON_TOKENS) if lower.endswith(token)), len(HORIZON_TOKENS))
    level_priority = int(any(token in lower for token in STATE_DISCOVERY_ACCELERATION_TOKENS))
    transform = 0 if any(token in lower for token in ("robust_z", "stress", "eig1_share", "effective_rank", "corr_frobenius", "covariance_frobenius", "realized_vol", "vol_of_vol")) else 1
    return level_priority, horizon, transform, name


def compact_observable_columns(path: Path, *, max_per_view: int = 20) -> list[str]:
    """Select a small predeclared observable proxy from Parquet schema names."""

    schema = pq.ParquetFile(path).schema.names
    candidates = [
        str(name)
        for name in schema
        if str(name).startswith("mv__")
        and any(str(name).endswith(token) for token in HORIZON_TOKENS)
        and not any(token in str(name).lower() for token in FORBIDDEN)
    ]
    by_view: list[list[str]] = []
    for tokens in VIEW_TOKENS.values():
        local = [name for name in candidates if any(token in name.lower() for token in tokens)]
        by_view.append(sorted(local, key=_horizon_rank)[: int(max_per_view)])
    # Round-robin the geometries.  This makes the compact primary proxy
    # intentionally diverse even though the latent specialists later receive
    # their own geometry-specific subsets.
    selected = [
        fields[index]
        for index in range(max((len(fields) for fields in by_view), default=0))
        for fields in by_view
        if index < len(fields)
    ]
    return list(dict.fromkeys(selected))


def _timeline_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose requested names plus existing OOF-stack-compatible aliases."""

    out = frame.copy()
    posterior = [f"market_regime__state_p_{state}" for state in range(5)]
    if any(name not in out for name in posterior):
        raise ValueError("primary five-state system did not emit every posterior coordinate")
    for state, name in enumerate(posterior):
        out[f"regime_state_p__{state}"] = out[name].to_numpy(np.float32)
    out["regime_state_ood_score"] = 1.0 - out["market_regime__ood_distance_percentile"].to_numpy(np.float32)
    out = derive_soft_state_fields(out, copy=False)
    for phase in PHASE_NAMES:
        out[f"transition_state_p__{phase}"] = out[f"market_regime__phase_p_{phase}"].to_numpy(np.float32)
    out["transition_state_ood_score"] = out["regime_state_ood_score"].to_numpy(np.float32)
    out = derive_soft_state_fields(out, probability_prefix="transition_state_p__", copy=False)
    return out


def _align_primary_centroids(
    current: np.ndarray,
    previous: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Causally align equal-K primary states to the preceding frozen fold.

    The current model and its centroids are fitted before its own evaluation
    block, while the reference is an earlier frozen model.  Therefore this
    alignment is available at block start and never uses evaluation outcomes.
    """

    values = np.asarray(current, dtype=np.float32)
    if previous is None:
        return values, np.arange(len(values), dtype=np.int64), {
            "status": "anchor_first_fold", "passed": True,
            "mapping_current_to_previous": list(range(len(values))),
            "mean_matched_centroid_distance": 0.0,
        }
    prior = np.asarray(previous, dtype=np.float32)
    if values.ndim != 2 or prior.ndim != 2 or values.shape != prior.shape:
        return values, np.arange(len(values), dtype=np.int64), {
            "status": "incompatible_state_count_or_centroid_shape", "passed": False,
            "mapping_current_to_previous": None,
            "mean_matched_centroid_distance": None,
        }
    k = len(values)
    distances = ((values[:, None, :] - prior[None, :, :]) ** 2).sum(axis=2)
    best = min(
        permutations(range(k)),
        key=lambda mapping: (float(sum(distances[current_state, previous_state] for current_state, previous_state in enumerate(mapping))), mapping),
    )
    mapping = np.asarray(best, dtype=np.int64)
    aligned = np.empty_like(values)
    for current_state, previous_state in enumerate(mapping):
        aligned[previous_state] = values[current_state]
    matched = np.asarray([distances[current_state, previous_state] for current_state, previous_state in enumerate(mapping)], dtype=np.float32)
    return aligned, mapping, {
        "status": "matched_to_prior_frozen_fold",
        "passed": True,
        "mapping_current_to_previous": mapping.tolist(),
        "mean_matched_centroid_distance": float(np.sqrt(matched).mean()),
        "max_matched_centroid_distance": float(np.sqrt(matched).max()),
    }


def materialize(
    *,
    panel_path: Path = DEFAULT_PANEL,
    output_dir: Path = DEFAULT_OUTPUT,
    evaluation_start: str,
    evaluation_end: str | None = None,
    candidate_path: Path | None = None,
    frequency: str = "quarter",
    purge_hours: int = 12,
    max_features_per_view: int = 20,
    max_lag_hours: int = 2,
    seed: int = 20260803,
    primary_state_count: int = 5,
    primary_merge_low_support_state: bool = False,
    systems: Sequence[str] | None = None,
) -> Path:
    """Build a bounded chronological OOF hourly sidecar, optionally candidate-keyed."""

    panel_path, output_dir = Path(panel_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    start = pd.to_datetime(evaluation_start, utc=True, errors="raise")
    end = pd.to_datetime(evaluation_end, utc=True, errors="raise") if evaluation_end else None
    columns = compact_observable_columns(panel_path, max_per_view=max_features_per_view)
    schema = set(pq.ParquetFile(panel_path).schema.names)
    # Synthetic/minimal panels used by diagnostics may not expose the full
    # production multiview contract.  Materialize the available named subset
    # (never invent substitutions); production manifests make any omission
    # explicit for the meta feature selector.
    continuous_source_contract = {
        alias: source
        for alias, source in CONTINUOUS_CONTEXT_SOURCE_CONTRACT.items()
        if source in schema
    }
    missing_context_sources = [
        source for source in CONTINUOUS_CONTEXT_SOURCE_CONTRACT.values()
        if source not in schema
    ]
    required = list(dict.fromkeys([
        "source_utc", *columns, *continuous_source_contract.values(),
    ]))
    segment_col = "calendar_segment_id" if "calendar_segment_id" in schema else None
    if segment_col:
        required.insert(1, segment_col)
    hourly = pd.read_parquet(panel_path, columns=required)
    hourly["source_utc"] = pd.to_datetime(hourly["source_utc"], utc=True, errors="raise")
    hourly = hourly.sort_values("source_utc", kind="stable").drop_duplicates("source_utc", keep="last").reset_index(drop=True)
    if end is not None:
        hourly = hourly.loc[hourly["source_utc"].lt(end)].copy()
    if hourly.empty:
        raise ValueError("no hourly panel rows remain in requested evaluation range")
    wanted_systems = tuple(systems) if systems is not None else tuple(spec.name for spec in DEFAULT_GEOMETRY_SPECS)
    specs = tuple(spec for spec in DEFAULT_GEOMETRY_SPECS if spec.name in wanted_systems)
    if not specs or set(wanted_systems) != {spec.name for spec in specs}:
        raise ValueError(f"unknown or empty regime systems: {wanted_systems}")
    if "primary" not in wanted_systems:
        raise ValueError("the OOF sidecar requires the primary regime system")
    cfg = CausalMarketRegimeConfig(
        timestamp_col="source_utc",
        group_columns=(segment_col,) if segment_col else (),
        max_gap_hours=2.0,
        transition_horizon_hours=6.0,
        random_state=int(seed),
        primary_state_count=int(primary_state_count),
        primary_merge_low_support_state=bool(primary_merge_low_support_state),
    )
    continuous_cfg = CausalContinuousContextConfig(
        timestamp_col="source_utc",
        group_columns=(segment_col,) if segment_col else (),
    )
    relationship_break_cfg = CausalRelationshipBreakConfig(
        timestamp_col="source_utc",
        group_columns=(segment_col,) if segment_col else (),
    )
    purge = pd.Timedelta(hours=int(purge_hours))
    outputs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    prior_primary_centroids: np.ndarray | None = None
    for number, (block_start, block_end) in enumerate(_blocks(hourly["source_utc"], start=start, frequency=frequency), start=1):
        evaluation = hourly["source_utc"].ge(max(start, block_start)) & hourly["source_utc"].lt(block_end)
        train = hourly["source_utc"].lt(block_start - purge)
        if not evaluation.any():
            continue
        if int(train.sum()) < 160:
            continue
        model = fit_causal_market_regime_systems(hourly.loc[train], columns, specs=specs, config=cfg)
        local = hourly.loc[evaluation, ["source_utc"]].copy()
        local = local.join(model.transform(hourly.loc[evaluation], carry_history=False))
        # Relative continuous context is computed from a bounded history window
        # ending at this block.  Every rolling reference is left-closed, so a
        # row sees only prior raw observables; no GMM posterior or membership
        # coordinate enters this path.
        continuous_start = max(hourly["source_utc"].min(), block_start - pd.Timedelta(days=180, hours=24))
        continuous_rows = hourly["source_utc"].ge(continuous_start) & hourly["source_utc"].lt(block_end)
        if continuous_source_contract:
            continuous = build_causal_continuous_context_features(
                hourly.loc[continuous_rows],
                continuous_source_contract,
                config=continuous_cfg,
            )
            local = local.join(continuous.loc[local.index])
            relationship_breaks = build_causal_relationship_break_features(
                hourly.loc[continuous_rows],
                continuous_source_contract,
                config=relationship_break_cfg,
            )
            local = local.join(relationship_breaks.loc[local.index])
        primary_prefix = "market_regime__state_p_"
        primary_state_columns = [name for name in local if name.startswith(primary_prefix)]
        current_centroids = np.asarray(
            model.diagnostics["systems"]["primary"]["effective_state_centroids"],
            dtype=np.float32,
        )
        aligned_centroids, mapping, alignment = _align_primary_centroids(
            current_centroids,
            prior_primary_centroids,
        )
        if bool(alignment["passed"]):
            primary_values = local.loc[:, primary_state_columns].to_numpy(dtype=np.float32, copy=True)
            aligned_values = np.empty_like(primary_values)
            for current_state, previous_state in enumerate(mapping):
                aligned_values[:, previous_state] = primary_values[:, current_state]
            local.loc[:, primary_state_columns] = aligned_values
            distance_columns = [
                f"market_regime__state_centroid_distance_p_{state}"
                for state in range(len(primary_state_columns))
            ]
            if set(distance_columns).issubset(local.columns):
                raw_distances = local.loc[:, distance_columns].to_numpy(dtype=np.float32, copy=True)
                aligned_distances = np.empty_like(raw_distances)
                for current_state, previous_state in enumerate(mapping):
                    aligned_distances[:, previous_state] = raw_distances[:, current_state]
                local.loc[:, distance_columns] = aligned_distances
            prior_primary_centroids = aligned_centroids
        else:
            # A different effective K after a rare-state merge cannot inherit
            # coordinate semantics.  Keep it diagnostic-only and force its
            # intrinsic alignment gate to fail.
            prior_primary_centroids = current_centroids
        local["market_regime__state_count"] = np.float32(len(primary_state_columns))
        # Candidate and downstream diagnostic contracts have five named
        # coordinates.  K=3/4 and K5-merge leave unused coordinates at zero;
        # invariant context fields are the only model inputs.
        for state in range(5):
            column = f"{primary_prefix}{state}"
            if column not in local:
                local[column] = np.float32(0.0)
            distance_column = f"market_regime__state_centroid_distance_p_{state}"
            if distance_column not in local:
                local[distance_column] = np.float32(0.0)
        # Geometry K is selected independently per fold.  Preserve a fixed
        # diagnostic schema by zero-padding unavailable posterior coordinates;
        # their accompanying state_count records the local support.  These
        # fold-local coordinates are intentionally excluded from model feature
        # keys, so padding cannot create a false cross-fold semantic feature.
        for system in ("trend_volatility", "breadth_dependence", "leverage_flow", "liquidity"):
            if system not in model.models:
                continue
            prefix = f"geometry_regime__{system}__state_p_"
            state_columns = [name for name in local if name.startswith(prefix)]
            local[f"geometry_regime__{system}__state_count"] = np.float32(len(state_columns))
            for state in range(6):
                column = f"{prefix}{state}"
                if column not in local:
                    local[column] = np.float32(0.0)
        local["regime_fold_id"] = f"{frequency}_{number:03d}_{block_start.strftime('%Y%m%d')}"
        local["regime_train_end_utc"] = hourly.loc[train, "source_utc"].max()
        local["regime_available_utc"] = local["source_utc"]
        local["transition_fold_id"] = local["regime_fold_id"].astype(str) + "__phase"
        local["transition_train_end_utc"] = local["regime_train_end_utc"]
        local["transition_available_utc"] = local["source_utc"]
        local = _timeline_aliases(local)
        outputs.append(local)
        for system, item in model.diagnostics["systems"].items():
            diagnostics.append({"fold_id": local["regime_fold_id"].iloc[0], "evaluation_start_utc": block_start, "evaluation_end_exclusive_utc": block_end, "system": system, **item, **({"primary_fold_alignment": alignment} if system == "primary" else {})})
            for field in item["feature_columns"]:
                coverage.append({"fold_id": local["regime_fold_id"].iloc[0], "system": system, "feature": field, "train_coverage": float(hourly.loc[train, field].notna().mean()), "train_nonconstant": bool(hourly.loc[train, field].nunique(dropna=True) > 1)})
        for alias, field in continuous_source_contract.items():
            coverage.append({"fold_id": local["regime_fold_id"].iloc[0], "system": "continuous_context", "feature": field, "feature_alias": alias, "train_coverage": float(hourly.loc[train, field].notna().mean()), "train_nonconstant": bool(hourly.loc[train, field].nunique(dropna=True) > 1)})
    if not outputs:
        raise ValueError("no OOF blocks had sufficient prior hourly support")
    timeline = pd.concat(outputs, ignore_index=True).sort_values("source_utc", kind="stable").reset_index(drop=True)
    if not (timeline["regime_train_end_utc"] < timeline["regime_available_utc"]).all():
        raise RuntimeError("regime OOF provenance is not strictly prior to availability")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        timeline.to_parquet(temporary / "hourly_oof_market_regimes.parquet", index=False, compression="zstd")
        pd.DataFrame(diagnostics).to_json(temporary / "parameter_diagnostics.json", orient="records", indent=2, date_format="iso")
        pd.DataFrame(coverage).to_parquet(temporary / "input_coverage.parquet", index=False, compression="zstd")
        candidate_rows = 0
        if candidate_path is not None:
            # Do not load an entire candidate/label panel merely to perform a
            # target-free backward-asof state join. Reading only immutable
            # identity columns also prevents resolved outcomes from entering
            # this unsupervised state materializer by accident.
            candidates = validate_candidate_identity(
                pd.read_parquet(candidate_path, columns=list(IDENTITY_COLUMNS))
            ).loc[:, list(IDENTITY_COLUMNS)].copy()
            candidates = candidates.loc[candidates["__ts__"].ge(start)].copy()
            if end is not None:
                candidates = candidates.loc[candidates["__ts__"].lt(end)].copy()
            candidate = asof_join_regime_timeline(candidates, timeline, by=(), timeline_timestamp_col="source_utc", max_lag=pd.Timedelta(hours=int(max_lag_hours)), provenance_columns=PROVENANCE_COLUMNS)
            # The second provenance layer is copied from the same frozen
            # hourly system but remains explicit because the phase simplex is
            # a distinct action-facing output contract.
            validate_combined_regime_transition_outputs(candidate)
            candidate.to_parquet(temporary / "candidate_oof_market_regimes.parquet", index=False, compression="zstd")
            candidate_rows = int(len(candidate))
        manifest = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_CAUSAL_FROZEN_OOF",
            "inputs": {"hourly_multiview_panel": {"path": str(panel_path.resolve()), "sha256": _sha256(panel_path)}, "candidate_path": str(Path(candidate_path).resolve()) if candidate_path else None},
            "contract": {"hourly_compute_once": True, "fit": "all imputation/scaling/GMM/K/stickiness selection uses rows strictly before each evaluation block", "primary": {"requested_state_count": int(primary_state_count), "postfit_low_support_merge": bool(primary_merge_low_support_state), "state_coordinate_padding": "five named diagnostic coordinates; raw state coordinates are not model features"}, "systems": list(wanted_systems), "transition": "forward-only stable/onset/active/settling simplex; no delayed phase label used as input", "continuous_context": {"source_contract": dict(continuous_source_contract), "missing_production_sources": missing_context_sources, "output_features": [name for name in CONTINUOUS_CONTEXT_FEATURE_KEYS if any(f"continuous_regime__{alias}__" in name for alias in continuous_source_contract)], "semantics": "strict-prequential 90/180d relative rank/z, exact 4/24h changes and prior-30d median distance; raw continuous sources only, no cluster membership inputs"}, "horizons": "1/3/6/12h primary plus 24h slow context", "selection": "no economic labels/outcomes used", "candidate_join": "backward as-of only" if candidate_path else "not requested"},
            "feature_proxy": {"fields": columns, "max_features_per_view": int(max_features_per_view), "source_selection": "schema-only predeclared token/horizon proxy; level/stress fields precede acceleration fields for the primary 5-state discovery view", "primary_state_discovery": {"target_feature_count": "15-25 stable market-level fields where coverage permits", "acceleration_tokens_deprioritized": list(STATE_DISCOVERY_ACCELERATION_TOKENS), "direction": "signed return/momentum fields excluded from primary geometry; emitted separately by frozen primary model"}},
            "coverage": {"hourly_rows": int(len(timeline)), "candidate_rows": candidate_rows, "evaluation_start_utc": start.isoformat(), "evaluation_end_exclusive_utc": end.isoformat() if end else None},
            "outputs": {},
        }
        manifest["contract"]["relationship_breaks"] = {
            "output_features": list(
                relationship_break_feature_names(
                    continuous_source_contract,
                    config=relationship_break_cfg,
                )
            ),
            "semantics": (
                "strict-prequential 30/90d trailing intercept-plus-slope OLS "
                "residuals; complete raw-observable pairs only; signed and "
                "absolute residuals; no clusters, targets or outcome features"
            ),
        }
        for path in temporary.iterdir():
            if path.is_file():
                manifest["outputs"][path.name] = _sha256(path)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "manifest.sha256").write_text(_sha256(manifest_path) + "  manifest.json\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--frequency", choices=("month", "quarter"), default="quarter")
    parser.add_argument("--purge-hours", type=int, default=12)
    parser.add_argument("--max-features-per-view", type=int, default=20)
    parser.add_argument("--max-lag-hours", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--primary-state-count", choices=(3, 4, 5), type=int, default=5)
    parser.add_argument("--primary-merge-low-support-state", action="store_true")
    parser.add_argument("--systems", nargs="+", choices=tuple(spec.name for spec in DEFAULT_GEOMETRY_SPECS))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    print(materialize(panel_path=args.panel, output_dir=args.output_dir, evaluation_start=args.evaluation_start, evaluation_end=args.evaluation_end, candidate_path=args.candidates, frequency=args.frequency, purge_hours=args.purge_hours, max_features_per_view=args.max_features_per_view, max_lag_hours=args.max_lag_hours, seed=args.seed, primary_state_count=args.primary_state_count, primary_merge_low_support_state=args.primary_merge_low_support_state, systems=args.systems))
