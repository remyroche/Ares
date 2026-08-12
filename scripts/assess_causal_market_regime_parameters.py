#!/usr/bin/env python3
"""Sequential, label-free K/stickiness assessment for causal regime systems.

The funnel is intentionally not a full factorial search:

1. screen K on three chronological proxy folds and two seeds at one moderate
   stickiness;
2. choose K by structural Pareto/one-SE diagnostics;
3. screen stickiness only for that selected K on the same proxy folds/seeds;
4. emit per-view bundles and cross-view redundancy for a later *supervised*
   feature experiment.

Neither the runner nor its selector reads labels, returns, scores, policies or
other outcome-like columns.  It is a representation-quality tool only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_market_regime_assessment import (  # noqa: E402
    RegimeAssessmentColumns,
    RegimeAssessmentConfig,
    assess_regime_candidate_timeline,
    candidate_grid,
    regime_feature_bundle,
    regime_output_columns,
    select_regime_parameter_recommendation,
)
from extreme_price_movements.causal_market_regime_systems import (  # noqa: E402
    CausalMarketRegimeConfig,
    DEFAULT_GEOMETRY_SPECS,
    FORBIDDEN_INPUT_TOKENS,
    fit_causal_market_regime_systems,
)
from scripts.materialize_oof_market_regime_systems import compact_observable_columns  # noqa: E402


SCHEMA = "causal_market_regime_parameter_funnel_v1"
DEFAULT_PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_market_regime_parameter_funnel_20260803_v1"
DEFAULT_SYSTEMS = ("primary", "trend_volatility", "breadth_dependence", "leverage_flow", "liquidity")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def proxy_windows(
    timestamp: pd.Series,
    *,
    evaluation_start: object,
    evaluation_end: object | None,
    folds: int = 3,
) -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    """Split the requested OOF era into contiguous, non-overlapping proxy folds."""

    if folds < 2:
        raise ValueError("proxy assessment requires at least two chronological folds")
    ts = pd.to_datetime(timestamp, utc=True, errors="raise")
    start = pd.Timestamp(evaluation_start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(evaluation_end) if evaluation_end else ts.max() + pd.Timedelta(hours=1)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    values = np.sort(ts.loc[ts.ge(start) & ts.lt(end)].unique())
    pieces = [piece for piece in np.array_split(values, folds) if len(piece)]
    if len(pieces) != folds:
        raise ValueError("requested assessment period has insufficient rows for chronological folds")
    result: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for position, piece in enumerate(pieces):
        left = pd.Timestamp(piece[0])
        right = pd.Timestamp(pieces[position + 1][0]) if position + 1 < len(pieces) else end
        result.append((f"proxy_{position + 1:02d}", left, right))
    return tuple(result)


def _prefix(system: str) -> str:
    return "market_regime" if system == "primary" else f"geometry_regime__{system}"


def _spec(system: str):
    matches = [item for item in DEFAULT_GEOMETRY_SPECS if item.name == system]
    if len(matches) != 1:
        raise ValueError(f"unknown regime system {system!r}")
    return matches[0]


def _centroid_separation(diagnostics: dict[str, Any]) -> tuple[float, float]:
    centroids = np.asarray(diagnostics.get("effective_state_centroids", []), dtype=np.float64)
    if centroids.ndim != 2 or len(centroids) < 2:
        return 0.0, 0.0
    distance = np.sqrt(np.maximum(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), 0.0))
    upper = distance[np.triu_indices(len(centroids), k=1)]
    return float(upper.min()), float(upper.mean())


def _one_candidate(
    panel: pd.DataFrame,
    fields: Sequence[str],
    *,
    system: str,
    k: int,
    stickiness: float,
    seed: int,
    windows: Sequence[tuple[str, pd.Timestamp, pd.Timestamp]],
    max_train_rows: int,
    max_proxy_rows: int,
    max_iter: int,
) -> pd.DataFrame:
    """Fit one K/rho proxy candidate separately before each held proxy fold."""

    parts: list[pd.DataFrame] = []
    candidate_id = f"{system}__k{k}__rho{stickiness:.2f}"
    for window, begin, end in windows:
        train = panel.source_utc.lt(begin)
        evaluate = panel.source_utc.ge(begin) & panel.source_utc.lt(end)
        if int(train.sum()) < max(100, int(k) * 20) or not evaluate.any():
            continue
        config = CausalMarketRegimeConfig(
            diagnostic_k_values=(int(k),),
            stickiness_values=(float(stickiness),),
            primary_state_count=int(k),
            max_train_rows=int(max_train_rows),
            max_proxy_rows=int(max_proxy_rows),
            max_iter=int(max_iter),
            random_state=int(seed),
        )
        systems = fit_causal_market_regime_systems(panel.loc[train], fields, specs=(_spec(system),), config=config)
        model = systems.models[system]
        local = systems.transform(panel.loc[evaluate], carry_history=False)
        min_separation, mean_separation = _centroid_separation(model.diagnostics)
        local.insert(0, "source_utc", panel.loc[evaluate, "source_utc"].to_numpy())
        local["candidate_id"] = candidate_id
        local["assessment_fold_id"] = f"{window}__seed{seed}"
        local["assessment_window_id"] = window
        local["assessment_seed"] = int(seed)
        local["regime_train_end_utc"] = panel.loc[train, "source_utc"].max()
        local["system"] = system
        local["candidate_k"] = int(k)
        local["candidate_stickiness"] = float(stickiness)
        local["centroid_min_separation"] = min_separation
        local["centroid_mean_separation"] = mean_separation
        parts.append(local)
    if not parts:
        raise ValueError(f"{candidate_id} had no proxy folds with adequate prior support")
    return pd.concat(parts, ignore_index=True)


def _summaries(timeline: pd.DataFrame, *, system: str, config: RegimeAssessmentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = assess_regime_candidate_timeline(
        timeline,
        prefix=_prefix(system),
        columns=RegimeAssessmentColumns(),
        config=config,
    )
    return result.fold_diagnostics, result.portability_diagnostics, result.candidate_summary


def _choose_k(summary: pd.DataFrame) -> int:
    recommendation = select_regime_parameter_recommendation(summary)
    if recommendation["recommended_candidate_id"]:
        return int(recommendation["candidate_k"])
    # Do not replace a failing structural gate with hidden tuning: the fallback
    # only lets the diagnostic complete and is explicitly marked in its report.
    return int(summary.sort_values(["structural_score", "candidate_id"], ascending=[False, True], kind="stable").iloc[0].candidate_k)


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3:
        return np.nan
    return float(left.loc[valid].rank(method="average").corr(right.loc[valid].rank(method="average")))


def cross_view_redundancy(
    timeline: pd.DataFrame,
    recommendations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Measure redundancy from permutation-invariant posterior shapes + scalars."""

    views: dict[str, pd.DataFrame] = {}
    for system, choice in recommendations.items():
        candidate = choice.get("recommended_candidate_id")
        if not candidate:
            continue
        local = timeline.loc[timeline.candidate_id.eq(candidate)].copy()
        if local.empty:
            continue
        probability, state = regime_output_columns(local, prefix=_prefix(system))
        ordered = np.sort(local.loc[:, probability].to_numpy(float), axis=1)
        fields = local.loc[:, ["source_utc", "assessment_window_id", "assessment_seed"]].copy()
        for number in range(ordered.shape[1]):
            fields[f"posterior_shape_{number}"] = ordered[:, number]
        for name in ("entropy", "top2_margin", "state_age_hours", "switch_probability"):
            if name in state:
                fields[name] = pd.to_numeric(local[state[name]], errors="coerce")
        views[system] = fields
    rows: list[dict[str, Any]] = []
    names = sorted(views)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            left, right = views[left_name], views[right_name]
            joined = left.merge(right, on=["source_utc", "assessment_window_id", "assessment_seed"], suffixes=("_left", "_right"), how="inner", validate="one_to_one")
            correlations: list[float] = []
            common = set(column.removesuffix("_left") for column in joined if column.endswith("_left")) & set(column.removesuffix("_right") for column in joined if column.endswith("_right"))
            for field in sorted(common):
                value = _rank_correlation(joined[f"{field}_left"], joined[f"{field}_right"])
                if np.isfinite(value):
                    correlations.append(abs(value))
            rows.append({"schema": SCHEMA, "left_system": left_name, "right_system": right_name, "rows": int(len(joined)), "mean_abs_spearman": float(np.mean(correlations)) if correlations else np.nan, "max_abs_spearman": float(np.max(correlations)) if correlations else np.nan, "redundancy_proxy": "permutation_invariant_sorted_posterior_shape_plus_entropy_margin_age_switch"})
    return pd.DataFrame(rows)


def run(
    *,
    panel_path: Path = DEFAULT_PANEL,
    output_dir: Path = DEFAULT_OUTPUT,
    evaluation_start: str,
    evaluation_end: str | None = None,
    systems: Sequence[str] = DEFAULT_SYSTEMS,
    k_values: Sequence[int] = (3, 4, 5, 6),
    stickiness_values: Sequence[float] = (0.0, 0.35, 0.60, 0.80),
    screen_stickiness: float = 0.35,
    seeds: Sequence[int] = (20260803, 20260819),
    proxy_folds: int = 3,
    max_features_per_view: int = 16,
    max_train_rows: int = 12_000,
    max_proxy_rows: int = 3_000,
    max_iter: int = 60,
) -> Path:
    """Run the bounded two-stage parameter funnel and persist no outcome data."""

    panel_path, output_dir = Path(panel_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    wanted = tuple(dict.fromkeys(str(value) for value in systems))
    if not wanted or any(value not in DEFAULT_SYSTEMS for value in wanted):
        raise ValueError(f"unknown systems: {wanted}")
    grid = candidate_grid(k_values, (screen_stickiness,))
    candidate_grid((k_values[0],), stickiness_values)  # validates stickiness without expanding stage A.
    if len(seeds) != 2:
        raise ValueError("the bounded robustness protocol requires exactly two deterministic seeds")
    schema = pq.ParquetFile(panel_path).schema.names
    fields = compact_observable_columns(panel_path, max_per_view=max_features_per_view)
    forbidden = [field for field in fields if any(token in field.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if forbidden:
        raise ValueError(f"observable proxy selection admitted forbidden fields: {forbidden}")
    panel = pd.read_parquet(panel_path, columns=["source_utc", *fields])
    panel["source_utc"] = pd.to_datetime(panel["source_utc"], utc=True, errors="raise")
    panel = panel.sort_values("source_utc", kind="stable").drop_duplicates("source_utc", keep="last").reset_index(drop=True)
    windows = proxy_windows(panel.source_utc, evaluation_start=evaluation_start, evaluation_end=evaluation_end, folds=proxy_folds)
    assessment_config = RegimeAssessmentConfig()
    all_timeline: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    all_portability: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    recommendations: dict[str, dict[str, Any]] = {}
    bundles: dict[str, list[str]] = {}
    for system in wanted:
        stage_a = [
            _one_candidate(panel, fields, system=system, k=k, stickiness=float(screen_stickiness), seed=int(seed), windows=windows, max_train_rows=max_train_rows, max_proxy_rows=max_proxy_rows, max_iter=max_iter)
            for k, _rho in grid for seed in seeds
        ]
        timeline_a = pd.concat(stage_a, ignore_index=True)
        _fold_a, _port_a, summary_a = _summaries(timeline_a, system=system, config=assessment_config)
        selected_k = _choose_k(summary_a)
        stage_b = [
            _one_candidate(panel, fields, system=system, k=selected_k, stickiness=float(rho), seed=int(seed), windows=windows, max_train_rows=max_train_rows, max_proxy_rows=max_proxy_rows, max_iter=max_iter)
            for rho in stickiness_values if not np.isclose(float(rho), float(screen_stickiness)) for seed in seeds
        ]
        timeline = pd.concat([timeline_a, *stage_b], ignore_index=True)
        folds, portability, summary = _summaries(timeline, system=system, config=assessment_config)
        recommendation = select_regime_parameter_recommendation(summary)
        recommendation.update({"system": system, "stage_a_selected_k": selected_k, "stage": "sequential_k_then_stickiness"})
        recommendations[system] = recommendation
        all_timeline.append(timeline)
        all_folds.append(folds)
        all_portability.append(portability)
        all_summary.append(summary)
        bundles[system] = list(regime_feature_bundle(timeline, prefix=_prefix(system)))
    timeline = pd.concat(all_timeline, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    portability = pd.concat(all_portability, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    redundancy = cross_view_redundancy(timeline, recommendations)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        timeline.to_parquet(temporary / "candidate_proxy_timelines.parquet", index=False, compression="zstd")
        folds.to_parquet(temporary / "candidate_fold_diagnostics.parquet", index=False, compression="zstd")
        portability.to_parquet(temporary / "candidate_portability_diagnostics.parquet", index=False, compression="zstd")
        summary.to_parquet(temporary / "candidate_parameter_summary.parquet", index=False, compression="zstd")
        redundancy.to_parquet(temporary / "cross_view_redundancy.parquet", index=False, compression="zstd")
        (temporary / "recommendations.json").write_text(json.dumps(_safe(recommendations), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "per_view_feature_bundles.json").write_text(json.dumps(bundles, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema": SCHEMA,
            "status": "LABEL_FREE_SEQUENTIAL_PROXY_PARAMETER_ASSESSMENT",
            "input": {"path": str(panel_path.resolve()), "sha256": _sha256(panel_path)},
            "contract": {"selection": "three chronological folds, two seeds, K screen then stickiness screen for selected K only; no full factorial", "outcomes": "forbidden; no labels, returns, score or policy fields read", "portability": "permutation-invariant sorted posterior occupancy/shape and scalar drift; no cross-fold state-id semantics", "horizon": "6-12h persistence/transition structural gate"},
            "parameters": {"systems": wanted, "k_values": list(k_values), "stickiness_values": list(stickiness_values), "screen_stickiness": screen_stickiness, "seeds": list(seeds), "proxy_windows": [[name, str(begin), str(end)] for name, begin, end in windows], "max_train_rows": max_train_rows, "max_proxy_rows": max_proxy_rows, "max_iter": max_iter, "feature_proxy": fields},
            "assessment_config": asdict(assessment_config),
            "outputs": {},
        }
        for path in temporary.iterdir():
            if path.is_file():
                manifest["outputs"][path.name] = _sha256(path)
        (temporary / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _comma_int(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _comma_float(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end")
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--k-values", type=_comma_int, default=(3, 4, 5, 6))
    parser.add_argument("--stickiness-values", type=_comma_float, default=(0.0, 0.35, 0.60, 0.80))
    parser.add_argument("--screen-stickiness", type=float, default=0.35)
    parser.add_argument("--seeds", type=_comma_int, default=(20260803, 20260819))
    parser.add_argument("--proxy-folds", type=int, default=3)
    parser.add_argument("--max-features-per-view", type=int, default=16)
    parser.add_argument("--max-train-rows", type=int, default=12_000)
    parser.add_argument("--max-proxy-rows", type=int, default=3_000)
    parser.add_argument("--max-iter", type=int, default=60)
    args = parser.parse_args()
    print(run(panel_path=args.panel, output_dir=args.output_dir, evaluation_start=args.evaluation_start, evaluation_end=args.evaluation_end, systems=args.systems, k_values=args.k_values, stickiness_values=args.stickiness_values, screen_stickiness=args.screen_stickiness, seeds=args.seeds, proxy_folds=args.proxy_folds, max_features_per_view=args.max_features_per_view, max_train_rows=args.max_train_rows, max_proxy_rows=args.max_proxy_rows, max_iter=args.max_iter))


if __name__ == "__main__":
    main()
