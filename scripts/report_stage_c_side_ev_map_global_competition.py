#!/usr/bin/env python3
"""Compare side-local meta models through one frozen expected-EV unit.

Raw scores from independently trained long and short models are not comparable.
This reporter first fits a train-only, side x archetype monotone expected-EV
map for each supplied model stream, then performs pooled global top-k selection
on that common expected-EV unit.  It intentionally has no timestamp-local
ranking mode: timestamp-side ranking is a candidate-stream diagnostic, not a
global-auction decision rule.

Every mapping reference must be supplied explicitly, be row-disjoint from its
evaluation ledger, and end before the earliest evaluation timestamp.  Net
outcomes are read from the ledger's stored ``ev_after_1pct`` column.  The
reporter never deducts the 1% round-trip cost a second time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.portfolio_policy_replay import (
    _select_ev_curve,
    fit_hierarchical_ev_curves,
)


PREDICTION_NAME = "s52_train_meta_regime_handoff_smoke_predictions.parquet"
TOP_FRACS = (0.01, 0.05, 0.10, 0.20, 0.30)
NET_EV_COL = "ev_after_1pct"
GROSS_EV_COL = "first_touch_gross"
SCORE_CANDIDATES = (
    "score_meta_base_soft_label",
    "score_meta",
    "meta_score_oof",
    "score",
)
REQUIRED_COLUMNS = ("__ts__", "__symbol__", "side_name", NET_EV_COL, GROSS_EV_COL)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path).expanduser()


def _resolve_prediction_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (
        path / "best_full_oos" / PREDICTION_NAME,
        path / PREDICTION_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    recursive = sorted(path.glob(f"**/{PREDICTION_NAME}"))
    if len(recursive) == 1:
        return recursive[0]
    if not recursive:
        raise FileNotFoundError(f"no {PREDICTION_NAME} below {path}")
    raise ValueError(
        f"ambiguous prediction directory {path}; pass the exact parquet path: "
        + ", ".join(str(item) for item in recursive[:6])
    )


def _pick_score_column(frame: pd.DataFrame, requested: str | None) -> str:
    candidates = [requested] if requested else []
    candidates.extend(SCORE_CANDIDATES)
    for column in candidates:
        if column and column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return str(column)
    raise ValueError(
        "ledger has no usable meta score column; expected one of "
        + ", ".join(SCORE_CANDIDATES)
    )


def _normalise_archetype(frame: pd.DataFrame) -> pd.Series:
    for column in (
        "archetype_policy_key",
        "policy_archetype",
        "local_side_archetype",
        "archetype_label_family",
        "__archetype_policy_key__",
    ):
        if column in frame.columns:
            values = frame[column].astype("string").fillna("missing").copy()
            sides = frame["side_name"].astype("string").str.lower()
            for side in ("long", "short"):
                mask = sides.eq(side) & values.str.startswith(f"{side}__", na=False)
                values.loc[mask] = values.loc[mask].str[len(side) + 2 :]
            return values.astype(str)
    return pd.Series("missing", index=frame.index, dtype="object")


def _read_ledger(
    path: Path,
    *,
    source_name: str,
    score_col: str | None,
) -> tuple[pd.DataFrame, str]:
    resolved = _resolve_prediction_path(path)
    frame = pd.read_parquet(resolved)
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{resolved} lacks required columns: {missing}")
    actual_score_col = _pick_score_column(frame, score_col)
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["side_name"] = out["side_name"].astype(str).str.lower()
    out["__archetype__"] = _normalise_archetype(out)
    out["__source__"] = str(source_name)
    out["__raw_score__"] = pd.to_numeric(out[actual_score_col], errors="coerce")
    out[NET_EV_COL] = pd.to_numeric(out[NET_EV_COL], errors="coerce")
    out[GROSS_EV_COL] = pd.to_numeric(out[GROSS_EV_COL], errors="coerce")
    valid = (
        out["__ts__"].notna()
        & out["__symbol__"].ne("")
        & out["side_name"].isin(("long", "short"))
        & np.isfinite(out["__raw_score__"])
        & np.isfinite(out[NET_EV_COL])
    )
    out = out.loc[valid].copy()
    if out.empty:
        raise ValueError(f"{resolved} has no finite scored outcome rows")
    stored_cost = out[GROSS_EV_COL] - out[NET_EV_COL]
    if not np.allclose(stored_cost.to_numpy(dtype=np.float64), 0.01, rtol=0.0, atol=1e-7):
        raise ValueError(
            f"{resolved} does not satisfy the exact stored 1% round-trip cost contract "
            f"({GROSS_EV_COL} - {NET_EV_COL} must equal 0.01)"
        )
    key = _row_key(out)
    if key.duplicated().any():
        raise ValueError(
            f"{resolved} has duplicate timestamp/symbol/side/archetype keys: "
            f"{int(key.duplicated().sum())}"
        )
    out.attrs["source_path"] = str(resolved)
    out.attrs["score_column"] = actual_score_col
    return out, actual_score_col


def _row_key(frame: pd.DataFrame) -> pd.Index:
    return pd.Index(
        frame["__ts__"].astype("int64").astype(str)
        + "|"
        + frame["__symbol__"].astype(str)
        + "|"
        + frame["side_name"].astype(str)
        + "|"
        + frame["__archetype__"].astype(str),
        dtype="object",
    )


def _validate_reference(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    source_name: str,
) -> None:
    overlap = _row_key(reference).intersection(_row_key(evaluation))
    if len(overlap):
        raise ValueError(
            f"mapping reference for {source_name!r} overlaps evaluation rows "
            f"({len(overlap)} keys); refusing a leaky map"
        )
    reference_end = reference["__ts__"].max()
    evaluation_start = evaluation["__ts__"].min()
    if not reference_end < evaluation_start:
        raise ValueError(
            f"mapping reference for {source_name!r} is not strictly prior to its "
            f"evaluation rows: reference_end={reference_end}, "
            f"evaluation_start={evaluation_start}"
        )


def _score_reference_keys(reference: pd.DataFrame) -> dict[tuple[str, ...], np.ndarray]:
    keys: dict[tuple[str, ...], np.ndarray] = {}
    groups = (
        ("source_side_archetype", ["__source__", "side_name", "__archetype__"]),
        ("source_side", ["__source__", "side_name"]),
        ("source", ["__source__"]),
    )
    for level, columns in groups:
        for value, group in reference.groupby(columns, observed=True, sort=False):
            values = np.sort(group["__raw_score__"].to_numpy(dtype=np.float64))
            key_values = value if isinstance(value, tuple) else (value,)
            keys[(level, *map(str, key_values))] = values
    return keys


def _rank_from_reference(
    frame: pd.DataFrame,
    score_reference: dict[tuple[str, ...], np.ndarray],
) -> np.ndarray:
    out = np.full(len(frame), np.nan, dtype=np.float64)
    score = frame["__raw_score__"].to_numpy(dtype=np.float64)
    groups = frame.groupby(
        ["__source__", "side_name", "__archetype__"], observed=True, sort=False
    ).indices
    for raw_key, positions in groups.items():
        source, side, archetype = map(str, raw_key)
        candidates = (
            ("source_side_archetype", source, side, archetype),
            ("source_side", source, side),
            ("source", source),
        )
        values = next(
            (score_reference[key] for key in candidates if key in score_reference),
            None,
        )
        if values is None or len(values) == 0:
            continue
        out[positions] = (
            np.searchsorted(values, score[positions], side="right") / float(len(values))
        )
    if not np.isfinite(out).all():
        raise ValueError("score rank mapping has missing train-only reference support")
    return np.clip(out, 0.0, 1.0)


def fit_side_archetype_expected_ev_map(
    reference: pd.DataFrame,
    *,
    bins: int = 30,
    min_group_rows: int = 80,
    shrink_rows: int = 240,
) -> dict[str, Any]:
    """Fit common-unit expected-EV curves from pre-evaluation OOF rows only."""
    score_reference = _score_reference_keys(reference)
    ranks = _rank_from_reference(reference, score_reference)
    fit_rows = pd.DataFrame(
        {
            "timestamp": reference["__ts__"].to_numpy(),
            "symbol": reference["__symbol__"].astype(str).to_numpy(),
            "side": reference["side_name"].astype(str).to_numpy(),
            "strategy_id": reference["__source__"].astype(str).to_numpy(),
            "policy_archetype": reference["__archetype__"].astype(str).to_numpy(),
            "normalized_rank_score": ranks,
            "base_strategy_threshold": 0.0,
            "calibrated_score": ranks,
            "entry_price": 1.0,
            "exit_timestamp": reference["__ts__"].to_numpy(),
            "exit_price": 1.0,
            "holding_bars": 1.0,
            "net_return": reference[NET_EV_COL].to_numpy(dtype=np.float64),
            # Required only by the generic candidate-table normaliser. The map
            # learns exclusively from the already net, stored outcome above.
            "gross_return": reference[NET_EV_COL].to_numpy(dtype=np.float64),
            "simple_policy_exit_reason": "stored_outcome",
            "fees_bps": 0.0,
        }
    )
    curves = fit_hierarchical_ev_curves(
        fit_rows,
        bins=int(bins),
        min_group_rows=int(min_group_rows),
        shrink_rows=int(shrink_rows),
    )
    return {
        "schema": "stage_c_side_archetype_expected_ev_map_v1",
        "net_outcome_column": NET_EV_COL,
        "round_trip_cost_contract": "stored_once_in_ev_after_1pct",
        "score_reference": {
            "source_side_archetype": {
                "|".join(key[1:]): values.tolist()
                for key, values in score_reference.items()
                if key[0] == "source_side_archetype"
            },
            "source_side": {
                "|".join(key[1:]): values.tolist()
                for key, values in score_reference.items()
                if key[0] == "source_side"
            },
            "source": {
                "|".join(key[1:]): values.tolist()
                for key, values in score_reference.items()
                if key[0] == "source"
            },
        },
        "curves": curves,
        "reference_rows": int(len(reference)),
        "reference_start": reference["__ts__"].min(),
        "reference_end": reference["__ts__"].max(),
    }


def _unpack_score_reference(mapping: dict[str, Any]) -> dict[tuple[str, ...], np.ndarray]:
    raw = dict(mapping.get("score_reference") or {})
    out: dict[tuple[str, ...], np.ndarray] = {}
    for level, expected_size in (
        ("source_side_archetype", 3),
        ("source_side", 2),
        ("source", 1),
    ):
        for joined, values in dict(raw.get(level) or {}).items():
            parts = tuple(str(item) for item in str(joined).split("|"))
            if len(parts) == expected_size:
                out[(level, *parts)] = np.asarray(values, dtype=np.float64)
    return out


def apply_side_archetype_expected_ev_map(
    evaluation: pd.DataFrame,
    mapping: dict[str, Any],
) -> pd.DataFrame:
    """Apply a frozen mapping. No realized evaluation fields are consumed."""
    score_reference = _unpack_score_reference(mapping)
    result = evaluation.copy()
    rank = _rank_from_reference(result, score_reference)
    curves = dict(mapping.get("curves") or {})
    expected = np.full(len(result), np.nan, dtype=np.float64)
    groups = result.groupby(
        ["__source__", "side_name", "__archetype__"], observed=True, sort=False
    ).indices
    for raw_key, positions in groups.items():
        source, side, archetype = map(str, raw_key)
        curve = _select_ev_curve(
            curves,
            strategy_id=source,
            side=side,
            policy_archetype=archetype,
        )
        x = np.asarray(curve.get("x", (0.0, 1.0)), dtype=np.float64)
        y = np.asarray(curve.get("y", (0.0, 0.0)), dtype=np.float64)
        expected[positions] = np.interp(
            rank[positions], x, y, left=y[0], right=y[-1]
        )
    if not np.isfinite(expected).all():
        raise ValueError("frozen expected-EV map emitted non-finite values")
    result["score_train_reference_pct"] = rank.astype(np.float32)
    result["expected_ev_side_archetype"] = expected.astype(np.float32)
    return result


def _topk(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    n = max(1, int(math.ceil(len(frame) * float(fraction))))
    return frame.sort_values(
        ["expected_ev_side_archetype", "__ts__", "__symbol__", "side_name"],
        ascending=[False, True, True, True],
        kind="stable",
    ).head(n).copy()


def _outcome_rate(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    return float(pair["left"].corr(pair["right"], method="spearman"))


def _metric_row(
    selected: pd.DataFrame,
    *,
    variant: str,
    top_frac: float,
    scope: str,
    group_values: dict[str, Any],
    global_selected_rows: int,
) -> dict[str, Any]:
    ev = pd.to_numeric(selected[NET_EV_COL], errors="coerce")
    ts = pd.to_datetime(selected["__ts__"], utc=True, errors="coerce")
    calendar_ts = ts.dt.tz_localize(None)
    days = int(ts.dt.floor("D").nunique())
    daily = pd.DataFrame({"day": ts.dt.floor("D"), "ev": ev}).groupby("day", observed=True)["ev"].mean()
    weekly = pd.DataFrame({"week": calendar_ts.dt.to_period("W-SUN").dt.start_time, "ev": ev}).groupby("week", observed=True)["ev"].mean()
    monthly = pd.DataFrame({"month": calendar_ts.dt.to_period("M").astype(str), "ev": ev}).groupby("month", observed=True)["ev"].mean()
    side_values = selected["side_name"].astype(str).str.lower()
    raw_score = pd.to_numeric(selected["__raw_score__"], errors="coerce")
    expected_ev = pd.to_numeric(selected["expected_ev_side_archetype"], errors="coerce")
    return {
        "variant": variant,
        "scope": scope,
        "selection_basis": "pooled_global_topk_after_train_only_side_archetype_ev_mapping",
        "top_frac": float(top_frac),
        "selected_rows": int(len(selected)),
        "selected_share_of_global": float(len(selected) / max(global_selected_rows, 1)),
        "selected_days": days,
        "trades_per_day": float(len(selected) / max(days, 1)),
        "mean_ev_after_1pct": float(ev.mean()),
        "sum_ev_after_1pct": float(ev.sum()),
        "positive_ev_rate": float(ev.gt(0.0).mean()),
        "selected_long_share": float(side_values.eq("long").mean()),
        "selected_short_share": float(side_values.eq("short").mean()),
        "mean_expected_ev": float(expected_ev.mean()),
        "mean_raw_score": float(raw_score.mean()),
        "mean_score_train_reference_pct": float(pd.to_numeric(selected["score_train_reference_pct"], errors="coerce").mean()),
        "raw_score_ev_spearman": _spearman(raw_score, ev),
        "mapped_expected_ev_spearman": _spearman(expected_ev, ev),
        "clean_exec_precision": _outcome_rate(selected, "clean_exec"),
        "dirty_positive_rate": _outcome_rate(selected, "dirty_positive"),
        "first_touch_bad_mae_rate": _outcome_rate(selected, "first_touch_bad_mae_1r"),
        "full_path_bad_mae_rate": _outcome_rate(selected, "full_path_bad_mae_1r"),
        "timeout_rate": _outcome_rate(selected, "timeout"),
        "worst_week_mean_ev_after_1pct": float(weekly.min()) if len(weekly) else float("nan"),
        "worst_month_mean_ev_after_1pct": float(monthly.min()) if len(monthly) else float("nan"),
        **group_values,
    }


def _metrics_for_variant(mapped: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected_parts: list[pd.DataFrame] = []
    for fraction in TOP_FRACS:
        selected = _topk(mapped, fraction)
        selected["variant"] = variant
        selected["top_frac"] = float(fraction)
        selected_parts.append(selected)
        rows.append(
            _metric_row(
                selected,
                variant=variant,
                top_frac=fraction,
                scope="global",
                group_values={"side_name": "all", "month": "all", "archetype": "all"},
                global_selected_rows=len(selected),
            )
        )
        grouped_scopes: Iterable[tuple[str, list[str], dict[str, str]]] = (
            ("side", ["side_name"], {"month": "all", "archetype": "all"}),
            ("month", ["__month__"], {"side_name": "all", "archetype": "all"}),
            ("archetype", ["__archetype__"], {"side_name": "all", "month": "all"}),
            ("month_side_archetype", ["__month__", "side_name", "__archetype__"], {}),
        )
        selected["__month__"] = selected["__ts__"].dt.tz_localize(None).dt.to_period("M").astype(str)
        for scope, columns, defaults in grouped_scopes:
            for values, group in selected.groupby(columns, observed=True, sort=True):
                values_tuple = values if isinstance(values, tuple) else (values,)
                detail = dict(defaults)
                for column, value in zip(columns, values_tuple):
                    detail[{"__month__": "month", "__archetype__": "archetype"}.get(column, column)] = str(value)
                rows.append(
                    _metric_row(
                        group,
                        variant=variant,
                        top_frac=fraction,
                        scope=scope,
                        group_values=detail,
                        global_selected_rows=len(selected),
                    )
                )
    return pd.DataFrame(rows), pd.concat(selected_parts, ignore_index=True, copy=False)


def _compose_variant(
    *,
    name: str,
    long_source: pd.DataFrame,
    short_source: pd.DataFrame,
) -> pd.DataFrame:
    long_rows = long_source.loc[long_source["side_name"].eq("long")].copy()
    short_rows = short_source.loc[short_source["side_name"].eq("short")].copy()
    if long_rows.empty or short_rows.empty:
        raise ValueError(f"{name} needs both long and short evaluation rows")
    return pd.concat([long_rows, short_rows], ignore_index=True, copy=False)


def run_global_competition(
    *,
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    stage_c_names: list[str],
    benchmark_long_name: str,
    benchmark_short_name: str,
    bins: int = 30,
    min_group_rows: int = 80,
    shrink_rows: int = 240,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    """Return metrics and mappings for Stage-C and benchmark-side hybrids."""
    for name, (evaluation, reference) in sources.items():
        _validate_reference(reference, evaluation, source_name=name)
    reference = pd.concat([item[1] for item in sources.values()], ignore_index=True, copy=False)
    mapping = fit_side_archetype_expected_ev_map(
        reference,
        bins=bins,
        min_group_rows=min_group_rows,
        shrink_rows=shrink_rows,
    )
    mapped_sources = {
        name: apply_side_archetype_expected_ev_map(evaluation, mapping)
        for name, (evaluation, _) in sources.items()
    }
    variants: dict[str, pd.DataFrame] = {
        name: _compose_variant(name=name, long_source=mapped_sources[name], short_source=mapped_sources[name])
        for name in stage_c_names
    }
    variants["hybrid_weighted_packb_long_current_purged_short"] = _compose_variant(
        name="hybrid_weighted_packb_long_current_purged_short",
        long_source=mapped_sources[benchmark_long_name],
        short_source=mapped_sources[benchmark_short_name],
    )
    metric_parts: list[pd.DataFrame] = []
    selected_by_variant: dict[str, pd.DataFrame] = {}
    for name, frame in variants.items():
        metrics, selected = _metrics_for_variant(frame, name)
        metric_parts.append(metrics)
        selected_by_variant[name] = selected
    return (
        pd.concat(metric_parts, ignore_index=True, copy=False),
        pd.concat(selected_by_variant.values(), ignore_index=True, copy=False),
        mapping,
        variants,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-c-arm", action="append", default=[], metavar="NAME=PATH", help="Completed Stage-C v2 arm directory or prediction parquet. Repeatable.")
    parser.add_argument("--benchmark-long", type=Path, required=True, help="Weighted Pack-B benchmark prediction ledger or directory.")
    parser.add_argument("--benchmark-short", type=Path, required=True, help="Current Purged benchmark prediction ledger or directory.")
    parser.add_argument("--mapping-reference", action="append", default=[], metavar="NAME=PATH", help="Disjoint, strictly earlier OOF/training reference for each source. Names: Stage-C arm names, weighted_packb_long, current_purged_short.")
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--min-group-rows", type=int, default=80)
    parser.add_argument("--shrink-rows", type=int, default=240)
    args = parser.parse_args()
    if not args.stage_c_arm:
        parser.error("at least one --stage-c-arm NAME=PATH is required")
    stage_specs = dict(_parse_named_path(value) for value in args.stage_c_arm)
    if len(stage_specs) != len(args.stage_c_arm):
        parser.error("duplicate Stage-C arm names")
    references = dict(_parse_named_path(value) for value in args.mapping_reference)
    required_names = set(stage_specs) | {"weighted_packb_long", "current_purged_short"}
    missing_refs = sorted(required_names.difference(references))
    if missing_refs:
        parser.error("missing disjoint --mapping-reference for: " + ", ".join(missing_refs))

    evaluation_paths: dict[str, Path] = {
        **stage_specs,
        "weighted_packb_long": args.benchmark_long,
        "current_purged_short": args.benchmark_short,
    }
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    source_manifest: dict[str, Any] = {}
    for name, evaluation_path in evaluation_paths.items():
        evaluation, score = _read_ledger(evaluation_path, source_name=name, score_col=args.score_col)
        reference, reference_score = _read_ledger(references[name], source_name=name, score_col=args.score_col)
        sources[name] = (evaluation, reference)
        source_manifest[name] = {
            "evaluation_path": evaluation.attrs["source_path"],
            "reference_path": reference.attrs["source_path"],
            "score_column": score,
            "reference_score_column": reference_score,
            "evaluation_rows": int(len(evaluation)),
            "reference_rows": int(len(reference)),
            "reference_end": reference["__ts__"].max(),
            "evaluation_start": evaluation["__ts__"].min(),
        }

    metrics, selected, mapping, variants = run_global_competition(
        sources=sources,
        stage_c_names=list(stage_specs),
        benchmark_long_name="weighted_packb_long",
        benchmark_short_name="current_purged_short",
        bins=args.bins,
        min_group_rows=args.min_group_rows,
        shrink_rows=args.shrink_rows,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "stage_c_side_ev_map_global_competition_metrics.csv", index=False)
    selected.to_parquet(args.out_dir / "stage_c_side_ev_map_global_competition_selected.parquet", index=False, compression="zstd")
    (args.out_dir / "stage_c_side_ev_map_global_competition_mapping.json").write_text(
        json.dumps(_json_safe(mapping), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "stage_c_side_ev_map_global_competition_v1",
        "selection_basis": "pooled_global_topk_after_train_only_side_archetype_ev_mapping",
        "top_fractions": list(TOP_FRACS),
        "net_ev_column": NET_EV_COL,
        "round_trip_cost": 0.01,
        "round_trip_cost_contract": "stored_once_in_ev_after_1pct; reporter subtracts no further fee or spread",
        "mapping_contract": "strictly earlier, row-disjoint supplied OOF/training reference; side_x_archetype monotone hierarchical EV curves",
        "sources": source_manifest,
        "variants": {name: int(len(frame)) for name, frame in variants.items()},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    global_top10 = metrics.loc[(metrics["scope"] == "global") & (metrics["top_frac"] == 0.10)]
    print(global_top10[["variant", "selected_rows", "mean_ev_after_1pct", "worst_week_mean_ev_after_1pct", "worst_month_mean_ev_after_1pct"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
