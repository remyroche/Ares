import hashlib
import json
import os
from dataclasses import asdict, replace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.periods_symbols_management import (
    ConsumerSlicePlan,
    EventSchema,
    InnerFold,
    OuterFold,
    SlicePlanner,
    SlicePlannerConfig,
    SymbolPolicy,
)
from extreme_price_movements.utils import tprint


def _event_symbol_count(events_df: Optional[pd.DataFrame]) -> int:
    if events_df is None or events_df.empty or "symbol" not in events_df.columns:
        return 0
    return int(events_df["symbol"].nunique())


_INDEX_ARRAY_MIN_COMPACT_LEN = 64


def _default(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _is_int_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.integer)
    if not isinstance(value, (list, tuple)):
        return False
    return all(isinstance(x, (int, np.integer)) for x in value)


def _int_sequence_to_ranges(values: Any) -> dict:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return {"encoding": "int_ranges_v1", "count": 0, "ranges": []}

    ranges: list[list[int]] = []
    start = int(arr[0])
    prev = start
    for raw in arr[1:]:
        current = int(raw)
        if current == prev + 1:
            prev = current
            continue
        ranges.append([start, prev + 1])
        start = current
        prev = current
    ranges.append([start, prev + 1])
    return {
        "encoding": "int_ranges_v1",
        "count": int(arr.size),
        "ranges": ranges,
    }


def _ranges_to_int_list(payload: dict) -> list[int]:
    values: list[int] = []
    for raw_range in payload.get("ranges", []) or []:
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            continue
        start, stop = int(raw_range[0]), int(raw_range[1])
        if stop <= start:
            continue
        values.extend(range(start, stop))
    return values


def _compact_json_indices(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _compact_json_indices(v) for k, v in value.items()}
    if _is_int_sequence(value):
        seq_len = int(len(value))
        if seq_len >= _INDEX_ARRAY_MIN_COMPACT_LEN:
            return _int_sequence_to_ranges(value)
        return [int(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_compact_json_indices(v) for v in value]
    if isinstance(value, list):
        return [_compact_json_indices(v) for v in value]
    return value


def _expand_compact_indices(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("encoding") == "int_ranges_v1":
            return _ranges_to_int_list(value)
        return {k: _expand_compact_indices(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_compact_indices(v) for v in value]
    return value


def decode_slice_plan_payload(payload: dict) -> dict:
    """Return a compatibility view with compact persisted index ranges expanded."""
    return _expand_compact_indices(payload)


def _deserialize_timestamp(val):
    if val is None:
        return None
    return pd.to_datetime(val, utc=True)


def slice_plan_path(data_root: str, run_id: str) -> str:
    return os.path.join(data_root, "artifacts", run_id, "slices", "slice_plan.json")


def load_slice_plan(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return decode_slice_plan_payload(json.load(f))
    except Exception as e:
        tprint(f"Failed to load slice plan from {path}: {e}")
        return None


def save_slice_plan_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=_default, separators=(",", ":"))
    os.replace(tmp_path, path)


def compute_event_fingerprint(events_df: pd.DataFrame) -> dict:
    if events_df is None or events_df.empty:
        return {"n_events": 0, "n_symbols": 0, "hash": ""}

    n_events = len(events_df)
    n_symbols = int(events_df["symbol"].nunique())
    min_t0 = (
        events_df["t0"].min().isoformat() if not events_df["t0"].isna().all() else None
    )
    max_t0 = (
        events_df["t0"].max().isoformat() if not events_df["t0"].isna().all() else None
    )

    # Hash some deterministic rows
    sorted_ev = events_df.sort_values(["t0", "symbol"]).head(1000)
    hash_str = hashlib.md5(pd.util.hash_pandas_object(sorted_ev).values).hexdigest()

    return {
        "n_events": n_events,
        "n_symbols": n_symbols,
        "min_t0": min_t0,
        "max_t0": max_t0,
        "hash": hash_str,
    }


def _load_baseline_events(data_root: str, run_id: str) -> Optional[pd.DataFrame]:
    events_path = os.path.join(
        data_root, "artifacts", run_id, "baseline_events.parquet"
    )
    if not os.path.exists(events_path):
        return None
    return pd.read_parquet(events_path)


def _load_label_events_for_slice_plan(
    data_root: str,
    run_id: str,
) -> Optional[pd.DataFrame]:
    labels_dir = os.path.join(data_root, "artifacts", run_id, "labels")
    if not os.path.isdir(labels_dir):
        return None

    frames: list[pd.DataFrame] = []
    for name in sorted(os.listdir(labels_dir)):
        if not name.startswith("train_") or not name.endswith(".parquet"):
            continue
        path = os.path.join(labels_dir, name)
        try:
            df = pd.read_parquet(path, columns=["__ts__", "__symbol__"])
            ts_col = "__ts__"
            sym_col = "__symbol__"
        except Exception:
            try:
                df = pd.read_parquet(path, columns=["timestamp", "symbol"])
                ts_col = "timestamp"
                sym_col = "symbol"
            except Exception:
                continue
        if df.empty:
            continue
        part = pd.DataFrame(
            {
                "symbol": df[sym_col].astype(str),
                "t0": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
            }
        )
        frames.append(part.dropna(subset=["t0", "symbol"]))

    if not frames:
        return None

    events = pd.concat(frames, axis=0, ignore_index=True)
    if events.empty:
        return None
    events = (
        events.drop_duplicates(["symbol", "t0"])
        .sort_values(["t0", "symbol"])
        .reset_index(drop=True)
    )
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    events["t1"] = events["t0"]
    return events[["event_id", "symbol", "t0", "t1"]]


def _load_best_slice_plan_events(
    data_root: str,
    run_id: str,
) -> Optional[pd.DataFrame]:
    baseline = _load_baseline_events(data_root, run_id)
    label_events = _load_label_events_for_slice_plan(data_root, run_id)
    baseline_symbols = _event_symbol_count(baseline)
    label_symbols = _event_symbol_count(label_events)

    if label_events is not None and label_symbols > max(baseline_symbols, 0):
        tprint(
            f"Using label-derived slice planning events for {run_id}: "
            f"symbols={label_symbols} rows={len(label_events)} "
            f"(baseline_symbols={baseline_symbols})."
        )
        return label_events

    if baseline is not None:
        tprint(
            f"Using baseline slice planning events for {run_id}: "
            f"symbols={baseline_symbols} rows={len(baseline)}."
        )
    return baseline


def slice_plan_is_stale(
    existing: dict,
    current_fingerprint: dict,
    planner_cfg: dict,
    allocation_targets: dict = None,
) -> bool:
    if not existing:
        return True
    if int(existing.get("version", 0) or 0) != 2:
        return True
    if existing.get("event_fingerprint") != current_fingerprint:
        return True

    existing_preset = existing.get("planner", {}).get("preset")
    current_preset = planner_cfg.get("preset", "fast")
    if existing_preset != current_preset:
        return True

    existing_symbol_policy = existing.get("planner", {}).get("symbol_policy_mode")
    current_symbol_policy = planner_cfg.get("symbol_policy_mode")
    if current_symbol_policy and existing_symbol_policy != current_symbol_policy:
        return True

    existing_symbol_fraction = existing.get("planner", {}).get("symbol_fraction")
    current_symbol_fraction = planner_cfg.get("symbol_fraction")
    if current_symbol_fraction is not None:
        if existing_symbol_fraction is None:
            return True
        if float(existing_symbol_fraction) != float(current_symbol_fraction):
            return True

    if allocation_targets and existing.get("allocation_targets") != allocation_targets:
        return True

    return False


def _stage_view_has_materialized_scope(stage_view: dict) -> bool:
    if not isinstance(stage_view, dict):
        return False
    if int(stage_view.get("n_plans", 0) or 0) <= 0:
        return False
    if not stage_view.get("source_roles"):
        return False
    if not (stage_view.get("symbols") or stage_view.get("allowed_symbols")):
        return False
    if stage_view.get("allowed_periods"):
        return True
    return bool(stage_view.get("allowed_start_ts") and stage_view.get("allowed_end_ts"))


def slice_plan_has_required_materialized_views(
    existing: dict,
    required_stages: tuple[str, ...] = (
        "train_base",
        "train_meta",
        "utility_policy_optimisation",
    ),
) -> bool:
    materialized = existing.get("materialized_views", {}) if existing else {}
    if not isinstance(materialized, dict):
        return False
    return all(
        _stage_view_has_materialized_scope(materialized.get(stage_name, {}))
        for stage_name in required_stages
    )


def _with_all_symbol_planning(config: SlicePlannerConfig) -> SlicePlannerConfig:
    """Keep the time/event preset, but never subsample the symbol universe."""
    sampling = replace(
        config.preset.sampling,
        symbol_fraction=1.0,
        max_symbols=None,
        sample_symbols_on_train=False,
        sample_symbols_on_inner_train=False,
        sample_symbols_on_valid=False,
        sample_symbols_on_inner_valid=False,
        sample_symbols_on_test=False,
    )
    preset = replace(
        config.preset,
        sampling=sampling,
        symbol_policy=SymbolPolicy(
            mode="all_symbols",
            min_symbols_per_split=max(
                1, int(config.preset.symbol_policy.min_symbols_per_split)
            ),
            random_state=config.preset.symbol_policy.random_state,
        ),
    )
    return replace(config, preset=preset)


def _serialize_consumer_plan(cp: ConsumerSlicePlan) -> dict:
    return {
        "tag": cp.tag,
        "fit_idx": (
            _int_sequence_to_ranges(cp.fit_idx) if cp.fit_idx is not None else []
        ),
        "predict_idx": (
            _int_sequence_to_ranges(cp.predict_idx)
            if cp.predict_idx is not None
            else []
        ),
        "val_idx": (
            _int_sequence_to_ranges(getattr(cp, "val_idx", None))
            if getattr(cp, "val_idx", None) is not None
            else None
        ),
        "symbols_fit": list(cp.symbols_fit) if cp.symbols_fit else [],
        "symbols_predict": list(cp.symbols_predict) if cp.symbols_predict else [],
        "metadata": _compact_json_indices(cp.metadata),
    }


def _build_stage_view(
    stage_name: str,
    consumer_plans: list[ConsumerSlicePlan],
    allocation_target: float,
    source_roles: list[str] = None,
) -> dict:
    all_symbols = set()
    fit_starts = []
    fit_ends = []
    pred_starts = []
    pred_ends = []

    for cp in consumer_plans:
        if cp.symbols_fit:
            all_symbols.update(cp.symbols_fit)
        if cp.symbols_predict:
            all_symbols.update(cp.symbols_predict)

        if cp.metadata.get("fit_actual_start"):
            fit_starts.append(pd.to_datetime(cp.metadata["fit_actual_start"]))
        if cp.metadata.get("fit_actual_end"):
            fit_ends.append(pd.to_datetime(cp.metadata["fit_actual_end"]))
        if cp.metadata.get("predict_actual_start"):
            pred_starts.append(pd.to_datetime(cp.metadata["predict_actual_start"]))
        if cp.metadata.get("predict_actual_end"):
            pred_ends.append(pd.to_datetime(cp.metadata["predict_actual_end"]))

    all_starts = fit_starts + pred_starts
    all_ends = fit_ends + pred_ends

    overall_start = min(all_starts).isoformat() if all_starts else None
    overall_end = max(all_ends).isoformat() if all_ends else None

    return {
        "stage_name": stage_name,
        "allocation_target": allocation_target,
        "source_roles": source_roles or [],
        "symbols": sorted(list(all_symbols)),
        "allowed_start_ts": overall_start,
        "allowed_end_ts": overall_end,
        "n_plans": len(consumer_plans),
    }


def _stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _build_week_windows(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[dict]:
    if start_ts is None or end_ts is None or pd.isna(start_ts) or pd.isna(end_ts):
        return []
    start_ts = pd.to_datetime(start_ts, utc=True)
    end_ts = pd.to_datetime(end_ts, utc=True)
    if end_ts <= start_ts:
        return []

    windows: list[dict] = []
    cur = start_ts
    step = pd.Timedelta(days=7)
    while cur < end_ts:
        nxt = min(cur + step, end_ts)
        windows.append({"start_ts": cur.isoformat(), "end_ts": nxt.isoformat()})
        cur = nxt
    return windows


def _largest_remainder_counts(weights: list[float], total: int) -> list[int]:
    if total <= 0 or not weights:
        return [0 for _ in weights]
    weight_sum = float(sum(max(float(w), 0.0) for w in weights))
    if weight_sum <= 0.0:
        weight_sum = float(len(weights))
        weights = [1.0 for _ in weights]
    raw = [max(float(w), 0.0) / weight_sum * float(total) for w in weights]
    counts = [int(np.floor(x)) for x in raw]
    remainder = int(total - sum(counts))
    if remainder > 0:
        order = sorted(
            range(len(weights)),
            key=lambda i: (raw[i] - counts[i], weights[i], -i),
            reverse=True,
        )
        for i in order[:remainder]:
            counts[i] += 1
    return counts


def _assign_interleaved_week_periods(
    materialized_views: dict,
    allocation_targets: dict,
    run_id: str,
) -> None:
    stage_order = [
        "train_base",
        "train_meta",
        "sizer_train",
        "utility_policy_optimisation",
        "holdout_strategy_eval",
    ]
    stage_names = [s for s in stage_order if s in materialized_views]
    if not stage_names:
        return

    starts = []
    ends = []
    for stage_name in stage_names:
        view = materialized_views.get(stage_name, {})
        if view.get("allowed_start_ts"):
            starts.append(pd.to_datetime(view["allowed_start_ts"], utc=True))
        if view.get("allowed_end_ts"):
            ends.append(pd.to_datetime(view["allowed_end_ts"], utc=True))

    if not starts or not ends:
        return

    global_start = min(starts)
    global_end = max(ends)
    week_windows = _build_week_windows(global_start, global_end)
    if not week_windows:
        return

    shared_groups = [
        ("train_base", "train_meta", "utility_policy_optimisation"),
    ]

    pool_for_stage: dict[str, str] = {}
    pool_weight: dict[str, float] = {}
    for stage_name in stage_names:
        pool_key = stage_name
        for group in shared_groups:
            if stage_name in group:
                pool_key = group[0]
                break
        pool_for_stage[stage_name] = pool_key
        if pool_key not in pool_weight:
            pool_weight[pool_key] = float(
                allocation_targets.get(stage_name, 0.0) or 0.0
            )

    unique_pools = list(dict.fromkeys(pool_for_stage[s] for s in stage_names))
    pool_weights = [pool_weight[p] for p in unique_pools]
    counts = _largest_remainder_counts(pool_weights, len(week_windows))
    if sum(counts) <= 0:
        counts = _largest_remainder_counts(
            [1.0 for _ in unique_pools], len(week_windows)
        )

    permuted_week_idxs = sorted(
        range(len(week_windows)),
        key=lambda i: _stable_hash_int(f"{run_id}:{i}"),
    )
    remaining = {pool: counts[idx] for idx, pool in enumerate(unique_pools)}
    assigned_periods_pool: dict[str, list[dict]] = {pool: [] for pool in unique_pools}

    for week_idx in permuted_week_idxs:
        candidates = [p for p in unique_pools if remaining.get(p, 0) > 0]
        if not candidates:
            break
        chosen = max(
            candidates,
            key=lambda p: (
                remaining.get(p, 0),
                pool_weight.get(p, 0.0),
                -unique_pools.index(p),
            ),
        )
        assigned_periods_pool[chosen].append(week_windows[week_idx])
        remaining[chosen] -= 1

    for stage_name in stage_names:
        pool_key = pool_for_stage[stage_name]
        allocated_periods = list(assigned_periods_pool[pool_key])
        materialized_views[stage_name]["allowed_periods"] = allocated_periods
        materialized_views[stage_name]["allocated_week_indices"] = [
            int(i)
            for i, window in enumerate(week_windows)
            if window in allocated_periods
        ]
        materialized_views[stage_name]["week_allocation"] = {
            "mode": "interleaved_weeks",
            "allocated_weeks": len(assigned_periods_pool[pool_key]),
            "total_weeks": len(week_windows),
        }


def _build_slice_allocation_diagnostics(
    materialized_views: dict, total_symbols: int = 0
) -> dict:
    diagnostics: dict = {}
    for stage_name, stage_view in materialized_views.items():
        week_indices = sorted(
            int(i) for i in stage_view.get("allocated_week_indices", [])
        )
        n_weeks = len(week_indices)
        has_gap = any((b - a) > 1 for a, b in zip(week_indices, week_indices[1:]))
        symbols_allocated = len(stage_view.get("symbols", []))
        symbol_share = (
            float(symbols_allocated) / float(total_symbols)
            if total_symbols > 0
            else np.nan
        )
        diagnostics[stage_name] = {
            "weeks_allocated": n_weeks,
            "non_contiguous_weeks": bool(has_gap) if n_weeks >= 2 else False,
            "symbols_allocated": symbols_allocated,
            "symbol_share": symbol_share,
        }
    return diagnostics


def build_slice_plan(
    events_df: pd.DataFrame,
    planner_config: SlicePlannerConfig,
    run_id: str,
    ts_sig: pd.Timestamp,
    allocation_targets: dict,
) -> dict:
    tprint(f"Building new slice plan for {run_id}")
    planner = SlicePlanner(planner_config)
    bundle = planner.build(events_df)

    consumer_plans_dict = bundle["consumer_plans"]

    # Map consumer roles to stage views
    stage_mappings = {
        "train_base": "base_model_fit",
        "train_meta": "meta_model_fit",
        "sizer_train": "ridge_sizer_fit",
        "utility_policy_optimisation": "utility_policy_tuning",
        "holdout_strategy_eval": "backtest_eval",  # Or combined with policy_optimiser
    }

    serialized_consumers = {}
    materialized_views = {}

    for stage_name, role in stage_mappings.items():
        plans = consumer_plans_dict.get(role, [])
        serialized_consumers[role] = [_serialize_consumer_plan(cp) for cp in plans]
        materialized_views[stage_name] = _build_stage_view(
            stage_name,
            plans,
            allocation_targets.get(stage_name, 0.0),
            source_roles=[role],
        )

    # Also serialize any other roles just in case
    for role, plans in consumer_plans_dict.items():
        if role not in serialized_consumers:
            serialized_consumers[role] = [_serialize_consumer_plan(cp) for cp in plans]

    fingerprint = compute_event_fingerprint(events_df)

    _assign_interleaved_week_periods(materialized_views, allocation_targets, run_id)

    total_symbols = (
        int(events_df["symbol"].nunique()) if "symbol" in events_df.columns else 0
    )
    allocation_diagnostics = _build_slice_allocation_diagnostics(
        materialized_views, total_symbols=total_symbols
    )
    for stage_name, diag in allocation_diagnostics.items():
        if diag.get("weeks_allocated", 0) >= 2 and not diag.get(
            "non_contiguous_weeks", False
        ):
            tprint(
                f"Slice plan warning: stage {stage_name} has contiguous-only week allocation; "
                "check planner diversity."
            )
        symbol_share = diag.get("symbol_share", np.nan)
        if np.isfinite(symbol_share) and symbol_share < 0.20:
            tprint(
                f"Slice plan warning: stage {stage_name} has low symbol coverage "
                f"({symbol_share:.1%} of universe)."
            )

    payload = {
        "version": 2,
        "run_id": run_id,
        "ts_sig": ts_sig.isoformat(),
        "planner": {
            "preset": planner_config.preset.preset_name,
            "symbol_policy_mode": planner_config.preset.symbol_policy.mode,
            "symbol_fraction": float(planner_config.preset.sampling.symbol_fraction),
            # include other config bits if needed
        },
        "allocation_targets": allocation_targets,
        "event_fingerprint": fingerprint,
        "consumer_plans": serialized_consumers,
        "materialized_views": materialized_views,
        "allocation_diagnostics": allocation_diagnostics,
    }

    return payload


def load_or_build_slice_plan(
    cfg: dict,
    ts_sig: pd.Timestamp,
    events_df: Optional[pd.DataFrame] = None,
    force_refresh: bool = False,
) -> dict:
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    path = slice_plan_path(cfg["data_root"], run_id)

    preset_name = cfg.get("slice_planner_preset", "fast")
    if preset_name == "robust":
        planner_config = SlicePlannerConfig.robust_defaults(schema=EventSchema())
    else:
        planner_config = SlicePlannerConfig.fast_defaults(schema=EventSchema())
    planner_config = _with_all_symbol_planning(planner_config)

    allocation_targets = {
        "train_base": 0.70,
        "train_meta": 0.70,
        "sizer_train": 0.25,
        "utility_policy_optimisation": 0.70,
        "holdout_strategy_eval": 0.05,
    }

    # Try to load existing
    if not force_refresh and os.path.exists(path):
        existing = load_slice_plan(path)
        if existing:
            # We need events to check if stale, unless we just trust it.
            # If events_df is none, try to load baseline events
            if events_df is None:
                events_df = _load_best_slice_plan_events(cfg["data_root"], run_id)

            fingerprint = (
                compute_event_fingerprint(events_df)
                if events_df is not None
                else existing.get("event_fingerprint")
            )

            is_stale = slice_plan_is_stale(
                existing,
                fingerprint,
                {
                    "preset": preset_name,
                    "symbol_policy_mode": planner_config.preset.symbol_policy.mode,
                    "symbol_fraction": planner_config.preset.sampling.symbol_fraction,
                },
                allocation_targets,
            )
            if not is_stale:
                if not slice_plan_has_required_materialized_views(existing):
                    tprint(
                        f"Slice plan for {run_id} has empty required materialized "
                        "training/policy views, rebuilding."
                    )
                else:
                    tprint(f"Loaded valid slice plan for {run_id}")
                    return existing
            else:
                tprint(f"Slice plan for {run_id} is stale, rebuilding.")

    if events_df is None:
        events_df = _load_best_slice_plan_events(cfg["data_root"], run_id)
        if events_df is None:
            raise ValueError(
                "events_df not provided and no baseline or label events found. "
                "Run labels first to generate events."
            )

    plan = build_slice_plan(
        events_df, planner_config, run_id, ts_sig, allocation_targets
    )
    save_slice_plan_atomic(path, plan)
    tprint(f"Saved new slice plan to {path}")
    return plan


def restrict_stage_symbols(stage_view: dict, max_assets: Optional[int]) -> dict:
    """Deterministically restrict the number of symbols in a stage view."""
    if max_assets is None or max_assets <= 0:
        return stage_view

    symbols = sorted(stage_view.get("symbols", []))
    if len(symbols) <= max_assets:
        return stage_view

    # Canonical subset: take the first max_assets after sorting
    # Alternatively, could use a hash of run_id + stage_name for stable random sampling.
    # We will use canonical sorting for simplicity and reproducibility.
    subset = symbols[:max_assets]

    new_view = dict(stage_view)
    new_view["symbols"] = subset
    tprint(
        f"[{stage_view.get('stage_name', 'stage')}] Downscaled symbols from {len(symbols)} (planned) to {len(subset)} (effective) based on max_assets={max_assets}"
    )
    return new_view


def restrict_stage_period(stage_view: dict, max_months: Optional[int]) -> dict:
    """Deterministically restrict the time period of a stage view to the most recent max_months."""
    if max_months is None or max_months <= 0:
        return stage_view

    start_ts_str = stage_view.get("allowed_start_ts")
    end_ts_str = stage_view.get("allowed_end_ts")

    if not start_ts_str or not end_ts_str:
        return stage_view

    start_ts = pd.to_datetime(start_ts_str)
    end_ts = pd.to_datetime(end_ts_str)

    # Calculate the new start based on max_months from the end
    duration = pd.DateOffset(months=max_months)
    new_start_ts = max(start_ts, end_ts - duration)

    def _intersect_period(period: dict) -> dict | None:
        p_start = pd.to_datetime(
            period.get("start_ts") or period.get("start"), utc=True, errors="coerce"
        )
        p_end = pd.to_datetime(
            period.get("end_ts") or period.get("end"), utc=True, errors="coerce"
        )
        if pd.isna(p_start) or pd.isna(p_end):
            return None
        p_start = max(p_start, new_start_ts)
        p_end = min(p_end, end_ts)
        if p_end <= p_start:
            return None
        return {"start_ts": p_start.isoformat(), "end_ts": p_end.isoformat()}

    new_view = dict(stage_view)
    if stage_view.get("allowed_periods"):
        new_periods = []
        for period in stage_view["allowed_periods"]:
            clipped = _intersect_period(period)
            if clipped is not None:
                new_periods.append(clipped)
        new_view["allowed_periods"] = new_periods
    new_view["allowed_start_ts"] = new_start_ts.isoformat()
    tprint(
        f"[{stage_view.get('stage_name', 'stage')}] Downscaled period from {start_ts_str} to {new_start_ts.isoformat()} (effective start), end {end_ts_str} based on max_months={max_months}"
    )
    return new_view


def apply_stage_usage_limits(
    stage_view: dict, max_assets: Optional[int], max_months: Optional[int]
) -> dict:
    view = restrict_stage_symbols(stage_view, max_assets)
    view = restrict_stage_period(view, max_months)
    return view


from extreme_price_movements.data_store import load_features_selected


def load_features_for_stage(
    ts_sig: pd.Timestamp,
    root_dir: str,
    stage_view: dict,
    feature_keys: Optional[list] = None,
    lookback_pad: Optional[pd.Timedelta] = None,
) -> Optional[pd.DataFrame]:
    """Load features restricted to the stage's resolved symbols and time window."""
    symbols = stage_view.get("symbols")
    start_ts_str = stage_view.get("allowed_start_ts")
    end_ts_str = stage_view.get("allowed_end_ts")

    start_ts = pd.to_datetime(start_ts_str) if start_ts_str else None
    end_ts = pd.to_datetime(end_ts_str) if end_ts_str else None
    allowed_periods = stage_view.get("allowed_periods") or None

    if start_ts and lookback_pad:
        start_ts = start_ts - lookback_pad

    tprint(
        f"Loading features for stage {stage_view['stage_name']}: "
        f"{len(symbols) if symbols else 'ALL'} symbols, "
        f"from {start_ts} to {end_ts}"
    )

    return load_features_selected(
        ts=ts_sig,
        root_dir=root_dir,
        feature_keys=feature_keys,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        allowed_periods=allowed_periods,
    )
