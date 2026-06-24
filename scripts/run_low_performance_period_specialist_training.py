#!/usr/bin/env python3
"""Train/assess low-performance-period specialist candidates.

The script does three things:

1. Find low-prediction-performance timestamp windows from existing meta OOF rows.
2. Optimize the low-period threshold and optional archetype refinements.
3. Persist slice plans, exponential badness weights, and train_base/train_meta commands.

The actual model training still goes through ``extreme_price_movements/run_pipeline.py``
so feature selection, HPO/preset reuse, labels, and meta construction stay on the
normal train_base -> train_meta path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HEAD_PREFIXES = {
    "long_bars": "long_bars",
    "long_dist": "long_dist",
    "short_asset": "short_asset",
    "short_boll": "short_bollinger",
}


@dataclass(frozen=True)
class HeadSource:
    head: str
    strategy_id: str
    meta_oof_path: str
    archetype_scores_path: str | None


@dataclass(frozen=True)
class SourceRuns:
    artifact_run_id: str
    label_source_run_id: str
    feature_source_run_id: str
    native_preset_source_run_id: str


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)}")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_float_grid(raw: str) -> list[float]:
    vals: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    return vals


def _head_from_strategy(strategy_id: str) -> str | None:
    sid = str(strategy_id)
    for head, prefix in HEAD_PREFIXES.items():
        if sid.startswith(prefix):
            return head
    return None


def resolve_source_runs(data_root: Path, source_run_id: str) -> SourceRuns:
    manifest_path = data_root / "artifacts" / source_run_id / "final_model_fit_manifest.json"
    label_source = source_run_id
    feature_source = source_run_id
    native_source = source_run_id
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            label_source = str(payload.get("label_artifact_run_id") or payload.get("label_source_run_id") or label_source).strip()
            feature_source = str(payload.get("feature_run_id") or payload.get("feature_source_run_id") or feature_source).strip()
            native_source = str(payload.get("native_preset_source_run_id") or native_source).strip()
        except Exception:
            pass
    return SourceRuns(
        artifact_run_id=str(source_run_id),
        label_source_run_id=label_source or str(source_run_id),
        feature_source_run_id=feature_source or str(source_run_id),
        native_preset_source_run_id=native_source or str(source_run_id),
    )


def _strategy_from_meta_oof_path(path: Path) -> str:
    name = path.name
    if name.startswith("meta_oof_"):
        name = name[len("meta_oof_") :]
    if name.endswith(".parquet"):
        name = name[: -len(".parquet")]
    if name.endswith("_tbm_clf"):
        name = name[: -len("_tbm_clf")]
    return name


def discover_head_sources(source_root: Path, archetype_root: Path) -> dict[str, HeadSource]:
    out: dict[str, HeadSource] = {}
    meta_dir = source_root / "meta_oof"
    for path in sorted(meta_dir.glob("meta_oof_*_tbm_clf.parquet")):
        strategy_id = _strategy_from_meta_oof_path(path)
        head = _head_from_strategy(strategy_id)
        if head is None:
            continue
        arch_path = archetype_root / f"{head}_archetype_scores.parquet"
        out[head] = HeadSource(
            head=head,
            strategy_id=strategy_id,
            meta_oof_path=str(path),
            archetype_scores_path=str(arch_path) if arch_path.exists() else None,
        )
    return out


def _rank_pct_by_timestamp(ts: pd.Series, score: pd.Series) -> np.ndarray:
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts, utc=True, errors="coerce"), "score": pd.to_numeric(score, errors="coerce")})
    out = np.full(len(df), np.nan, dtype=np.float32)
    valid = df["timestamp"].notna() & df["score"].notna()
    if not bool(valid.any()):
        return out
    ranks = df.loc[valid].groupby("timestamp", sort=False)["score"].rank(method="average", pct=True)
    out[np.flatnonzero(valid.to_numpy())] = ranks.to_numpy(dtype=np.float32)
    return out


def load_head_oof(source: HeadSource) -> pd.DataFrame:
    cols = ["timestamp", "symbol", "y_bin", "oof_pred", "oof_p_move"]
    raw = pd.read_parquet(source.meta_oof_path)
    df = raw[[c for c in cols if c in raw.columns]].copy()
    if "oof_pred" not in df.columns and "oof_p_move" in df.columns:
        df["oof_pred"] = df["oof_p_move"]
    required = {"timestamp", "symbol", "y_bin", "oof_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{source.meta_oof_path} missing required columns: {missing}")
    df = df.dropna(subset=["timestamp", "symbol", "y_bin", "oof_pred"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["y_bin"] = pd.to_numeric(df["y_bin"], errors="coerce").astype(np.float32)
    df["oof_pred"] = pd.to_numeric(df["oof_pred"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["timestamp", "y_bin", "oof_pred"]).reset_index(drop=True)
    df["rank_timestamp"] = _rank_pct_by_timestamp(df["timestamp"], df["oof_pred"])
    df["head"] = source.head
    df["strategy_id"] = source.strategy_id
    return df


def _top_hr_for_group(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    n = len(y)
    if n <= 0:
        return float("nan")
    k = max(1, int(math.ceil(float(frac) * n)))
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="mergesort")
    idx = order[-k:]
    return float(np.mean(np.asarray(y, dtype=np.float32)[idx]))


def timestamp_performance(df: pd.DataFrame, min_rows_per_timestamp: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ts, g in df.groupby("timestamp", sort=True):
        n = int(len(g))
        if n < int(min_rows_per_timestamp):
            continue
        y = g["y_bin"].to_numpy(dtype=np.float32)
        score = g["oof_pred"].to_numpy(dtype=np.float32)
        hr10 = _top_hr_for_group(y, score, 0.10)
        hr20 = _top_hr_for_group(y, score, 0.20)
        hr30 = _top_hr_for_group(y, score, 0.30)
        top_hr = (hr10 + 0.33 * hr20 + 0.25 * hr30) / 1.58
        rows.append(
            {
                "timestamp": ts,
                "row_count": n,
                "hr10": hr10,
                "hr20": hr20,
                "hr30": hr30,
                "top_hr": top_hr,
                "base_rate": float(np.mean(y)),
            }
        )
    perf = pd.DataFrame(rows)
    if perf.empty:
        return perf
    perf = perf.sort_values("timestamp").reset_index(drop=True)
    rank = perf["top_hr"].rank(method="average", pct=True)
    perf["badness_score"] = (1.0 - rank).astype(np.float32)
    perf["badness_raw"] = (float(perf["top_hr"].mean()) - perf["top_hr"]).astype(np.float32)
    perf["week"] = perf["timestamp"].dt.to_period("W").astype(str)
    return perf


def _merge_timestamps_to_periods(
    selected_ts: pd.Series,
    *,
    pad_hours: float,
    merge_gap_hours: float,
    min_period_hours: float,
    max_periods: int,
) -> list[dict[str, str]]:
    ts = pd.to_datetime(selected_ts, utc=True, errors="coerce").dropna().sort_values().drop_duplicates()
    if ts.empty:
        return []
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = ts.iloc[0]
    prev = start
    gap = pd.Timedelta(hours=float(merge_gap_hours))
    for cur in ts.iloc[1:]:
        if cur - prev <= gap:
            prev = cur
            continue
        periods.append((start, prev + pd.Timedelta(hours=1)))
        start = cur
        prev = cur
    periods.append((start, prev + pd.Timedelta(hours=1)))

    pad = pd.Timedelta(hours=float(pad_hours))
    min_dur = pd.Timedelta(hours=float(min_period_hours))
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in periods:
        start = start - pad
        end = end + pad
        if end - start < min_dur:
            mid = start + (end - start) / 2
            start = mid - min_dur / 2
            end = mid + min_dur / 2
        out.append((start, end))
    out = sorted(out, key=lambda x: x[0])
    if max_periods > 0 and len(out) > max_periods:
        out = out[:max_periods]
    return [{"start_ts": s.isoformat(), "end_ts": e.isoformat()} for s, e in out]


def _period_mask(timestamps: pd.Series, periods: list[dict[str, str]]) -> np.ndarray:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    out = np.zeros(len(ts), dtype=bool)
    for period in periods:
        start = pd.to_datetime(period.get("start_ts"), utc=True, errors="coerce")
        end = pd.to_datetime(period.get("end_ts"), utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        out |= ((ts >= start) & (ts < end)).to_numpy(dtype=bool)
    return out


def threshold_trials(
    perf: pd.DataFrame,
    *,
    threshold_grid: list[float],
    lambda_grid: list[float],
    min_timestamps: int,
    min_rows: int,
    pad_hours: float,
    merge_gap_hours: float,
    min_period_hours: float,
    max_periods: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if perf.empty:
        return pd.DataFrame(rows)
    global_top_hr = float(perf["top_hr"].mean())
    total_rows = int(perf["row_count"].sum())
    for share in threshold_grid:
        cutoff = float(perf["top_hr"].quantile(float(share)))
        selected = perf[perf["top_hr"] <= cutoff].copy()
        periods = _merge_timestamps_to_periods(
            selected["timestamp"],
            pad_hours=pad_hours,
            merge_gap_hours=merge_gap_hours,
            min_period_hours=min_period_hours,
            max_periods=max_periods,
        )
        in_period = _period_mask(perf["timestamp"], periods)
        selected_period = perf.loc[in_period].copy()
        selected_rows = int(selected_period["row_count"].sum()) if not selected_period.empty else 0
        selected_ts = int(len(selected_period))
        support = float(selected_rows / max(total_rows, 1))
        selected_top_hr = float(selected_period["top_hr"].mean()) if selected_ts else float("nan")
        difficulty_lift = float(global_top_hr - selected_top_hr) if np.isfinite(selected_top_hr) else float("nan")
        for lam in lambda_grid:
            if selected_period.empty:
                ess_frac = float("nan")
            else:
                bad = selected_period["badness_score"].to_numpy(dtype=np.float64)
                scale = float(np.nanpercentile(bad, 95.0)) if len(bad) else 1.0
                bad_norm = np.clip(bad / max(scale, 1e-6), 0.0, 1.0)
                w = np.exp(float(lam) * bad_norm)
                ess_frac = float((w.sum() ** 2) / max(float(np.sum(w**2)), 1e-12) / max(len(w), 1))
            train_candidate = bool(selected_ts >= min_timestamps and selected_rows >= min_rows)
            broad_penalty = max(0.0, support - 0.35) * 0.25
            score = (
                (difficulty_lift if np.isfinite(difficulty_lift) else -1.0)
                * math.sqrt(max(support, 1e-6))
                * math.sqrt(max(min(len(periods), 12), 1))
                + 0.05 * (ess_frac if np.isfinite(ess_frac) else 0.0)
                - broad_penalty
            )
            rows.append(
                {
                    "threshold_bottom_share": float(share),
                    "top_hr_cutoff": cutoff,
                    "lambda_badness": float(lam),
                    "period_count": int(len(periods)),
                    "selected_timestamps": selected_ts,
                    "selected_rows": selected_rows,
                    "selected_row_share": support,
                    "global_top_hr": global_top_hr,
                    "selected_top_hr": selected_top_hr,
                    "difficulty_lift": difficulty_lift,
                    "selected_weeks": int(selected_period["week"].nunique()) if selected_ts else 0,
                    "weight_ess_fraction_est": ess_frac,
                    "train_candidate": train_candidate,
                    "selection_score": score,
                }
            )
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)


def archetype_trials(
    source: HeadSource,
    df: pd.DataFrame,
    perf: pd.DataFrame,
    base_periods: list[dict[str, str]],
    *,
    min_timestamps: int,
    min_rows: int,
    top_n: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    if not source.archetype_scores_path:
        return pd.DataFrame(), []
    arch = pd.read_parquet(source.archetype_scores_path)
    if len(arch) != len(df):
        return pd.DataFrame(
            [
                {
                    "archetype_mode": "unavailable",
                    "reason": "row_count_mismatch",
                    "archetype_rows": int(len(arch)),
                    "meta_rows": int(len(df)),
                }
            ]
        ), []
    score_cols = [
        c
        for c in arch.columns
        if c.endswith("_score")
        and "support" not in c
        and pd.api.types.is_numeric_dtype(arch[c])
    ]
    if not score_cols:
        return pd.DataFrame(), []
    tmp = df[["timestamp"]].copy()
    for col in score_cols:
        tmp[col] = pd.to_numeric(arch[col], errors="coerce").astype(np.float32)
    agg = tmp.groupby("timestamp", sort=False)[score_cols].mean().reset_index()
    joined = perf.merge(agg, on="timestamp", how="left")
    base_mask = _period_mask(joined["timestamp"], base_periods)
    base_label = base_mask.astype(np.int8)
    ranked: list[dict[str, Any]] = []
    for col in score_cols:
        vals = pd.to_numeric(joined[col], errors="coerce")
        mask = vals.notna() & np.isfinite(joined["badness_score"].to_numpy(dtype=np.float64))
        if int(mask.sum()) < 30 or int(base_label[mask].sum()) < 5:
            continue
        corr = float(pd.Series(vals[mask]).corr(joined.loc[mask, "badness_score"], method="spearman"))
        high = vals >= float(vals.quantile(0.75))
        selected_bad = float(joined.loc[high, "badness_score"].mean()) if bool(high.any()) else float("nan")
        base_bad = float(joined["badness_score"].mean())
        score = abs(corr if np.isfinite(corr) else 0.0) + max(0.0, selected_bad - base_bad)
        ranked.append({"feature": col, "spearman_badness": corr, "top_quartile_badness_lift": selected_bad - base_bad, "archetype_score": score})
    ranked_df = pd.DataFrame(ranked).sort_values("archetype_score", ascending=False)
    top_features = ranked_df["feature"].head(int(top_n)).astype(str).tolist() if not ranked_df.empty else []
    if not top_features:
        return ranked_df, []

    rows: list[dict[str, Any]] = []
    total_rows = int(perf["row_count"].sum())
    global_top_hr = float(perf["top_hr"].mean())
    modes = {"none": np.ones(len(joined), dtype=bool)}
    modes["top1_high"] = pd.to_numeric(joined[top_features[0]], errors="coerce").ge(float(joined[top_features[0]].quantile(0.60))).to_numpy(dtype=bool)
    if len(top_features) > 1:
        combo = np.zeros(len(joined), dtype=bool)
        for col in top_features:
            combo |= pd.to_numeric(joined[col], errors="coerce").ge(float(joined[col].quantile(0.65))).to_numpy(dtype=bool)
        modes["top3_union_high"] = combo
    for mode, extra_mask in modes.items():
        final_mask = base_mask & extra_mask
        selected = joined.loc[final_mask].copy()
        selected_rows = int(selected["row_count"].sum()) if not selected.empty else 0
        selected_ts = int(len(selected))
        selected_top_hr = float(selected["top_hr"].mean()) if selected_ts else float("nan")
        support = float(selected_rows / max(total_rows, 1))
        difficulty_lift = float(global_top_hr - selected_top_hr) if np.isfinite(selected_top_hr) else float("nan")
        train_candidate = bool(selected_ts >= min_timestamps and selected_rows >= min_rows)
        score = (difficulty_lift if np.isfinite(difficulty_lift) else -1.0) * math.sqrt(max(support, 1e-6))
        rows.append(
            {
                "archetype_mode": mode,
                "top_archetype_features": "|".join(top_features),
                "selected_timestamps": selected_ts,
                "selected_rows": selected_rows,
                "selected_row_share": support,
                "global_top_hr": global_top_hr,
                "selected_top_hr": selected_top_hr,
                "difficulty_lift": difficulty_lift,
                "train_candidate": train_candidate,
                "selection_score": score,
                "alignment": "row_order_to_meta_oof",
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False), top_features


def _make_period_weights(
    source: HeadSource,
    perf: pd.DataFrame,
    periods: list[dict[str, str]],
    lambda_badness: float,
    *,
    max_multiplier: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in periods:
        mask = _period_mask(perf["timestamp"], [period])
        sub = perf.loc[mask]
        if sub.empty:
            continue
        bad = sub["badness_score"].to_numpy(dtype=np.float64)
        badness = float(np.nanmean(bad))
        multiplier = float(np.exp(float(lambda_badness) * np.clip(badness, 0.0, 1.0)))
        multiplier = float(np.clip(multiplier, 1.0, float(max_multiplier)))
        rows.append(
            {
                "head": source.head,
                "strategy_id": source.strategy_id,
                "start_ts": period["start_ts"],
                "end_ts": period["end_ts"],
                "badness": badness,
                "top_hr_mean": float(sub["top_hr"].mean()),
                "row_count": int(sub["row_count"].sum()),
                "timestamp_count": int(len(sub)),
                "weight_multiplier": multiplier,
            }
        )
    return pd.DataFrame(rows)


def _write_slice_plan(path: Path, source: HeadSource, periods: list[dict[str, str]]) -> None:
    starts = [pd.to_datetime(p["start_ts"], utc=True, errors="coerce") for p in periods]
    ends = [pd.to_datetime(p["end_ts"], utc=True, errors="coerce") for p in periods]
    starts = [x for x in starts if not pd.isna(x)]
    ends = [x for x in ends if not pd.isna(x)]
    view = {
        "stage_name": "low_performance_period_specialist",
        "source_roles": ["low_performance_oof_periods"],
        "symbols": [],
        "allowed_symbols": [],
        "allowed_periods": periods,
        "allowed_start_ts": min(starts).isoformat() if starts else None,
        "allowed_end_ts": max(ends).isoformat() if ends else None,
        "head": source.head,
        "strategy_id": source.strategy_id,
        "disable_exact_plan_row_filter": False,
    }
    payload = {
        "version": 1,
        "generated_by": Path(__file__).name,
        "materialized_views": {
            "train_base": {**view, "stage_name": "train_base"},
            "train_meta": {**view, "stage_name": "train_meta"},
        },
        "metadata": {
            "purpose": "low_performance_period_specialist_training",
            "head": source.head,
            "strategy_id": source.strategy_id,
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _shell_env(env: dict[str, str]) -> str:
    return " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(env.items()))


def _build_commands(
    *,
    source: HeadSource,
    run_id: str,
    source_runs: SourceRuns,
    data_root: str,
    slice_plan_path: Path,
    weights_path: Path,
    log_dir: Path,
) -> dict[str, Any]:
    common_env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": ".",
        "MPLCONFIGDIR": "/tmp/mplconfig_epm",
        "EPM_DATA_ROOT": data_root,
        "EPM_ARTIFACT_SOURCE_RUN_ID": source_runs.artifact_run_id,
        "EPM_LABEL_SOURCE_RUN_ID": source_runs.label_source_run_id,
        "EPM_LABEL_ARTIFACT_RUN_ID": source_runs.label_source_run_id,
        "EPM_FEATURE_SOURCE_RUN_ID": source_runs.feature_source_run_id,
        "EPM_STRATEGY_SOURCE_RUN_ID": source_runs.artifact_run_id,
        "EPM_POLICY_ARTIFACT_RUN_ID": source_runs.artifact_run_id,
        "EPM_ALLOW_CONTRACT_STRATEGY_WITHOUT_POLICY_MASK": "1",
        "EPM_SKIP_MASK_STRATEGY_PARAMS": "1",
        "EPM_LGBM_USE_NATIVE_PRESET": "1",
        "EPM_LGBM_REQUIRE_NATIVE_PRESET": "1",
        "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": source_runs.native_preset_source_run_id,
        "EPM_TRAIN_SLICE_PLAN_PATH": str(slice_plan_path),
        "EPM_TRAIN_EXTEND_TO_LATEST": "0",
        "EPM_LOW_PERFORMANCE_PERIOD_WEIGHTS_ENABLED": "1",
        "EPM_LOW_PERFORMANCE_PERIOD_WEIGHTS_PATH": str(weights_path),
        "EPM_LOW_PERFORMANCE_PERIOD_HEAD": source.head,
        "EPM_LOW_PERFORMANCE_PERIOD_STRATEGY_ID": source.strategy_id,
        "EPM_REQUIRE_STRATEGY_ALLOWLIST": "1",
        "EPM_LABEL_STRATEGY_IDS": source.strategy_id,
        "EPM_BASE_STRATEGY_IDS": source.strategy_id,
        "EPM_META_STRATEGY_IDS": source.strategy_id,
        "EPM_POLICY_STRATEGY_IDS": source.strategy_id,
        "EPM_MODEL_BACKEND": "lgbm_pipeline",
        "EPM_LGBM_FINAL_MODEL_COUNT": "1",
        "EPM_LGBM_HPO_TRIALS": "0",
        "EPM_LGBM_OOF_DISTILLATION_PASSES": "0",
        "EPM_LGBM_META_MIN_OOF_DISTILLATION_PASSES": "0",
        "EPM_LGBM_MIN_OOF_DISTILLATION_PASSES": "0",
        "EPM_META_REQUIRE_DISTILLED_BASE_OOF": "0",
        "EPM_META_MIN_BASE_OOF_DISTILLATION_PASSES": "0",
        "EPM_META_ALLOW_NEUTRAL_MODEL_DERIVED_PLACEHOLDERS": "1",
    }
    base_log = log_dir / f"{run_id}_{source.head}_train_base.log"
    meta_log = log_dir / f"{run_id}_{source.head}_train_meta.log"
    base_cmd = (
        f"{_shell_env(common_env)} python3 -u extreme_price_movements/run_pipeline.py "
        f"train_base --market-mode perps --model-backend lgbm_pipeline --run-id {shlex.quote(run_id)} "
        f"--ts {shlex.quote(source_runs.feature_source_run_id)} > {shlex.quote(str(base_log))} 2>&1"
    )
    meta_cmd = (
        f"{_shell_env(common_env)} python3 -u extreme_price_movements/run_pipeline.py "
        f"train_meta --market-mode perps --model-backend lgbm_pipeline --run-id {shlex.quote(run_id)} "
        f"--ts {shlex.quote(source_runs.feature_source_run_id)} > {shlex.quote(str(meta_log))} 2>&1"
    )
    return {
        "run_id": run_id,
        "head": source.head,
        "strategy_id": source.strategy_id,
        "env": common_env,
        "source_runs": asdict(source_runs),
        "train_base_command": base_cmd,
        "train_meta_command": meta_cmd,
        "train_base_log": str(base_log),
        "train_meta_log": str(meta_log),
    }


def _launch_command(cmd: str, cwd: Path) -> int:
    proc = subprocess.Popen(["/bin/zsh", "-lc", cmd], cwd=str(cwd))
    return int(proc.pid)


def _launch_sequential_script(script_path: Path, cwd: Path, log_path: Path) -> int:
    cmd = f"nohup /bin/zsh {shlex.quote(str(script_path))} > {shlex.quote(str(log_path))} 2>&1"
    proc = subprocess.Popen(["/bin/zsh", "-lc", cmd], cwd=str(cwd))
    return int(proc.pid)


def run(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root)
    source_runs = resolve_source_runs(data_root, args.source_run_id)
    source_root = data_root / "artifacts" / args.source_run_id
    archetype_root = Path(args.archetype_root)
    out_dir = _ensure_dir(Path(args.output_dir))
    log_dir = _ensure_dir(Path("logs"))
    sources = discover_head_sources(source_root, archetype_root)
    wanted = set(args.only_head or sorted(sources))
    sources = {h: s for h, s in sources.items() if h in wanted}
    if not sources:
        raise SystemExit(f"No head sources found for {sorted(wanted)} under {source_root}")

    threshold_grid = _parse_float_grid(args.threshold_grid)
    lambda_grid = _parse_float_grid(args.lambda_grid)
    all_threshold_trials: list[pd.DataFrame] = []
    all_archetype_trials: list[pd.DataFrame] = []
    all_selected_periods: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    commands: list[dict[str, Any]] = []
    launch_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for head, source in sources.items():
        df = load_head_oof(source)
        perf = timestamp_performance(df, int(args.min_rows_per_timestamp))
        perf.to_parquet(out_dir / f"{head}_timestamp_performance.parquet", index=False)
        trials = threshold_trials(
            perf,
            threshold_grid=threshold_grid,
            lambda_grid=lambda_grid,
            min_timestamps=int(args.min_timestamps),
            min_rows=int(args.min_rows),
            pad_hours=float(args.pad_hours),
            merge_gap_hours=float(args.merge_gap_hours),
            min_period_hours=float(args.min_period_hours),
            max_periods=int(args.max_periods),
        )
        trials.insert(0, "head", head)
        trials.insert(1, "strategy_id", source.strategy_id)
        all_threshold_trials.append(trials)
        best = trials.iloc[0].to_dict() if not trials.empty else {}
        if not bool(best.get("train_candidate", False)):
            summary_rows.append(
                {
                    "head": head,
                    "strategy_id": source.strategy_id,
                    "train_candidate": False,
                    "reason": "insufficient_low_period_support",
                    **{k: best.get(k) for k in best.keys()},
                }
            )
            continue
        cutoff = float(best["top_hr_cutoff"])
        selected_base = perf[perf["top_hr"] <= cutoff].copy()
        periods = _merge_timestamps_to_periods(
            selected_base["timestamp"],
            pad_hours=float(args.pad_hours),
            merge_gap_hours=float(args.merge_gap_hours),
            min_period_hours=float(args.min_period_hours),
            max_periods=int(args.max_periods),
        )
        arch_trials, top_arch = archetype_trials(
            source,
            df,
            perf,
            periods,
            min_timestamps=int(args.min_timestamps),
            min_rows=int(args.min_rows),
            top_n=int(args.archetype_top_n),
        )
        if not arch_trials.empty:
            arch_trials.insert(0, "head", head)
            arch_trials.insert(1, "strategy_id", source.strategy_id)
            all_archetype_trials.append(arch_trials)
            arch_best = arch_trials.iloc[0].to_dict()
            if (
                bool(arch_best.get("train_candidate", False))
                and str(arch_best.get("archetype_mode")) not in {"none", "unavailable"}
                and float(arch_best.get("selection_score", -np.inf)) > float((arch_trials[arch_trials["archetype_mode"] == "none"]["selection_score"].max() if "none" in set(arch_trials["archetype_mode"].astype(str)) else -np.inf))
            ):
                joined = perf.copy()
                arch = pd.read_parquet(source.archetype_scores_path) if source.archetype_scores_path else pd.DataFrame()
                if len(arch) == len(df) and top_arch:
                    tmp = df[["timestamp"]].copy()
                    for col in top_arch:
                        tmp[col] = pd.to_numeric(arch[col], errors="coerce").astype(np.float32)
                    agg = tmp.groupby("timestamp", sort=False)[top_arch].mean().reset_index()
                    joined = joined.merge(agg, on="timestamp", how="left")
                    base_mask = _period_mask(joined["timestamp"], periods)
                    if str(arch_best.get("archetype_mode")) == "top1_high":
                        col = top_arch[0]
                        extra = pd.to_numeric(joined[col], errors="coerce").ge(float(joined[col].quantile(0.60))).to_numpy(dtype=bool)
                    else:
                        extra = np.zeros(len(joined), dtype=bool)
                        for col in top_arch:
                            extra |= pd.to_numeric(joined[col], errors="coerce").ge(float(joined[col].quantile(0.65))).to_numpy(dtype=bool)
                    periods = _merge_timestamps_to_periods(
                        joined.loc[base_mask & extra, "timestamp"],
                        pad_hours=float(args.pad_hours),
                        merge_gap_hours=float(args.merge_gap_hours),
                        min_period_hours=float(args.min_period_hours),
                        max_periods=int(args.max_periods),
                    )
        weights = _make_period_weights(
            source,
            perf,
            periods,
            float(best["lambda_badness"]),
            max_multiplier=float(args.max_weight_multiplier),
        )
        if weights.empty:
            summary_rows.append(
                {
                    "head": head,
                    "strategy_id": source.strategy_id,
                    "train_candidate": False,
                    "reason": "empty_period_weights",
                }
            )
            continue
        selected_periods = weights.copy()
        selected_periods["threshold_bottom_share"] = float(best["threshold_bottom_share"])
        selected_periods["lambda_badness"] = float(best["lambda_badness"])
        all_selected_periods.append(selected_periods)
        all_weights.append(weights)
        head_weights_path = out_dir / f"{head}_low_period_weights.parquet"
        head_slice_path = out_dir / f"{head}_slice_plan.json"
        weights.to_parquet(head_weights_path, index=False)
        _write_slice_plan(head_slice_path, source, periods)
        run_id = f"{args.run_id_prefix}_{head}_{time.strftime('%Y%m%d_%H%M%S')}"
        cmd_pack = _build_commands(
            source=source,
            run_id=run_id,
            source_runs=source_runs,
            data_root=args.data_root,
            slice_plan_path=head_slice_path,
            weights_path=head_weights_path,
            log_dir=log_dir,
        )
        commands.append(cmd_pack)
        summary_rows.append(
            {
                "head": head,
                "strategy_id": source.strategy_id,
                "run_id": run_id,
                "train_candidate": True,
                "period_count": int(len(periods)),
                "period_weight_rows": int(len(weights)),
                "selected_rows": int(weights["row_count"].sum()),
                "selected_timestamps": int(weights["timestamp_count"].sum()),
                "mean_period_badness": float(weights["badness"].mean()),
                "mean_weight_multiplier": float(weights["weight_multiplier"].mean()),
                "max_weight_multiplier": float(weights["weight_multiplier"].max()),
                "threshold_bottom_share": float(best["threshold_bottom_share"]),
                "lambda_badness": float(best["lambda_badness"]),
                "slice_plan_path": str(head_slice_path),
                "weights_path": str(head_weights_path),
                "top_archetype_features": "|".join(top_arch),
                "artifact_source_run_id": source_runs.artifact_run_id,
                "label_source_run_id": source_runs.label_source_run_id,
                "feature_source_run_id": source_runs.feature_source_run_id,
                "native_preset_source_run_id": source_runs.native_preset_source_run_id,
            }
        )
        if args.launch:
            base_pid = _launch_command(cmd_pack["train_base_command"], Path.cwd())
            launch_records.append({"head": head, "stage": "train_base", "pid": base_pid, "log": cmd_pack["train_base_log"]})
            if args.launch_meta:
                meta_pid = _launch_command(cmd_pack["train_meta_command"], Path.cwd())
                launch_records.append({"head": head, "stage": "train_meta", "pid": meta_pid, "log": cmd_pack["train_meta_log"]})

    if all_threshold_trials:
        pd.concat(all_threshold_trials, ignore_index=True).to_csv(out_dir / "low_period_threshold_trials.csv", index=False)
    if all_archetype_trials:
        pd.concat(all_archetype_trials, ignore_index=True).to_csv(out_dir / "low_period_archetype_trials.csv", index=False)
    if all_selected_periods:
        pd.concat(all_selected_periods, ignore_index=True).to_csv(out_dir / "low_period_selected_periods.csv", index=False)
    if all_weights:
        pd.concat(all_weights, ignore_index=True).to_parquet(out_dir / "low_period_weights_all_heads.parquet", index=False)

    pd.DataFrame(summary_rows).to_csv(out_dir / "low_period_specialist_summary.csv", index=False)
    (out_dir / "train_commands.json").write_text(json.dumps(commands, indent=2, default=_json_default), encoding="utf-8")
    with (out_dir / "train_commands.sh").open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env zsh\nset -euo pipefail\n\n")
        for pack in commands:
            f.write(f"# {pack['head']} train_base\n{pack['train_base_command']}\n")
            f.write(f"# {pack['head']} train_meta\n{pack['train_meta_command']}\n\n")
    if launch_records:
        pd.DataFrame(launch_records).to_csv(out_dir / "launched_jobs.csv", index=False)

    if args.launch_sequential and commands:
        master_log = log_dir / f"{args.run_id_prefix}_sequential_{time.strftime('%Y%m%d_%H%M%S')}.log"
        pid = _launch_sequential_script(out_dir / "train_commands.sh", Path.cwd(), master_log)
        seq_record = {
            "stage": "sequential_train_base_then_meta",
            "pid": pid,
            "log": str(master_log),
            "command_script": str(out_dir / "train_commands.sh"),
        }
        pd.DataFrame([seq_record]).to_csv(out_dir / "launched_sequential_job.csv", index=False)

    lines = [
        "# Low-Performance Period Specialist Training",
        "",
        f"Source run: `{args.source_run_id}`",
        f"Label source run: `{source_runs.label_source_run_id}`",
        f"Feature source run: `{source_runs.feature_source_run_id}`",
        f"Native preset source run: `{source_runs.native_preset_source_run_id}`",
        f"Output dir: `{out_dir}`",
        "",
        "## Summary",
        "",
    ]
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        display_cols = [
            "head",
            "train_candidate",
            "period_count",
            "selected_rows",
            "selected_timestamps",
            "threshold_bottom_share",
            "lambda_badness",
            "mean_period_badness",
            "mean_weight_multiplier",
            "max_weight_multiplier",
            "top_archetype_features",
        ]
        lines.append(summary_df[[c for c in display_cols if c in summary_df.columns]].to_markdown(index=False))
    if launch_records:
        lines += ["", "## Launched Jobs", "", pd.DataFrame(launch_records).to_markdown(index=False)]
    elif args.launch_sequential and commands:
        lines += [
            "",
            "## Launched Sequential Job",
            "",
            pd.DataFrame([seq_record]).to_markdown(index=False),
        ]
    else:
        lines += ["", "Training was not launched. Run `train_commands.sh` or rerun with `--launch`."]
    (out_dir / "low_period_specialist_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--source-run-id", default="20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument(
        "--archetype-root",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1",
    )
    parser.add_argument("--output-dir", default=f"data_perp/reports/low_performance_period_specialist_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--only-head", nargs="*", default=None)
    parser.add_argument("--threshold-grid", default="0.05,0.075,0.10,0.15,0.20")
    parser.add_argument("--lambda-grid", default="1.0,2.0,3.0")
    parser.add_argument("--min-rows-per-timestamp", type=int, default=2)
    parser.add_argument("--min-timestamps", type=int, default=24)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--pad-hours", type=float, default=6.0)
    parser.add_argument("--merge-gap-hours", type=float, default=12.0)
    parser.add_argument("--min-period-hours", type=float, default=6.0)
    parser.add_argument("--max-periods", type=int, default=80)
    parser.add_argument("--max-weight-multiplier", type=float, default=8.0)
    parser.add_argument("--archetype-top-n", type=int, default=3)
    parser.add_argument("--run-id-prefix", default="low_perf_specialist")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--launch-meta", action="store_true", help="Launch train_meta immediately too. By default only train_base is launched.")
    parser.add_argument("--launch-sequential", action="store_true", help="Launch the generated train_commands.sh as one sequential background job.")
    return parser.parse_args()


def main() -> None:
    out_dir = run(parse_args())
    print(out_dir)


if __name__ == "__main__":
    main()
