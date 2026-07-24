#!/usr/bin/env python3
"""Ablate base-layer AE/GMM archetype input features with pre-train anchors.

This driver keeps the downstream base model feature contract and fixed LGBM
params unchanged.  Only the feature list used to fit the AE/GMM state changes.
Each arm runs the materialized trailing-label base scorer with:

* 150-day train window
* AE/GMM state fitted on the 30 days immediately before that train window
* OOS scoring on April, May, and June 2026
* 1% cost already embedded in the S59 cost100bps materialized labels
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_mpl_cache = ROOT / ".cache" / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - pyarrow is expected in this repo.
    pq = None

from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import FUTURE_OR_LABEL_COLUMNS  # noqa: E402
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import _load_fixed_selected_features  # noqa: E402


DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/"
    "20260708_s59_h5_2025start_monthly_v6_15mchart_trailing_cost100bps_labels/labels"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260711_070000")
DEFAULT_FEATURE_LIST_CSV = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_config_meta_full_feature_list.csv"
)
DEFAULT_FIXED_PARAMS_JSON = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/topk_lgbm_hpo_best.json"
)
DEFAULT_FIXED_SELECTED_FEATURES_CSV = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/topk_lgbm_feature_selection_by_fold.csv"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/base_archetype_anchor_location_ablation_20260709")

TOP_FRACS = (0.10, 0.20, 0.30)
AE_GMM_GENERATED = {str(c) for c in AE_GMM_FEATURE_COLUMNS}


@dataclass(frozen=True)
class ArmResult:
    arm: str
    groups: tuple[str, ...]
    output_dir: Path
    ledger_path: Path
    primary_top10_net: float
    metrics: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _read_csv_feature_list(path: Path) -> list[str]:
    if path is None or not path.exists():
        return []
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        return []
    return [str(v) for v in frame["feature"].dropna().tolist() if str(v).strip()]


def _label_schema_columns(labels_path: Path) -> list[str]:
    if pq is None:
        return []
    files = sorted(Path(labels_path).glob("*.parquet")) if Path(labels_path).is_dir() else [Path(labels_path)]
    cols: list[str] = []
    for path in files[:8]:
        try:
            cols.extend(str(c) for c in pq.read_schema(path).names)
        except Exception:
            continue
    return list(dict.fromkeys(cols))


def _is_candidate_feature(name: str) -> bool:
    if name in FUTURE_OR_LABEL_COLUMNS:
        return False
    if name in AE_GMM_GENERATED:
        return False
    if name.startswith("__") and not (name.startswith("__regime_") or name.startswith("__meta_raw__")):
        return False
    return True


def _available_features(labels_path: Path, feature_list_csv: Path) -> list[str]:
    values = [*_label_schema_columns(labels_path), *_read_csv_feature_list(feature_list_csv)]
    return [name for name in dict.fromkeys(values) if _is_candidate_feature(str(name))]


def _write_feature_csv(path: Path, features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": list(dict.fromkeys(features))}).to_csv(path, index=False)


def _contains_any(name: str, tokens: tuple[str, ...]) -> bool:
    low = str(name).lower()
    return any(token in low for token in tokens)


def _build_groups(available: list[str], fixed_selected: list[str]) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    available_set = set(available)
    selected_raw = [
        str(feature)
        for feature in fixed_selected
        if str(feature) in available_set and str(feature) not in AE_GMM_GENERATED
    ]
    selected_raw = list(dict.fromkeys(selected_raw))

    momentum_tokens = (
        "lr_",
        "ret",
        "return",
        "trend",
        "mom",
        "adx",
        "impulse",
        "breakout",
        "z_r",
        "zr_",
        "convexity",
        "slope",
        "velocity",
        "speed",
        "thrust",
    )
    normalized_tokens = (
        "atr",
        "vol_norm",
        "_z",
        "z_",
        "cp_z",
        "ts_resid",
        "ratio",
        "rank",
        "pct",
        "tanh",
        "bps",
        "rsi",
        "autocorr",
    )
    raw_momentum = [
        f
        for f in selected_raw
        if _contains_any(f, momentum_tokens) and not _contains_any(f, normalized_tokens)
    ]
    normalized_momentum = [
        f
        for f in available
        if _contains_any(f, momentum_tokens) and _contains_any(f, normalized_tokens)
    ]
    a0bis = [
        f
        for f in selected_raw
        if f not in set(raw_momentum)
    ] + normalized_momentum

    groups = {
        "A1_location_vwap_prior_range": [
            f
            for f in available
            if _contains_any(
                f,
                (
                    "vwap",
                    "prior_day",
                    "prev_day",
                    "prev_week",
                    "range_pos",
                    "loc_",
                    "dist_rolling_7d",
                    "rolling_7d_high",
                    "rolling_7d_low",
                    "range_pct",
                    "donchian",
                ),
            )
        ],
        "A2_bb_compression_squeeze": [
            f
            for f in available
            if _contains_any(
                f,
                (
                    "bb",
                    "boll",
                    "keltner",
                    "compression",
                    "squeeze",
                    "chop",
                    "channel",
                    "rv_ratio",
                    "vol_compression",
                    "atr_compression",
                    "comp_to_exp",
                ),
            )
        ],
        "A3_support_resistance_barrier_pressure": [
            f
            for f in available
            if _contains_any(
                f,
                (
                    "support",
                    "resistance",
                    "barrier",
                    "pressure",
                    "swing",
                    "retest",
                    "breakout_confirmed",
                    "breakout_min",
                    "up_barrier",
                    "down_barrier",
                ),
            )
        ],
        "A4_move_speed_recency": [
            f
            for f in available
            if _contains_any(
                f,
                (
                    "move_speed_1h_atr",
                    "move_speed_3h_atr",
                    "bars_since_last_1r_move",
                    "bars_since_last_2r_move",
                    "log_bars_since_above_1atr",
                    "log_bars_since_above_2atr",
                    "speed",
                    "dip_velocity",
                    "decel",
                    "thrust_decay",
                ),
            )
        ],
    }
    diagnostics = {
        "selected_raw_count": int(len(selected_raw)),
        "a0bis_removed_raw_momentum_count": int(len(raw_momentum)),
        "a0bis_added_normalized_momentum_count": int(len(set(normalized_momentum).difference(selected_raw))),
        "a0bis_removed_raw_momentum": raw_momentum,
        "a0bis_added_normalized_momentum": sorted(set(normalized_momentum).difference(selected_raw)),
        "group_counts": {name: len(values) for name, values in groups.items()},
    }
    return selected_raw, {"A0bis": list(dict.fromkeys(a0bis)), **groups}, diagnostics


def _net_series(frame: pd.DataFrame) -> pd.Series:
    for col in (
        "__first_touch_capture_net__",
        "first_touch_net",
        "__u_policy_net__",
        "__r_policy_net__",
    ):
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    if "__first_touch_capture_gross__" in frame.columns:
        return pd.to_numeric(frame["__first_touch_capture_gross__"], errors="coerce").fillna(0.0) - 0.01
    if "__y_ret__" in frame.columns:
        return pd.to_numeric(frame["__y_ret__"], errors="coerce").fillna(0.0) - 0.01
    return pd.Series(0.0, index=frame.index, dtype=np.float32)


def _summarize_ledger(ledger_path: Path, *, arm: str, groups: tuple[str, ...]) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_parquet(ledger_path)
    if "__ts__" not in frame.columns:
        raise ValueError(f"{ledger_path} has no __ts__ column")
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.assign(__ts__=ts, net_ev=_net_series(frame))
    frame["week_start"] = frame["__ts__"].dt.to_period("W-SUN").dt.start_time.astype(str)
    row: dict[str, Any] = {
        "arm": arm,
        "groups": "+".join(groups) if groups else "",
        "rows": int(len(frame)),
        "days": int(ts.dt.date.nunique()),
    }
    weekly_rows: list[dict[str, Any]] = []
    top10_week_means: list[float] = []
    for frac in TOP_FRACS:
        tag = f"top{int(round(frac * 100))}"
        selected_col = f"selected_{tag}"
        if selected_col not in frame.columns:
            threshold = frame["score"].rank(method="first", ascending=False, pct=True).le(frac)
            selected = frame.loc[threshold]
        else:
            selected = frame.loc[frame[selected_col].astype(bool)]
        net = selected["net_ev"]
        row[f"{tag}_selected_rows"] = int(len(selected))
        row[f"{tag}_avg_net_return_per_trade"] = float(net.mean()) if len(net) else float("nan")
        row[f"{tag}_net_pnl"] = float(net.sum()) if len(net) else 0.0
        row[f"{tag}_trades_per_day"] = float(len(selected) / max(row["days"], 1))
        if frac == 0.10:
            week_mean = selected.groupby(selected["__ts__"].dt.to_period("W-SUN"))["net_ev"].mean()
            top10_week_means = [float(v) for v in week_mean.dropna().tolist()]
        for week, part in selected.groupby("week_start", dropna=False):
            weekly_rows.append(
                {
                    "arm": arm,
                    "groups": "+".join(groups) if groups else "",
                    "week_start": str(week),
                    "top_frac": float(frac),
                    "selected_rows": int(len(part)),
                    "avg_net_return_per_trade": float(part["net_ev"].mean()) if len(part) else float("nan"),
                    "net_pnl": float(part["net_ev"].sum()) if len(part) else 0.0,
                }
            )
    row["q10_week_ev_top10"] = float(np.nanquantile(top10_week_means, 0.10)) if top10_week_means else float("nan")
    return row, pd.DataFrame(weekly_rows)


def _rss_mb(pid: int) -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(int(pid)).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            text=True,
            capture_output=True,
        )
        text = result.stdout.strip()
        if not text:
            return None
        return float(text.splitlines()[-1].strip()) / 1024.0
    except Exception:
        return None


def _run_arm(
    *,
    arm: str,
    groups: tuple[str, ...],
    input_csv: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> ArmResult:
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    log_path = arm_dir / "run.log"
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_materialized_trailing_label_topk_lgbm_hpo.py",
        "--labels-path",
        str(args.labels_path),
        "--feature-dir",
        str(args.feature_dir),
        "--feature-list-csv",
        str(args.feature_list_csv),
        "--output-dir",
        str(arm_dir),
        "--months",
        str(args.months),
        "--max-train-rows",
        str(args.max_train_rows),
        "--hpo-max-train-rows",
        str(args.hpo_max_train_rows),
        "--n-trials",
        "0",
        "--seed",
        str(args.seed),
        "--fixed-params-json",
        str(args.fixed_params_json),
        "--fixed-selected-features-csv",
        str(args.fixed_selected_features_csv),
        "--allow-refit-ae-gmm-with-fixed-features",
        "--refit-ae-gmm-per-window",
        "--train-window-days",
        str(args.train_window_days),
        "--ae-gmm-anchor-days",
        str(args.ae_gmm_anchor_days),
        "--ae-gmm-input-features-csv",
        str(input_csv),
        "--ae-gmm-state-feature-max-train-rows",
        str(args.ae_gmm_ae_rows),
        "--ae-gmm-state-feature-gmm-max-train-rows",
        str(args.ae_gmm_gmm_rows),
        "--ae-gmm-state-feature-max-iter",
        str(args.ae_gmm_max_iter),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("EPM_AE_GMM_SIDE_CONTEXT_MODE", "off")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("NUMEXPR_MAX_THREADS", "2")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        peak_rss = 0.0
        while proc.poll() is None:
            rss = _rss_mb(proc.pid)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
                log.write(f"[monitor] rss_mb={rss:.1f} peak_rss_mb={peak_rss:.1f}\n")
                log.flush()
                if float(args.max_rss_gb) > 0 and rss > float(args.max_rss_gb) * 1024.0:
                    proc.terminate()
                    raise RuntimeError(f"{arm} exceeded RSS limit {args.max_rss_gb} GiB; log={log_path}")
            time.sleep(float(args.monitor_interval_seconds))
        if proc.returncode != 0:
            raise RuntimeError(f"{arm} failed with exit code {proc.returncode}; log={log_path}")
        log.write(f"[monitor] completed peak_rss_mb={peak_rss:.1f}\n")
    ledger = arm_dir / "best_oos_scored_ledger.parquet"
    metrics, weekly = _summarize_ledger(ledger, arm=arm, groups=groups)
    metrics["peak_rss_mb"] = float(peak_rss)
    weekly.to_csv(arm_dir / "week_metrics.csv", index=False)
    (arm_dir / "arm_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    return ArmResult(
        arm=arm,
        groups=groups,
        output_dir=arm_dir,
        ledger_path=ledger,
        primary_top10_net=float(metrics.get("top10_avg_net_return_per_trade", float("nan"))),
        metrics=metrics,
    )


def _write_incremental_tables(output_dir: Path, rows: list[dict[str, Any]], weekly_parts: list[pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_df = pd.DataFrame(rows)
    if not global_df.empty:
        global_df = global_df.sort_values("top10_avg_net_return_per_trade", ascending=False, kind="mergesort")
    global_df.to_csv(output_dir / "table1_global_performance.csv", index=False)
    if weekly_parts:
        pd.concat(weekly_parts, ignore_index=True).to_csv(output_dir / "table2_weekly_returns.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "table2_weekly_returns.csv", index=False)


def _best_by_arm(results: list[ArmResult], arm: str) -> ArmResult | None:
    matches = [res for res in results if res.arm == arm]
    return matches[0] if matches else None


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_selected = _load_fixed_selected_features(args.fixed_selected_features_csv) or []
    available = _available_features(args.labels_path, args.feature_list_csv)
    a0, groups, feature_diag = _build_groups(available, fixed_selected)
    arm_feature_dir = output_dir / "arm_feature_inputs"
    manifest: dict[str, Any] = {
        "schema": "base_archetype_anchor_ablation_v1",
        "labels_path": str(args.labels_path),
        "feature_dir": str(args.feature_dir),
        "feature_list_csv": str(args.feature_list_csv),
        "fixed_params_json": str(args.fixed_params_json),
        "fixed_selected_features_csv": str(args.fixed_selected_features_csv),
        "months": str(args.months),
        "train_window_days": int(args.train_window_days),
        "ae_gmm_anchor_days": int(args.ae_gmm_anchor_days),
        "ae_gmm_ae_rows": int(args.ae_gmm_ae_rows),
        "ae_gmm_gmm_rows": int(args.ae_gmm_gmm_rows),
        "ae_gmm_max_iter": int(args.ae_gmm_max_iter),
        "available_feature_count": int(len(available)),
        "feature_diagnostics": feature_diag,
        "selection_rule": (
            "A0 vs A0bis first; test standalone A1-A4 groups against winner; "
            "only combine survivor groups, and keep a combination only when top10 net EV/trade "
            "beats both parents."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    results: list[ArmResult] = []

    def execute(arm: str, features: list[str], groups_tuple: tuple[str, ...]) -> ArmResult:
        input_csv = arm_feature_dir / f"{arm}.csv"
        _write_feature_csv(input_csv, features)
        res = _run_arm(arm=arm, groups=groups_tuple, input_csv=input_csv, output_dir=output_dir, args=args)
        rows.append(res.metrics)
        weekly_parts.append(pd.read_csv(res.output_dir / "week_metrics.csv"))
        results.append(res)
        _write_incremental_tables(output_dir, rows, weekly_parts)
        return res

    a0_res = execute("A0_current_selected_inputs", a0, tuple())
    a0bis_res = execute("A0bis_atr_normalized_momentum_inputs", groups["A0bis"], ("A0bis",))
    base_res = a0bis_res if a0bis_res.primary_top10_net > a0_res.primary_top10_net else a0_res
    base_features = groups["A0bis"] if base_res is a0bis_res else a0
    baseline_top10 = float(base_res.primary_top10_net)

    standalone_survivors: list[tuple[str, list[str], ArmResult]] = []
    for group_name in (
        "A1_location_vwap_prior_range",
        "A2_bb_compression_squeeze",
        "A3_support_resistance_barrier_pressure",
        "A4_move_speed_recency",
    ):
        features = list(dict.fromkeys([*base_features, *groups[group_name]]))
        res = execute(group_name, features, (group_name,))
        if res.primary_top10_net > baseline_top10:
            standalone_survivors.append((group_name, features, res))

    tested_group_sets = {tuple([name]) for name, _features, _res in standalone_survivors}
    kept_group_sets = {tuple([name]) for name, _features, _res in standalone_survivors}
    best_combo = max([base_res, *[r for _n, _f, r in standalone_survivors]], key=lambda r: r.primary_top10_net)
    for size in range(2, 5):
        next_level: list[tuple[str, list[str], ArmResult]] = []
        survivor_names = [name for name, _features, _res in standalone_survivors]
        for combo_names in itertools.combinations(survivor_names, size):
            combo_key = tuple(sorted(combo_names))
            if combo_key in tested_group_sets:
                continue
            parent_keys = [tuple(sorted(parent)) for parent in itertools.combinations(combo_key, size - 1)]
            viable_parent_keys = [key for key in parent_keys if key in kept_group_sets]
            if not viable_parent_keys:
                continue
            tested_group_sets.add(combo_key)
            parent_values = [
                res.primary_top10_net
                for res in results
                if tuple(sorted(res.groups)) in set(viable_parent_keys)
            ]
            standalone_values = [
                res.primary_top10_net
                for res in results
                if tuple(sorted(res.groups)) in {(name,) for name in combo_key}
            ]
            if not parent_values or not standalone_values:
                continue
            features = list(base_features)
            for name in combo_key:
                features.extend(groups[name])
            arm = "combo_" + "__".join(name.split("_", 1)[0] for name in combo_key)
            res = execute(arm, list(dict.fromkeys(features)), combo_key)
            if res.primary_top10_net > max(parent_values) and res.primary_top10_net > max(standalone_values):
                next_level.append(("+".join(combo_key), list(dict.fromkeys(features)), res))
                kept_group_sets.add(combo_key)
                if res.primary_top10_net > best_combo.primary_top10_net:
                    best_combo = res
        if not next_level:
            break

    rel_improvement = (
        (float(best_combo.primary_top10_net) - baseline_top10) / abs(baseline_top10)
        if abs(baseline_top10) > 1e-12
        else float("inf")
    )
    final_manifest = {
        **manifest,
        "baseline_arm": base_res.arm,
        "baseline_top10_avg_net_return_per_trade": baseline_top10,
        "best_arm": best_combo.arm,
        "best_groups": list(best_combo.groups),
        "best_top10_avg_net_return_per_trade": float(best_combo.primary_top10_net),
        "relative_top10_improvement_vs_baseline": float(rel_improvement),
        "full_training_confirmation_required": bool(rel_improvement >= float(args.full_training_relative_threshold)),
        "full_training_relative_threshold": float(args.full_training_relative_threshold),
        "note": (
            "This script performs the base ablation only. If full_training_confirmation_required is true, "
            "run the proper base+meta pipeline using the best arm's generated AE/GMM input CSV."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(final_manifest), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(final_manifest), indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--fixed-params-json", type=Path, default=DEFAULT_FIXED_PARAMS_JSON)
    parser.add_argument("--fixed-selected-features-csv", type=Path, default=DEFAULT_FIXED_SELECTED_FEATURES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--train-window-days", type=int, default=150)
    parser.add_argument("--ae-gmm-anchor-days", type=int, default=30)
    parser.add_argument("--ae-gmm-ae-rows", type=int, default=15_000)
    parser.add_argument("--ae-gmm-gmm-rows", type=int, default=100_000)
    parser.add_argument("--ae-gmm-max-iter", type=int, default=80)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--hpo-max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=135042)
    parser.add_argument("--max-rss-gb", type=float, default=56.0)
    parser.add_argument("--monitor-interval-seconds", type=float, default=30.0)
    parser.add_argument("--full-training-relative-threshold", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
