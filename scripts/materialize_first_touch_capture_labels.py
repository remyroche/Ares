#!/usr/bin/env python3
"""Materialize first-touch fixed-capture labels into a trainable label artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.side_aware import add_side_contract_columns  # noqa: E402
from scripts.run_label_first_touch_capture_proxy import (  # noqa: E402
    EXECUTABLE_MARGIN_COST_FLOOR,
    _fetch_policy_paths,
    _first_touch_capture_outcome,
)
from scripts.run_label_quality_proxy_diagnostics import _json_safe, _sigmoid  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import ROUND_TRIP_COST  # noqa: E402
from scripts.run_label_widestop_capture_proxy import CaptureArm  # noqa: E402


DEFAULT_SOURCE_LABELS_DIR = Path(
    "data_perp/artifacts/"
    "20260701_193000_single_head_monthly_walkforward_forwardburnin_no_window_hpo_no_regime_fe_labels_s10_policy_net/"
    "labels"
)
DEFAULT_OUTPUT_RUN_ID = "20260702_094500_first_touch_c0_fast6_s10_policy_net_labels"
OUT_SL = np.int8(0)
OUT_TO = np.int8(1)
OUT_TP = np.int8(2)


def _read_manifest(labels_dir: Path) -> dict[str, Any]:
    path = labels_dir / "labels_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing labels manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_side(dataset_name: str, file_name: str, explicit_side: str | None) -> str:
    if explicit_side:
        return str(explicit_side).strip().lower()
    joined = f"{dataset_name} {file_name}".lower()
    if "train_short" in joined or "_short_" in joined:
        return "short"
    if "train_long" in joined or "_long_" in joined:
        return "long"
    raise ValueError(f"Could not infer side from dataset/file name: {dataset_name} / {file_name}")


def _parse_side_arm_specs(value: str | None) -> dict[str, CaptureArm]:
    """Parse side-specific geometry specs.

    Format:
      side:name:tp_r:sl_r:max_bars_to_mfe:max_barrier[:trail_r]

    Multiple specs are separated by semicolons. Example:
      long:ft_long:0.75:0.5:16:0.05;short:ft_short:1.0:0.5:16:0.03
    """
    if value is None or not str(value).strip():
        return {}
    out: dict[str, CaptureArm] = {}
    for raw_spec in str(value).split(";"):
        spec = raw_spec.strip()
        if not spec:
            continue
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) not in {6, 7}:
            raise ValueError(
                "Invalid --side-arm-specs entry. Expected "
                "side:name:tp_r:sl_r:max_bars_to_mfe:max_barrier[:trail_r], got "
                f"{spec!r}"
            )
        side = parts[0].lower()
        if side not in {"long", "short"}:
            raise ValueError(f"Invalid side in --side-arm-specs entry {spec!r}; expected long or short")
        trail_r = float(parts[6]) if len(parts) == 7 else 0.50
        out[side] = CaptureArm(
            name=parts[1],
            tp_r=float(parts[2]),
            sl_r=float(parts[3]),
            max_bars_to_mfe=float(parts[4]),
            max_barrier=float(parts[5]),
            trail_r=trail_r,
        )
    return out


def _safe_rate(values: Any) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(ser.mean()) if len(ser) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(ser.quantile(q)) if len(ser) else float("nan")


def _monthly_stats(df: pd.DataFrame, capture: pd.DataFrame, policy_soft: np.ndarray) -> list[dict[str, Any]]:
    ts = pd.to_datetime(df["__ts__"], errors="coerce")
    month = ts.dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for period, idx in pd.Series(np.arange(len(df)), index=df.index).groupby(month, dropna=False):
        pos = idx.to_numpy(dtype=np.int64)
        net = pd.to_numeric(capture["capture_net"].iloc[pos], errors="coerce")
        executable_margin = pd.to_numeric(capture.get("executable_margin", pd.Series(np.nan, index=capture.index)).iloc[pos], errors="coerce")
        rows.append(
            {
                "period": str(period),
                "rows": int(len(pos)),
                "capture_net_mean": float(net.mean()) if len(net.dropna()) else float("nan"),
                "capture_net_q10": float(net.quantile(0.10)) if len(net.dropna()) else float("nan"),
                "executable_margin_mean": float(executable_margin.mean())
                if len(executable_margin.dropna())
                else float("nan"),
                "executable_margin_positive_rate": _safe_rate(executable_margin > 0.0),
                "hit_rate": _safe_rate(capture["capture_hit"].iloc[pos]),
                "stop_rate": _safe_rate(capture["capture_stop"].iloc[pos]),
                "timeout_rate": _safe_rate(capture["capture_timeout"].iloc[pos]),
                "eligible_rate": _safe_rate(capture["capture_eligible"].iloc[pos]),
                "net_positive_rate": _safe_rate(net > 0.0),
                "policy_soft_mean": float(np.mean(policy_soft[pos])) if len(pos) else float("nan"),
                "policy_soft_std": float(np.std(policy_soft[pos])) if len(pos) else float("nan"),
                "effective_sl_abs_p90": _safe_quantile(capture["effective_sl_abs"].iloc[pos], 0.90),
            }
        )
    return rows


def _source_copy_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "__y_lbl__",
        "__y_bin__",
        "__y_ret__",
        "__y_outcome__",
        "__is_timeout__",
        "__tp__",
        "__sl__",
        "__u_policy_net__",
        "__r_policy_net__",
        # Preserve pre-rematerialization path-support fields for audits.  The
        # causal materializer overwrites these aliases from its rebuilt path;
        # consumers must not silently inherit support labels from an older
        # entry-timing contract.
        "__mfe__",
        "__mae__",
        "__mfe_ret__",
        "__mae_ret__",
        "__bars_to_mfe__",
        "__bars_to_mae__",
        "__quality__",
        "__w__",
    ]
    return [col for col in cols if col in df.columns]


def _materialize_dataset(
    *,
    source_path: Path,
    output_path: Path,
    dataset_name: str,
    side: str,
    arm: CaptureArm,
    data_root: Path,
    market_mode: str,
    exchange: str,
    timeframe: str,
    path_len: int,
    apply_delayed_entry: bool,
    policy_label_center: float,
    policy_label_temperature: float,
    outcome_mode: str,
    round_trip_cost: float,
    target_mode: str,
    executable_cost_floor: float,
) -> dict[str, Any]:
    df = pd.read_parquet(source_path).reset_index(drop=True)
    required = {"__ts__", "__symbol__", "__barrier_pct__"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"{source_path}: missing required columns {missing}")

    _rows_exec, paths, path_stats = _fetch_policy_paths(
        df,
        labels_path=source_path,
        side=side,
        data_root=data_root,
        market_mode=market_mode,
        exchange=exchange,
        path_len=path_len,
        apply_delayed_entry=apply_delayed_entry,
        timeframe=timeframe,
    )
    capture = _first_touch_capture_outcome(
        df,
        paths,
        arm,
        side_name=side,
        outcome_mode=outcome_mode,
        round_trip_cost=float(round_trip_cost),
        target_mode=str(target_mode),
        executable_cost_floor=float(executable_cost_floor),
    )
    capture_net = pd.to_numeric(capture["capture_net"], errors="coerce").to_numpy(dtype=np.float32)
    capture_gross = pd.to_numeric(capture.get("capture_gross", capture["capture_net"] + float(round_trip_cost)), errors="coerce").to_numpy(dtype=np.float32)
    executable_margin = pd.to_numeric(
        capture.get("executable_margin", capture_gross - max(float(round_trip_cost), float(executable_cost_floor))),
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    policy_soft = _sigmoid(
        (np.nan_to_num(capture_net, nan=float(policy_label_center)) - float(policy_label_center))
        / max(float(policy_label_temperature), 1e-12)
    ).astype(np.float32)

    out = df.copy()
    for col in _source_copy_columns(out):
        out[f"__source{col}"] = out[col].to_numpy(copy=False)

    hit = pd.to_numeric(capture["capture_hit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    stop = pd.to_numeric(capture["capture_stop"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    timeout = pd.to_numeric(capture["capture_timeout"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    eligible = pd.to_numeric(capture["capture_eligible"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    label_code = np.full(len(out), OUT_SL, dtype=np.int8)
    label_code[timeout > 0.5] = OUT_TO
    label_code[hit > 0.5] = OUT_TP

    out["__y_lbl__"] = label_code
    out["__y_outcome__"] = label_code
    out["__y_bin__"] = hit.astype(np.float32)
    out["__y_ret__"] = capture_net
    out["__is_timeout__"] = timeout.astype(np.float32)
    out["__tp__"] = pd.to_numeric(capture["effective_tp_abs"], errors="coerce").to_numpy(dtype=np.float32)
    out["__sl__"] = pd.to_numeric(capture["effective_sl_abs"], errors="coerce").to_numpy(dtype=np.float32)
    out["__u_policy_net__"] = capture_net
    out["__r_policy_net__"] = capture_net

    out["__first_touch_target_soft__"] = pd.to_numeric(
        capture["target_soft"],
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    out["__first_touch_policy_soft__"] = policy_soft
    out["__first_touch_capture_net__"] = capture_net
    out["__first_touch_capture_gross__"] = capture_gross
    out["__first_touch_executable_cost__"] = pd.to_numeric(
        capture.get("executable_cost", pd.Series(max(float(round_trip_cost), float(executable_cost_floor)), index=capture.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    out["__first_touch_executable_cost_floor__"] = np.full(
        len(out),
        max(float(round_trip_cost), float(executable_cost_floor)),
        dtype=np.float32,
    )
    out["__first_touch_executable_margin__"] = executable_margin
    out["__first_touch_gross_minus_cost_floor__"] = executable_margin
    out["__first_touch_executable_margin_positive__"] = (executable_margin > 0.0).astype(np.float32)
    out["__first_touch_round_trip_cost__"] = np.full(
        len(out),
        float(round_trip_cost),
        dtype=np.float32,
    )
    out["__first_touch_hit__"] = hit.astype(np.float32)
    out["__first_touch_stop__"] = stop.astype(np.float32)
    out["__first_touch_timeout__"] = timeout.astype(np.float32)
    out["__first_touch_eligible__"] = eligible.astype(np.float32)
    out["__first_touch_valid_path__"] = pd.to_numeric(
        capture["capture_valid_path"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    out["__first_touch_net_positive__"] = (capture_net > 0.0).astype(np.float32)
    out["__first_touch_bar__"] = pd.to_numeric(
        capture["first_touch_bar"],
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    out["__first_touch_same_bar_both__"] = pd.to_numeric(
        capture["same_bar_both_hit"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    out["__first_touch_effective_tp_abs__"] = out["__tp__"]
    out["__first_touch_effective_sl_abs__"] = out["__sl__"]
    if "effective_trail_abs" in capture.columns:
        out["__first_touch_effective_trail_abs__"] = pd.to_numeric(
            capture["effective_trail_abs"],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
    if "trailing_activated" in capture.columns:
        out["__trailing_profit_activated__"] = pd.to_numeric(
            capture["trailing_activated"],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32)
    if "trailing_activation_bar" in capture.columns:
        out["__trailing_profit_activation_bar__"] = pd.to_numeric(
            capture["trailing_activation_bar"],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
    out["__first_touch_mae_to_sl__"] = pd.to_numeric(
        capture["mae_to_sl"],
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    out["__first_touch_mfe_to_tp__"] = pd.to_numeric(
        capture["mfe_to_tp"],
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    if "first_touch_mae_norm" in capture.columns:
        out["__first_touch_mae_norm__"] = pd.to_numeric(
            capture["first_touch_mae_norm"],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
    if "first_touch_mfe_norm" in capture.columns:
        out["__first_touch_mfe_norm__"] = pd.to_numeric(
            capture["first_touch_mfe_norm"],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
    for col in (
        "bars_to_mfe_05r",
        "bars_to_mfe_075r",
        "bars_to_mfe_1r",
        "bars_to_mfe_125r",
        "bars_to_mfe_15r",
        "bars_to_mae_05r",
        "bars_to_mae_075r",
        "bars_to_mae_1r",
        "bars_to_mae_15r",
        "mfe_1r_before_mae_05r",
        "mfe_1r_before_mae_075r",
        "mfe_1r_before_mae_1r",
        "mae_05r_before_mfe_1r",
        "mae_075r_before_mfe_1r",
        "mae_1r_before_mfe_1r",
        "max_adverse_before_mfe_1r",
        "underwater_bars_before_mfe_1r",
        "underwater_fraction_before_mfe_1r",
        "area_underwater_before_mfe_1r",
    ):
        if col in capture.columns:
            out[f"__{col}__"] = pd.to_numeric(capture[col], errors="coerce").to_numpy(dtype=np.float32)
    for source_col, output_col in (
        ("full_path_mae_to_sl", "__first_touch_full_path_mae_to_sl__"),
        ("full_path_mfe_to_tp", "__first_touch_full_path_mfe_to_tp__"),
        ("full_path_mae_norm", "__first_touch_full_path_mae_norm__"),
        ("full_path_mfe_norm", "__first_touch_full_path_mfe_norm__"),
    ):
        if source_col in capture.columns:
            out[output_col] = pd.to_numeric(capture[source_col], errors="coerce").to_numpy(dtype=np.float32)

    out = add_side_contract_columns(
        out,
        side=side,
        timestamp_col="__ts__",
        asset_col="__symbol__",
        timeframe=timeframe,
        copy=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    finite = np.isfinite(capture_net)
    summary = {
        "dataset": str(dataset_name),
        "source_file": str(source_path),
        "output_file": str(output_path),
        "side": str(side),
        "rows": int(len(out)),
        "finite": int(np.sum(finite)),
        "finite_frac": float(np.mean(finite)) if len(finite) else 0.0,
        "capture_net_mean": float(np.nanmean(capture_net)) if np.any(finite) else float("nan"),
        "capture_net_std": float(np.nanstd(capture_net)) if np.any(finite) else float("nan"),
        "capture_net_p10": float(np.nanpercentile(capture_net, 10)) if np.any(finite) else float("nan"),
        "capture_net_p90": float(np.nanpercentile(capture_net, 90)) if np.any(finite) else float("nan"),
        "capture_gross_mean": float(np.nanmean(capture_gross)) if np.any(np.isfinite(capture_gross)) else float("nan"),
        "executable_margin_mean": float(np.nanmean(executable_margin))
        if np.any(np.isfinite(executable_margin))
        else float("nan"),
        "executable_margin_positive_rate": _safe_rate(executable_margin > 0.0),
        "hit_rate": _safe_rate(hit),
        "stop_rate": _safe_rate(stop),
        "timeout_rate": _safe_rate(timeout),
        "eligible_rate": _safe_rate(eligible),
        "net_positive_rate": _safe_rate(capture_net > 0.0),
        "policy_soft_mean": float(np.mean(policy_soft)) if len(policy_soft) else float("nan"),
        "policy_soft_std": float(np.std(policy_soft)) if len(policy_soft) else float("nan"),
        "effective_tp_abs_p90": _safe_quantile(out["__first_touch_effective_tp_abs__"], 0.90),
        "effective_sl_abs_p90": _safe_quantile(out["__first_touch_effective_sl_abs__"], 0.90),
        "effective_trail_abs_p90": _safe_quantile(out["__first_touch_effective_trail_abs__"], 0.90)
        if "__first_touch_effective_trail_abs__" in out.columns
        else float("nan"),
        "trailing_activated_rate": _safe_rate(out["__trailing_profit_activated__"])
        if "__trailing_profit_activated__" in out.columns
        else float("nan"),
        "outcome_mode": str(outcome_mode),
        "target_mode": str(target_mode),
        "round_trip_cost": float(round_trip_cost),
        "executable_cost_floor": float(executable_cost_floor),
        "path_fetch": path_stats,
        "monthly": _monthly_stats(out, capture, policy_soft),
    }
    return summary


def run_materialization(
    *,
    source_labels_dir: Path,
    output_labels_dir: Path,
    output_run_id: str,
    data_root: Path,
    market_mode: str,
    exchange: str,
    timeframe: str,
    side: str | None,
    include_side: str | None,
    arm: CaptureArm,
    side_arms: dict[str, CaptureArm] | None,
    path_len: int,
    apply_delayed_entry: bool,
    policy_label_center: float,
    policy_label_temperature: float,
    outcome_mode: str,
    round_trip_cost: float,
    target_mode: str,
    executable_cost_floor: float,
    overwrite: bool,
) -> dict[str, Any]:
    if output_labels_dir.exists() and any(output_labels_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_labels_dir} already exists; pass --overwrite to replace files")
    source_manifest = _read_manifest(source_labels_dir)
    datasets = source_manifest.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError(f"No datasets found in {source_labels_dir / 'labels_manifest.json'}")

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = {
        "run_id": str(output_run_id),
        "source_labels_dir": str(source_labels_dir),
        "source_manifest": str(source_labels_dir / "labels_manifest.json"),
        "datasets": {},
        "materialized_first_touch_capture": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root),
            "market_mode": str(market_mode),
            "exchange": str(exchange),
                "timeframe": str(timeframe),
                "include_side": str(include_side) if include_side else None,
                "path_len": int(path_len),
            "apply_delayed_entry": bool(apply_delayed_entry),
                "policy_label_center": float(policy_label_center),
                "policy_label_temperature": float(policy_label_temperature),
                "outcome_mode": str(outcome_mode),
                "target_mode": str(target_mode),
                "round_trip_cost": float(round_trip_cost),
                "executable_cost_floor": float(executable_cost_floor),
                "arm": {
                    "name": arm.name,
                    "tp_r": float(arm.tp_r),
                    "sl_r": float(arm.sl_r),
                    "trail_r": float(getattr(arm, "trail_r", 0.50)),
                    "max_bars_to_mfe": float(arm.max_bars_to_mfe),
                    "max_barrier": float(arm.max_barrier),
                },
                "side_arms": {
                    side_name: {
                        "name": side_arm.name,
                        "tp_r": float(side_arm.tp_r),
                        "sl_r": float(side_arm.sl_r),
                        "trail_r": float(getattr(side_arm, "trail_r", 0.50)),
                        "max_bars_to_mfe": float(side_arm.max_bars_to_mfe),
                        "max_barrier": float(side_arm.max_barrier),
                    }
                    for side_name, side_arm in sorted((side_arms or {}).items())
                },
        },
    }
    summaries: list[dict[str, Any]] = []
    for dataset_name, meta in datasets.items():
        if not isinstance(meta, dict):
            continue
        file_name = str(meta.get("file") or "")
        if not file_name or not file_name.endswith(".parquet"):
            continue
        source_path = source_labels_dir / file_name
        output_path = output_labels_dir / file_name
        inferred_side = _infer_side(str(dataset_name), file_name, None)
        if include_side and inferred_side != str(include_side).strip().lower():
            continue
        resolved_side = _infer_side(str(dataset_name), file_name, side or inferred_side)
        resolved_arm = (side_arms or {}).get(resolved_side, arm)
        summary = _materialize_dataset(
            source_path=source_path,
            output_path=output_path,
            dataset_name=str(dataset_name),
            side=resolved_side,
            arm=resolved_arm,
            data_root=data_root,
            market_mode=market_mode,
            exchange=exchange,
            timeframe=timeframe,
            path_len=path_len,
            apply_delayed_entry=apply_delayed_entry,
            policy_label_center=policy_label_center,
            policy_label_temperature=policy_label_temperature,
            outcome_mode=outcome_mode,
            round_trip_cost=float(round_trip_cost),
            target_mode=str(target_mode),
            executable_cost_floor=float(executable_cost_floor),
        )
        summaries.append(summary)
        out_meta = dict(meta)
        out_meta["file"] = file_name
        out_meta["rows"] = int(summary["rows"])
        out_meta["columns"] = list(pd.read_parquet(output_path).columns)
        out_manifest["datasets"][dataset_name] = out_meta

    if not summaries:
        raise RuntimeError(f"No parquet datasets were materialized from {source_labels_dir}")

    summary_path = output_labels_dir / "first_touch_capture_materialization_summary.json"
    manifest_path = output_labels_dir / "labels_manifest.json"
    summary_path.write_text(json.dumps(_json_safe({"datasets": summaries}), indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(_json_safe(out_manifest), indent=2), encoding="utf-8")
    return {
        "output_labels_dir": str(output_labels_dir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "datasets": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-labels-dir", type=Path, default=DEFAULT_SOURCE_LABELS_DIR)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--output-labels-dir", type=Path, default=None)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--side", choices=("long", "short"), default=None)
    parser.add_argument(
        "--include-side",
        choices=("long", "short"),
        default=None,
        help="Only materialize source datasets inferred to match this side. --side remains an override for outcome computation.",
    )
    parser.add_argument("--arm-name", default="FT_C0_tp075_sl150_fast6_bar30")
    parser.add_argument("--tp-r", type=float, default=0.75)
    parser.add_argument("--sl-r", type=float, default=1.50)
    parser.add_argument("--trail-r", type=float, default=0.50)
    parser.add_argument("--max-bars-to-mfe", type=float, default=6.0)
    parser.add_argument("--max-barrier", type=float, default=0.030)
    parser.add_argument(
        "--side-arm-specs",
        default="",
        help=(
            "Optional semicolon-separated side-specific specs: "
            "side:name:tp_r:sl_r:max_bars_to_mfe:max_barrier[:trail_r]. "
            "When provided, matching datasets use the side-specific arm and the single --arm-* values remain fallback."
        ),
    )
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--no-delayed-entry", action="store_true")
    parser.add_argument("--policy-label-center", type=float, default=0.0)
    parser.add_argument("--policy-label-temperature", type=float, default=0.004)
    parser.add_argument("--outcome-mode", choices=("fixed_tp", "trailing_profit"), default="fixed_tp")
    parser.add_argument("--round-trip-cost", type=float, default=float(ROUND_TRIP_COST))
    parser.add_argument(
        "--target-mode",
        choices=("path_ordered", "executable_margin", "executable_margin_hybrid"),
        default="path_ordered",
        help="Soft-label target mode to materialize.",
    )
    parser.add_argument("--executable-cost-floor", type=float, default=float(EXECUTABLE_MARGIN_COST_FLOOR))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_labels_dir = args.output_labels_dir
    if output_labels_dir is None:
        output_labels_dir = args.data_root / "artifacts" / str(args.output_run_id) / "labels"
    arm = CaptureArm(
        name=str(args.arm_name),
        tp_r=float(args.tp_r),
        sl_r=float(args.sl_r),
        trail_r=float(args.trail_r),
        max_bars_to_mfe=float(args.max_bars_to_mfe),
        max_barrier=float(args.max_barrier),
    )
    side_arms = _parse_side_arm_specs(args.side_arm_specs)
    result = run_materialization(
        source_labels_dir=args.source_labels_dir,
        output_labels_dir=output_labels_dir,
        output_run_id=str(args.output_run_id),
        data_root=args.data_root,
        market_mode=str(args.market_mode),
        exchange=str(args.exchange),
        timeframe=str(args.timeframe),
        side=args.side,
        include_side=args.include_side,
        arm=arm,
        side_arms=side_arms,
        path_len=int(args.path_len),
        apply_delayed_entry=not bool(args.no_delayed_entry),
        policy_label_center=float(args.policy_label_center),
        policy_label_temperature=float(args.policy_label_temperature),
        outcome_mode=str(args.outcome_mode),
        round_trip_cost=float(args.round_trip_cost),
        target_mode=str(args.target_mode),
        executable_cost_floor=float(args.executable_cost_floor),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(_json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
