#!/usr/bin/env python3
"""Materialize strict-OOS proxy prediction handoff from monthly policy-OOS files.

This is a diagnostic helper. It does not train models or score new rows; it
only rebuilds the compact proxy-prediction parquet consumed by source-label
diagnostics from already-generated per-fold ``policy_oos_predictions`` files.
"""

from __future__ import annotations

import argparse
import json
import os
from calendar import month_name
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("data_perp")
DEFAULT_EXPERIMENT_ID = "20260701_193000_single_head_monthly_walkforward_forwardburnin_no_window_hpo_no_regime_fe"
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_OUTPUT_PATH = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "source_tags_s10_policy_net_v17_proxy_alignment_diagnostic"
    / "policy_oos_proxy_predictions_apr_may_jun_rebuilt.parquet"
)

STRICT_PROXY_COLUMNS = [
    "timestamp",
    "symbol",
    "oof_pred",
    "oof_base_clf",
    "oof_meta_clf",
    "base_rank_pct",
    "base_model_score_pct",
    "mr_tf_policy_score_source",
    "pred_H10_pred_mean",
    "base_H10_pred_mean",
    "prediction_source_path",
]


def _parse_csv(value: str | None, default: tuple[str, ...] = ()) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_period_run_ids(value: str | None) -> dict[str, str]:
    """Parse comma-separated period=run_id overrides."""
    out: dict[str, str] = {}
    if value is None or str(value).strip() == "":
        return out
    for part in str(value).split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--period-run-ids entries must be formatted as YYYY-MM=run_id"
            )
        period, run_id = item.split("=", 1)
        period = period.strip()
        run_id = run_id.strip()
        if not period or not run_id:
            raise ValueError(
                "--period-run-ids entries must include both period and run_id"
            )
        out[period] = run_id
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    return value


def month_to_fold_suffix(period: str) -> str:
    ts = pd.Period(str(period), freq="M").to_timestamp()
    prev = ts - pd.DateOffset(months=1)
    prev_name = month_name[int(prev.month)].lower()
    cur_name = month_name[int(ts.month)].lower()
    return f"train_{prev_name}_score_{cur_name}"


def policy_oos_run_id(experiment_id: str, period: str) -> str:
    return f"{experiment_id}_{month_to_fold_suffix(period)}"


def find_policy_oos_file(
    run_root: Path,
    *,
    strategy_id: str | None = None,
) -> Path:
    pred_dir = run_root / "policy_oos_predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(pred_dir)

    if strategy_id:
        direct = pred_dir / f"policy_oos_{strategy_id}_clf.parquet"
        if direct.exists():
            return direct
        matches = sorted(pred_dir.glob(f"policy_oos_*{strategy_id}*_clf.parquet"))
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise RuntimeError(
                f"Multiple policy-OOS files match strategy_id={strategy_id}: "
                + ", ".join(str(path) for path in matches[:10])
            )
        raise FileNotFoundError(direct)

    matches = sorted(pred_dir.glob("policy_oos_*_clf.parquet"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(pred_dir / "policy_oos_*_clf.parquet")
    raise RuntimeError(
        f"Multiple policy-OOS files found under {pred_dir}; pass --strategy-id. "
        + ", ".join(path.name for path in matches[:10])
    )


def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _first_existing(frame: pd.DataFrame, columns: list[str], default: Any = np.nan) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series(default, index=frame.index)


def _rank_pct(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index, dtype="float32")
    return numeric.rank(method="max", pct=True).astype("float32")


def compact_policy_oos_frame(path: Path, *, period: str | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing_required = [col for col in ("timestamp", "symbol") if col not in frame.columns]
    if missing_required:
        raise RuntimeError(f"{path} missing required columns: {missing_required}")

    out = pd.DataFrame(index=frame.index)
    out["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    out["symbol"] = frame["symbol"].astype(str)
    out["oof_pred"] = pd.to_numeric(_first_existing(frame, ["oof_pred", "clf"]), errors="coerce")
    out["oof_base_clf"] = pd.to_numeric(
        _first_existing(frame, ["oof_base_clf", "base_H10_pred_mean", "pred_H10_pred_mean", "clf"]),
        errors="coerce",
    )
    out["oof_meta_clf"] = pd.to_numeric(
        _first_existing(frame, ["oof_meta_clf", "oof_pred", "clf"]),
        errors="coerce",
    )
    out["base_rank_pct"] = pd.to_numeric(
        _first_existing(frame, ["base_rank_pct"], default=np.nan),
        errors="coerce",
    )
    if out["base_rank_pct"].notna().sum() == 0:
        out["base_rank_pct"] = _rank_pct(out["oof_base_clf"])
    out["base_model_score_pct"] = pd.to_numeric(
        _first_existing(frame, ["base_model_score_pct"], default=np.nan),
        errors="coerce",
    )
    if out["base_model_score_pct"].notna().sum() == 0:
        out["base_model_score_pct"] = _rank_pct(out["oof_base_clf"])
    out["mr_tf_policy_score_source"] = _series_or_default(
        frame,
        "mr_tf_policy_score_source",
        "unknown",
    ).astype(str)
    out["pred_H10_pred_mean"] = pd.to_numeric(
        _first_existing(frame, ["pred_H10_pred_mean", "oof_base_clf", "clf"]),
        errors="coerce",
    )
    out["base_H10_pred_mean"] = pd.to_numeric(
        _first_existing(frame, ["base_H10_pred_mean", "oof_base_clf", "clf"]),
        errors="coerce",
    )
    out["prediction_source_path"] = str(path)
    out = out.dropna(subset=["timestamp", "symbol", "oof_pred"]).copy()
    if period:
        periods = out["timestamp"].dt.to_period("M").astype(str)
        out = out.loc[periods.eq(str(period))].copy()
    return out[STRICT_PROXY_COLUMNS].reset_index(drop=True)


def materialize_predictions(
    *,
    data_root: Path,
    experiment_id: str,
    months: list[str],
    output_path: Path,
    strategy_id: str | None = None,
    period_run_ids: dict[str, str] | None = None,
    csv_path: Path | None = None,
    allow_duplicate_keys: bool = False,
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    period_run_ids = period_run_ids or {}
    for period in months:
        run_id = period_run_ids.get(str(period), policy_oos_run_id(experiment_id, period))
        run_root = data_root / "artifacts" / run_id
        source_path = find_policy_oos_file(run_root, strategy_id=strategy_id)
        compact = compact_policy_oos_frame(source_path, period=period)
        frames.append(compact)
        sources.append(
            {
                "period": period,
                "run_id": run_id,
                "source_path": str(source_path),
                "rows": int(len(compact)),
                "min_timestamp": compact["timestamp"].min() if len(compact) else None,
                "max_timestamp": compact["timestamp"].max() if len(compact) else None,
            }
        )

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=STRICT_PROXY_COLUMNS)
    combined = combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    duplicate_key_count = int(combined.duplicated(["timestamp", "symbol"]).sum())
    if duplicate_key_count and not allow_duplicate_keys:
        raise RuntimeError(
            f"Combined strict-OOS proxy predictions have {duplicate_key_count} "
            "duplicate timestamp/symbol keys."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(csv_path, index=False)

    period_counts = (
        combined["timestamp"].dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
        if not combined.empty
        else {}
    )
    manifest = {
        "generated_by": "scripts/materialize_strict_oos_proxy_predictions.py",
        "data_root": str(data_root),
        "experiment_id": experiment_id,
        "strategy_id": strategy_id,
        "months": months,
        "period_run_ids": period_run_ids,
        "rows": int(len(combined)),
        "period_counts": {str(k): int(v) for k, v in period_counts.items()},
        "duplicate_timestamp_symbol_rows": duplicate_key_count,
        "columns": STRICT_PROXY_COLUMNS,
        "sources": sources,
        "output_path": str(output_path),
        "csv_path": str(csv_path) if csv_path else None,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--experiment-id",
        default=os.environ.get("EPM_MONTHLY_WF_ID", DEFAULT_EXPERIMENT_ID),
    )
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument(
        "--period-run-ids",
        default="",
        help=(
            "Optional comma-separated YYYY-MM=run_id overrides. Use this when "
            "one validation month was generated by a different experiment id."
        ),
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--allow-duplicate-keys", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_predictions(
        data_root=args.data_root,
        experiment_id=args.experiment_id,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        output_path=args.output_path,
        strategy_id=args.strategy_id,
        period_run_ids=_parse_period_run_ids(args.period_run_ids),
        csv_path=args.csv_path,
        allow_duplicate_keys=bool(args.allow_duplicate_keys),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
