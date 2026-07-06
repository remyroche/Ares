#!/usr/bin/env python3
"""Production OOF path-order metrics for S52 base ranker artifacts.

This report evaluates the trained base OOF scores against the path-ordered S52
labels.  The primary top-k metric is clean first-touch precision weighted by
gross EV per selected trade, so the base geometry/ranker is judged on the
tradeable edge it ranks into the selected bucket rather than plain hit count.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATA_ROOT = Path("data_perp")
DEFAULT_ARTIFACT_RUN_ID = "20260704_213000_s52_prod_ranker_base"
DEFAULT_LABEL_RUN_ID = "20260704_s52_bidirectional_first_touch_tp075_sl075_fast16_bar50_cost100bps_ordercols_v2_labels"
DEFAULT_TOP_FRACS = (0.10, 0.20, 0.30)

THRESHOLDS: dict[str, float] = {
    "top10_gross_ev_weighted_clean_first_touch_precision_min": 0.70,
    "top20_gross_ev_weighted_clean_first_touch_precision_min": 0.65,
    "top30_gross_ev_weighted_clean_first_touch_precision_min": 0.60,
    "top10_first_touch_bad_mae_1r_rate_max": 0.25,
    "top10_timeout_rate_max": 0.12,
    "top10_mean_first_touch_mae_norm_max": 1.50,
    "top10_p90_first_touch_mae_norm_max": 3.00,
    "top10_mfe_1r_before_mae_1r_rate_min": 0.55,
    "top10_mae_1r_before_mfe_1r_rate_max": 0.35,
    "top10_mean_u_policy_net_min": 0.0,
}

LABEL_COLS = [
    "__ts__",
    "__symbol__",
    "side",
    "side_name",
    "__side__",
    "__u_policy_net__",
    "__source__u_policy_net__",
    "__y_ret__",
    "__source__y_ret__",
    "__first_touch_capture_net__",
    "__first_touch_round_trip_cost__",
    "__first_touch_hit__",
    "__first_touch_stop__",
    "__first_touch_timeout__",
    "__first_touch_eligible__",
    "__first_touch_valid_path__",
    "__first_touch_same_bar_both__",
    "__first_touch_bar__",
    "__first_touch_mae_norm__",
    "__first_touch_mfe_norm__",
    "__mfe_1r_before_mae_05r__",
    "__mfe_1r_before_mae_075r__",
    "__mfe_1r_before_mae_1r__",
    "__mae_05r_before_mfe_1r__",
    "__mae_075r_before_mfe_1r__",
    "__mae_1r_before_mfe_1r__",
    "__max_adverse_before_mfe_1r__",
    "__underwater_bars_before_mfe_1r__",
    "__underwater_fraction_before_mfe_1r__",
    "__area_underwater_before_mfe_1r__",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if pd.isna(value):
        return None
    return value


def _safe_num(values: Any, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        out = pd.to_numeric(values, errors="coerce")
        return out.astype(float)
    return pd.Series(default, index=pd.RangeIndex(0))


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.lower()
    numeric = pd.to_numeric(values, errors="coerce")
    return text.isin({"true", "t", "yes", "y"}) | numeric.fillna(0.0).ne(0.0)


def _norm_ts(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _infer_side(path: Path) -> str:
    name = path.name.lower()
    if "short" in name:
        return "short"
    if "long" in name:
        return "long"
    raise ValueError(f"Cannot infer side from OOF file name: {path}")


def _infer_horizon(path: Path) -> str | None:
    match = re.search(r"_H(\d+)", path.stem)
    return match.group(1) if match else None


def _find_label_file(labels_dir: Path, side: str, horizon: str | None) -> Path:
    patterns: list[str] = []
    if horizon:
        patterns.extend(
            [
                f"train_{side}_*_{horizon}.parquet",
                f"train_{side}_*H{horizon}*.parquet",
            ]
        )
    patterns.append(f"train_{side}_*.parquet")
    for pattern in patterns:
        files = sorted(labels_dir.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No train_{side}_*.parquet label file found under {labels_dir}")


def _read_label_file(path: Path) -> pd.DataFrame:
    all_cols = pd.read_parquet(path, engine="pyarrow").head(0).columns.tolist()
    cols = [c for c in LABEL_COLS if c in all_cols]
    if "__ts__" not in cols or "__symbol__" not in cols:
        raise ValueError(f"Label file lacks __ts__/__symbol__ keys: {path}")
    frame = pd.read_parquet(path, columns=cols)
    frame = frame.rename(columns={"__ts__": "timestamp", "__symbol__": "symbol"})
    frame["timestamp"] = _norm_ts(frame["timestamp"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.dropna(subset=["timestamp", "symbol"])
    return frame.drop_duplicates(["timestamp", "symbol"], keep="last")


def _read_oof_file(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns or "symbol" not in frame.columns:
        raise ValueError(f"OOF file lacks timestamp/symbol keys: {path}")
    score_col = "oof_prob" if "oof_prob" in frame.columns else None
    if score_col is None:
        candidates = [c for c in frame.columns if c.startswith("oof_") and frame[c].dtype.kind in "fc"]
        if not candidates:
            raise ValueError(f"OOF file lacks a usable score column: {path}")
        score_col = candidates[0]
    out = frame.copy()
    out["timestamp"] = _norm_ts(out["timestamp"])
    out["symbol"] = out["symbol"].astype(str)
    out["score"] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.dropna(subset=["timestamp", "symbol", "score"])
    return out


def _read_reference_prediction_file(path: Path, *, artifact_dir: Path, side: str) -> pd.DataFrame:
    """Read lgbm_reference train-time OOF predictions.

    Non-deployable base runs do not always materialize the legacy ``oof/``
    files.  The LGBM reference provenance still stores forward OOF predictions
    keyed by row_index; join those back to the row universe and keep only finite
    OOF scores so burn-in rows cannot enter the top-k buckets.
    """
    frame = pd.read_parquet(path)
    if "row_index" not in frame.columns:
        raise ValueError(f"Reference prediction file lacks row_index: {path}")
    score_col = "oof_prediction" if "oof_prediction" in frame.columns else None
    if score_col is None:
        candidates = [
            c
            for c in ("oof_calibrated_probability", "oof_raw_margin")
            if c in frame.columns and frame[c].dtype.kind in "fc"
        ]
        if not candidates:
            raise ValueError(f"Reference prediction file lacks a usable OOF score column: {path}")
        score_col = candidates[0]
    row_path = artifact_dir / "row_universe" / f"train_global_{side}_5.parquet"
    if not row_path.exists():
        raise FileNotFoundError(f"Row universe not found for reference predictions: {row_path}")
    row_universe = pd.read_parquet(row_path).reset_index(drop=True)
    row_universe = row_universe.reset_index(names="row_index")
    joined = frame[["row_index", score_col]].merge(row_universe, on="row_index", how="inner", validate="one_to_one")
    joined = joined.rename(columns={score_col: "score"})
    joined["timestamp"] = _norm_ts(joined["timestamp"])
    joined["symbol"] = joined["symbol"].astype(str)
    joined["score"] = pd.to_numeric(joined["score"], errors="coerce")
    return joined.dropna(subset=["timestamp", "symbol", "score"])


def _discover_oof_sources(artifact_dir: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    oof_dir = artifact_dir / "oof"
    if oof_dir.exists():
        for oof_file in sorted(oof_dir.glob("oof_*_H*.parquet")):
            sources.append(
                {
                    "path": oof_file,
                    "side": _infer_side(oof_file),
                    "horizon": _infer_horizon(oof_file),
                    "kind": "legacy_oof",
                }
            )
    if sources:
        return sources
    reference_base = artifact_dir / "lgbm_reference" / "base"
    for side in ("long", "short"):
        ref_path = reference_base / f"global_{side}" / "train_time_provenance" / "predictions.parquet"
        if ref_path.exists():
            sources.append(
                {
                    "path": ref_path,
                    "side": side,
                    "horizon": "5",
                    "kind": "lgbm_reference_predictions",
                }
            )
    return sources


def _quantile(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _gross_outcome(frame: pd.DataFrame) -> pd.Series:
    if "__first_touch_capture_net__" in frame.columns:
        net = _safe_num(frame["__first_touch_capture_net__"]).fillna(0.0)
    elif "__u_policy_net__" in frame.columns:
        net = _safe_num(frame["__u_policy_net__"]).fillna(0.0)
    elif "__source__u_policy_net__" in frame.columns:
        net = _safe_num(frame["__source__u_policy_net__"]).fillna(0.0)
    else:
        net = pd.Series(0.0, index=frame.index)
    cost = (
        _safe_num(frame["__first_touch_round_trip_cost__"]).fillna(0.0)
        if "__first_touch_round_trip_cost__" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    return net + cost


def _metric_slice(frame: pd.DataFrame, *, top_frac: float, side: str, scope: str) -> dict[str, Any]:
    total = int(len(frame))
    if total == 0:
        return {
            "scope": scope,
            "side": side,
            "top_frac": float(top_frac),
            "total_rows": 0,
            "selected_rows": 0,
        }
    n_select = max(1, int(math.ceil(float(top_frac) * total)))
    selected = frame.sort_values("score", ascending=False).head(n_select).copy()
    gross = _gross_outcome(selected)
    abs_gross = gross.abs()
    hit = _as_bool(selected.get("__first_touch_hit__", pd.Series(False, index=selected.index)))
    stop = _as_bool(selected.get("__first_touch_stop__", pd.Series(False, index=selected.index)))
    timeout = _as_bool(selected.get("__first_touch_timeout__", pd.Series(False, index=selected.index)))
    valid_path = _as_bool(selected.get("__first_touch_valid_path__", pd.Series(True, index=selected.index)))
    same_bar = _as_bool(selected.get("__first_touch_same_bar_both__", pd.Series(False, index=selected.index)))
    clean_hit = hit & valid_path & ~same_bar & gross.gt(0.0)
    denom = float(abs_gross.sum())
    gross_ev_weighted_precision = float(gross.where(clean_hit, 0.0).clip(lower=0.0).sum() / denom) if denom > 1e-12 else float("nan")
    mae_norm = _safe_num(selected.get("__first_touch_mae_norm__", pd.Series(np.nan, index=selected.index)))
    mfe_norm = _safe_num(selected.get("__first_touch_mfe_norm__", pd.Series(np.nan, index=selected.index)))
    u_policy = _safe_num(
        selected.get("__u_policy_net__", selected.get("__source__u_policy_net__", pd.Series(np.nan, index=selected.index)))
    )
    capture_net = _safe_num(selected.get("__first_touch_capture_net__", pd.Series(np.nan, index=selected.index)))
    return {
        "scope": scope,
        "side": side,
        "top_frac": float(top_frac),
        "total_rows": total,
        "selected_rows": int(len(selected)),
        "score_mean": float(selected["score"].mean()),
        "score_min": float(selected["score"].min()),
        "score_max": float(selected["score"].max()),
        "gross_ev_weighted_clean_first_touch_precision": gross_ev_weighted_precision,
        "raw_clean_first_touch_precision": float(clean_hit.mean()),
        "first_touch_hit_rate": float(hit.mean()),
        "first_touch_stop_rate": float(stop.mean()),
        "first_touch_timeout_rate": float(timeout.mean()),
        "first_touch_bad_mae_1r_rate": float(mae_norm.ge(1.0).mean()),
        "mean_first_touch_mae_norm": float(mae_norm.mean()),
        "p90_first_touch_mae_norm": _quantile(mae_norm, 0.90),
        "mean_first_touch_mfe_norm": float(mfe_norm.mean()),
        "p90_first_touch_mfe_norm": _quantile(mfe_norm, 0.90),
        "mfe_1r_before_mae_05r_rate": float(
            _safe_num(selected.get("__mfe_1r_before_mae_05r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mfe_1r_before_mae_075r_rate": float(
            _safe_num(selected.get("__mfe_1r_before_mae_075r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mfe_1r_before_mae_1r_rate": float(
            _safe_num(selected.get("__mfe_1r_before_mae_1r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mae_05r_before_mfe_1r_rate": float(
            _safe_num(selected.get("__mae_05r_before_mfe_1r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mae_075r_before_mfe_1r_rate": float(
            _safe_num(selected.get("__mae_075r_before_mfe_1r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mae_1r_before_mfe_1r_rate": float(
            _safe_num(selected.get("__mae_1r_before_mfe_1r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mean_max_adverse_before_mfe_1r": float(
            _safe_num(selected.get("__max_adverse_before_mfe_1r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "p90_max_adverse_before_mfe_1r": _quantile(
            _safe_num(selected.get("__max_adverse_before_mfe_1r__", pd.Series(np.nan, index=selected.index))), 0.90
        ),
        "mean_underwater_bars_before_mfe_1r": float(
            _safe_num(selected.get("__underwater_bars_before_mfe_1r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mean_underwater_fraction_before_mfe_1r": float(
            _safe_num(selected.get("__underwater_fraction_before_mfe_1r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mean_area_underwater_before_mfe_1r": float(
            _safe_num(selected.get("__area_underwater_before_mfe_1r__", pd.Series(np.nan, index=selected.index))).mean()
        ),
        "mean_u_policy_net": float(u_policy.mean()),
        "mean_first_touch_capture_net": float(capture_net.mean()),
        "mean_gross_first_touch_outcome": float(gross.mean()),
        "gross_first_touch_ev_sum": float(gross.sum()),
        "gross_first_touch_abs_ev_sum": denom,
        "symbols": int(selected["symbol"].nunique()),
        "days": int(selected["timestamp"].dt.floor("D").nunique()),
    }


def _checks_for_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aggregate = summary[summary["scope"].eq("all")]
    for _, row in aggregate.iterrows():
        side = str(row["side"])
        frac = int(round(float(row["top_frac"]) * 100))
        prefix = f"top{frac}"
        for metric, threshold_key, op in (
            (
                "gross_ev_weighted_clean_first_touch_precision",
                f"{prefix}_gross_ev_weighted_clean_first_touch_precision_min",
                ">=",
            ),
            ("first_touch_bad_mae_1r_rate", f"{prefix}_first_touch_bad_mae_1r_rate_max", "<="),
            ("first_touch_timeout_rate", f"{prefix}_timeout_rate_max", "<="),
            ("mean_first_touch_mae_norm", f"{prefix}_mean_first_touch_mae_norm_max", "<="),
            ("p90_first_touch_mae_norm", f"{prefix}_p90_first_touch_mae_norm_max", "<="),
            ("mfe_1r_before_mae_1r_rate", f"{prefix}_mfe_1r_before_mae_1r_rate_min", ">="),
            ("mae_1r_before_mfe_1r_rate", f"{prefix}_mae_1r_before_mfe_1r_rate_max", "<="),
            ("mean_u_policy_net", f"{prefix}_mean_u_policy_net_min", ">="),
        ):
            if threshold_key not in THRESHOLDS:
                continue
            value = float(row.get(metric, np.nan))
            threshold = float(THRESHOLDS[threshold_key])
            if not math.isfinite(value):
                status = "missing"
            elif op == ">=":
                status = "pass" if value >= threshold else "fail"
            else:
                status = "pass" if value <= threshold else "fail"
            rows.append(
                {
                    "side": side,
                    "top_frac": float(row["top_frac"]),
                    "metric": metric,
                    "value": value,
                    "operator": op,
                    "threshold": threshold,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def _append_side_coverage_checks(checks: pd.DataFrame, sides_seen: set[str]) -> pd.DataFrame:
    rows = []
    for side in ("long", "short"):
        rows.append(
            {
                "side": side,
                "top_frac": np.nan,
                "metric": "side_oof_present",
                "value": 1.0 if side in sides_seen else 0.0,
                "operator": ">=",
                "threshold": 1.0,
                "status": "pass" if side in sides_seen else "fail",
            }
        )
    side_checks = pd.DataFrame(rows)
    if checks.empty:
        return side_checks
    return pd.concat([side_checks, checks], ignore_index=True)


def _status(checks: pd.DataFrame, sides_seen: set[str]) -> str:
    if "long" not in sides_seen or "short" not in sides_seen:
        return "fail_missing_side_oof"
    if checks.empty:
        return "missing"
    statuses = set(checks["status"].astype(str))
    if "missing" in statuses:
        return "fail_missing_evidence"
    if "fail" in statuses:
        return "fail"
    return "pass"


def _format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No rows.\n"
    view = frame[[c for c in columns if c in frame.columns]].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False) + "\n"


def _write_markdown_report(
    *,
    path: Path,
    manifest: dict[str, Any],
    heads: pd.DataFrame,
    summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    lines = [
        "# S52 Production OOF Path-Order Metrics",
        "",
        f"Status: `{manifest['status']}`",
        f"Artifact run: `{manifest['artifact_run_id']}`",
        f"Label run: `{manifest['label_run_id']}`",
        "",
        "## Head Coverage",
        "",
        _format_table(
            heads,
            ["side", "horizon", "oof_rows", "label_rows", "joined_rows", "join_rate", "oof_file"],
        ),
        "## Top-K Path Metrics",
        "",
        _format_table(
            summary,
            [
                "side",
                "top_frac",
                "selected_rows",
                "gross_ev_weighted_clean_first_touch_precision",
                "raw_clean_first_touch_precision",
                "first_touch_bad_mae_1r_rate",
                "first_touch_timeout_rate",
                "mean_first_touch_mae_norm",
                "p90_first_touch_mae_norm",
                "mfe_1r_before_mae_1r_rate",
                "mae_1r_before_mfe_1r_rate",
                "mean_u_policy_net",
                "symbols",
                "days",
            ],
        ),
        "## Gate Checks",
        "",
        _format_table(checks, ["side", "top_frac", "metric", "value", "operator", "threshold", "status"]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(
    *,
    artifact_run_id: str,
    label_run_id: str,
    data_root: Path,
    output_dir: Path,
    top_fracs: tuple[float, ...] = DEFAULT_TOP_FRACS,
) -> dict[str, Any]:
    artifact_dir = data_root / "artifacts" / artifact_run_id
    labels_dir = data_root / "artifacts" / label_run_id / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")
    oof_sources = _discover_oof_sources(artifact_dir)
    if not oof_sources:
        raise FileNotFoundError(
            f"No OOF sources found under {artifact_dir / 'oof'} or {artifact_dir / 'lgbm_reference' / 'base'}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    sides_seen: set[str] = set()
    for source in oof_sources:
        oof_file = Path(source["path"])
        side = str(source["side"])
        horizon = str(source["horizon"]) if source.get("horizon") is not None else None
        label_file = _find_label_file(labels_dir, side, horizon)
        if source.get("kind") == "lgbm_reference_predictions":
            oof = _read_reference_prediction_file(oof_file, artifact_dir=artifact_dir, side=side)
        else:
            oof = _read_oof_file(oof_file)
        labels = _read_label_file(label_file)
        joined = oof.merge(labels, on=["timestamp", "symbol"], how="inner", validate="many_to_one")
        sides_seen.add(side)
        head_rows.append(
            {
                "side": side,
                "horizon": horizon,
                "source_kind": str(source.get("kind", "unknown")),
                "oof_file": str(oof_file),
                "label_file": str(label_file),
                "oof_rows": int(len(oof)),
                "label_rows": int(len(labels)),
                "joined_rows": int(len(joined)),
                "join_rate": float(len(joined) / max(len(oof), 1)),
            }
        )
        if joined.empty:
            continue
        for frac in top_fracs:
            rows.append(_metric_slice(joined, top_frac=frac, side=side, scope="all"))
            by_month = joined.assign(month=joined["timestamp"].dt.strftime("%Y-%m"))
            for month, month_frame in by_month.groupby("month", observed=True):
                monthly_rows.append(_metric_slice(month_frame, top_frac=frac, side=side, scope=str(month)))

    summary = pd.DataFrame(rows)
    monthly = pd.DataFrame(monthly_rows)
    heads = pd.DataFrame(head_rows)
    checks = _checks_for_summary(summary) if not summary.empty else pd.DataFrame()
    checks = _append_side_coverage_checks(checks, sides_seen)
    status = _status(checks, sides_seen)

    summary_path = output_dir / "s52_oof_path_order_summary.csv"
    monthly_path = output_dir / "s52_oof_path_order_monthly.csv"
    checks_path = output_dir / "s52_oof_path_order_gate_checks.csv"
    heads_path = output_dir / "s52_oof_path_order_heads.csv"
    summary.to_csv(summary_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    checks.to_csv(checks_path, index=False)
    heads.to_csv(heads_path, index=False)
    manifest = {
        "artifact_run_id": artifact_run_id,
        "label_run_id": label_run_id,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "top_fracs": list(top_fracs),
        "thresholds": THRESHOLDS,
        "sides_seen": sorted(sides_seen),
        "status": status,
        "summary_csv": str(summary_path),
        "monthly_csv": str(monthly_path),
        "gate_checks_csv": str(checks_path),
        "heads_csv": str(heads_path),
        "markdown_report": str(output_dir / "s52_oof_path_order_report.md"),
    }
    _write_markdown_report(
        path=output_dir / "s52_oof_path_order_report.md",
        manifest=manifest,
        heads=heads,
        summary=summary,
        checks=checks,
    )
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "summary": summary,
        "monthly": monthly,
        "checks": checks,
        "heads": heads,
    }


def _parse_top_fracs(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-run-id", default=DEFAULT_ARTIFACT_RUN_ID)
    parser.add_argument("--label-run-id", default=DEFAULT_LABEL_RUN_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-fracs", default=",".join(str(x) for x in DEFAULT_TOP_FRACS))
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.data_root / "artifacts" / args.artifact_run_id / "reports" / "s52_oof_path_order_metrics"
    result = build_report(
        artifact_run_id=str(args.artifact_run_id),
        label_run_id=str(args.label_run_id),
        data_root=args.data_root,
        output_dir=output_dir,
        top_fracs=_parse_top_fracs(args.top_fracs),
    )
    print(json.dumps(_json_safe(result["manifest"]), indent=2))


if __name__ == "__main__":
    main()
