#!/usr/bin/env python3
"""Build target and label validation diagnostics from a candidate ledger.

The report is intentionally ledger-based: it validates whether observable labels
and target-like columns separate realized utility/path outcomes in the exact
candidate stream used by base/meta smoke tests.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = (
    "u_policy_net",
    "ret_net",
    "mae_norm",
    "mfe_norm",
    "barrier",
    "is_timeout",
    "bad_mae_1r",
    "clean_positive",
    "dirty_positive",
    "oracle_top",
    "clean_oracle_top",
    "selector_score",
    "selector_rank_pct",
    "selector_ts_rank_pct",
    "selector_ts_side_rank_pct",
    "base_model_score",
    "bad_mae_pred",
    "timeout_pred",
    "clean_path_pred",
    "lgbm_bad_mae_pred",
    "lgbm_timeout_pred",
    "lgbm_clean_path_pred",
    "lgbm_dirty_positive_bad_mae_pred",
    "lgbm_positive_clean_path_pred",
    "lgbm_ranker_score",
    "lgbm_path_ranker_score",
    "lgbm_path_first_ranker_score",
    "lgbm_s24_broad_path_first_ranker_score",
    "lgbm_s28_side_s24_ranker_score",
    "lgbm_s30_side_asym_ranker_score",
)

BAD_LABEL_TOKENS = ("bad", "timeout", "mae", "adverse", "dirty", "risk")
GOOD_LABEL_TOKENS = ("utility", "ret", "mfe", "favorable", "clean", "oracle", "score", "ranker")


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


def _safe_mean(values: pd.Series) -> float:
    out = pd.to_numeric(values, errors="coerce").mean()
    return float(out) if pd.notna(out) else float("nan")


def _safe_quantile(values: pd.Series, q: float) -> float:
    out = pd.to_numeric(values, errors="coerce").astype(float).quantile(q)
    return float(out) if pd.notna(out) else float("nan")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 30:
        return float("nan")
    return float(x.loc[mask].corr(y.loc[mask], method="spearman"))


def _binary_like(values: pd.Series) -> bool:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return False
    unique = set(float(v) for v in x.unique()[:8])
    return unique.issubset({0.0, 1.0}) or len(unique) <= 2


def _expected_direction(name: str) -> int:
    lower = str(name).lower()
    if any(token in lower for token in BAD_LABEL_TOKENS):
        return -1
    if any(token in lower for token in GOOD_LABEL_TOKENS):
        return 1
    return 0


def _top_bottom_masks(values: pd.Series, quantile: float) -> tuple[pd.Series, pd.Series]:
    score = pd.to_numeric(values, errors="coerce")
    valid = score.notna()
    if int(valid.sum()) < 50:
        empty = pd.Series(False, index=values.index)
        return empty, empty
    if _binary_like(score):
        high = score.ge(0.5) & valid
        low = score.lt(0.5) & valid
    else:
        lo = float(score.loc[valid].quantile(quantile))
        hi = float(score.loc[valid].quantile(1.0 - quantile))
        high = score.ge(hi) & valid
        low = score.le(lo) & valid
    return high, low


def _spread(values: pd.Series, outcome: pd.Series, quantile: float, direction: int) -> float:
    high, low = _top_bottom_masks(values, quantile)
    if int(high.sum()) < 5 or int(low.sum()) < 5:
        return float("nan")
    raw = _safe_mean(outcome.loc[high]) - _safe_mean(outcome.loc[low])
    return float(raw * direction) if direction else float(raw)


def _positive_slice_rate(
    frame: pd.DataFrame,
    label: str,
    outcome: str,
    slice_cols: list[str],
    *,
    quantile: float,
    direction: int,
    min_rows: int,
) -> float:
    signs: list[float] = []
    for _key, group in frame.groupby(slice_cols, dropna=False, observed=False):
        if len(group) < min_rows:
            continue
        spread = _spread(group[label], group[outcome], quantile, direction)
        if math.isfinite(spread):
            signs.append(1.0 if spread > 0.0 else 0.0)
    return float(np.mean(signs)) if signs else float("nan")


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0.0].sum())
    losses = float(-x[x < 0.0].sum())
    if losses <= 0.0:
        return float("inf") if gains > 0.0 else float("nan")
    return gains / losses


def _decile_rows(frame: pd.DataFrame, column: str, *, quantile_count: int) -> list[dict[str, Any]]:
    score = pd.to_numeric(frame[column], errors="coerce")
    valid = frame.loc[score.notna()].copy()
    if len(valid) < quantile_count * 10 or _binary_like(valid[column]):
        return []
    try:
        valid["bucket"] = pd.qcut(pd.to_numeric(valid[column], errors="coerce"), quantile_count, labels=False, duplicates="drop")
    except ValueError:
        return []
    rows: list[dict[str, Any]] = []
    for bucket, group in valid.groupby("bucket", dropna=True):
        rows.append(
            {
                "label_name": column,
                "bucket": int(bucket),
                "rows": int(len(group)),
                "mean_u": _safe_mean(group["u_policy_net"]),
                "median_u": _safe_quantile(group["u_policy_net"], 0.50),
                "p10_u": _safe_quantile(group["u_policy_net"], 0.10),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r"]) if "bad_mae_1r" in group.columns else float("nan"),
                "timeout_rate": _safe_mean(group["is_timeout"]) if "is_timeout" in group.columns else float("nan"),
                "clean_positive_rate": _safe_mean(group["clean_positive"]) if "clean_positive" in group.columns else float("nan"),
                "dirty_positive_rate": _safe_mean(group["dirty_positive"]) if "dirty_positive" in group.columns else float("nan"),
                "mfe_mean": _safe_mean(group["mfe_norm"]) if "mfe_norm" in group.columns else float("nan"),
                "mae_mean": _safe_mean(group["mae_norm"]) if "mae_norm" in group.columns else float("nan"),
                "profit_factor": _profit_factor(group["u_policy_net"]),
            }
        )
    return rows


def build_report(
    input_path: Path,
    output_dir: Path,
    *,
    columns: list[str] | None,
    quantile: float,
    min_slice_rows: int,
) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    required = {"u_policy_net"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if "period" not in frame.columns:
        frame["period"] = "unknown"
    side = pd.to_numeric(frame.get("side", pd.Series(1.0, index=frame.index)), errors="coerce").fillna(1.0)
    frame["__side_name__"] = np.where(side < 0.0, "short", "long")
    spread_col = "ctx_median_spread_bps" if "ctx_median_spread_bps" in frame.columns else None
    if spread_col is not None:
        spread = pd.to_numeric(frame[spread_col], errors="coerce")
        try:
            frame["__spread_bucket__"] = pd.qcut(spread, 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
        except ValueError:
            frame["__spread_bucket__"] = "unknown"
    else:
        frame["__spread_bucket__"] = "unknown"

    candidates = columns or [col for col in DEFAULT_COLUMNS if col in frame.columns]
    rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        direction = _expected_direction(column)
        high, low = _top_bottom_masks(values, quantile)
        label_type = "binary" if _binary_like(values) else "continuous"
        u_spread = _spread(values, frame["u_policy_net"], quantile, direction)
        bad_spread = (
            _spread(values, frame["bad_mae_1r"], quantile, -direction if direction else 1)
            if "bad_mae_1r" in frame.columns
            else float("nan")
        )
        timeout_spread = (
            _spread(values, frame["is_timeout"], quantile, -direction if direction else 1)
            if "is_timeout" in frame.columns
            else float("nan")
        )
        clean_spread = (
            _spread(values, frame["clean_positive"], quantile, direction)
            if "clean_positive" in frame.columns
            else float("nan")
        )
        stable_rates = {
            "month_positive_spread_rate": _positive_slice_rate(
                frame,
                column,
                "u_policy_net",
                ["period"],
                quantile=quantile,
                direction=direction or 1,
                min_rows=min_slice_rows,
            ),
            "side_positive_spread_rate": _positive_slice_rate(
                frame,
                column,
                "u_policy_net",
                ["__side_name__"],
                quantile=quantile,
                direction=direction or 1,
                min_rows=min_slice_rows,
            ),
            "spread_positive_spread_rate": _positive_slice_rate(
                frame,
                column,
                "u_policy_net",
                ["__spread_bucket__"],
                quantile=quantile,
                direction=direction or 1,
                min_rows=min_slice_rows,
            ),
        }
        promotion_status = "diagnostic"
        if math.isfinite(u_spread) and u_spread > 0.0:
            if min(v for v in stable_rates.values() if math.isfinite(v)) >= 0.60:
                promotion_status = "candidate_auxiliary"
        if label_type == "binary":
            prevalence = float((valid > 0.5).mean())
            if prevalence < 0.02 or prevalence > 0.98:
                promotion_status = "quarantine_imbalance"
        rows.append(
            {
                "label_name": column,
                "label_type": label_type,
                "expected_direction": direction,
                "rows": int(len(frame)),
                "valid_rows": int(valid.size),
                "missing_rate": float(1.0 - valid.size / max(len(frame), 1)),
                "prevalence": float((valid > 0.5).mean()) if label_type == "binary" else float("nan"),
                "mean": _safe_mean(values),
                "std": float(valid.std()) if valid.size else float("nan"),
                "skew": float(valid.skew()) if valid.size > 2 else float("nan"),
                "p01": _safe_quantile(values, 0.01),
                "p05": _safe_quantile(values, 0.05),
                "p10": _safe_quantile(values, 0.10),
                "p50": _safe_quantile(values, 0.50),
                "p90": _safe_quantile(values, 0.90),
                "p95": _safe_quantile(values, 0.95),
                "p99": _safe_quantile(values, 0.99),
                "spearman_ic_u": _spearman(values, frame["u_policy_net"]),
                "utility_top_bottom_spread_aligned": u_spread,
                "raw_high_minus_low_u": _safe_mean(frame.loc[high, "u_policy_net"]) - _safe_mean(frame.loc[low, "u_policy_net"])
                if int(high.sum()) >= 5 and int(low.sum()) >= 5
                else float("nan"),
                "p10_u_high_minus_low": _safe_quantile(frame.loc[high, "u_policy_net"], 0.10)
                - _safe_quantile(frame.loc[low, "u_policy_net"], 0.10)
                if int(high.sum()) >= 5 and int(low.sum()) >= 5
                else float("nan"),
                "bad_mae_spread_aligned": bad_spread,
                "timeout_spread_aligned": timeout_spread,
                "clean_positive_spread_aligned": clean_spread,
                "high_bucket_rows": int(high.sum()),
                "low_bucket_rows": int(low.sum()),
                **stable_rates,
                "promotion_status": promotion_status,
            }
        )
        decile_rows.extend(_decile_rows(frame, column, quantile_count=10))

    dashboard = pd.DataFrame(rows).sort_values(
        ["promotion_status", "utility_top_bottom_spread_aligned", "spearman_ic_u"],
        ascending=[True, False, False],
        na_position="last",
    )
    deciles = pd.DataFrame(decile_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "target_label_validation_dashboard.csv"
    decile_path = output_dir / "target_label_validation_deciles.csv"
    json_path = output_dir / "target_label_validation_dashboard.json"
    md_path = output_dir / "target_label_validation_dashboard.md"
    dashboard.to_csv(csv_path, index=False)
    deciles.to_csv(decile_path, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "rows": int(len(frame)),
        "quantile": float(quantile),
        "min_slice_rows": int(min_slice_rows),
        "outputs": {"csv": str(csv_path), "deciles_csv": str(decile_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    display_cols = [
        "label_name",
        "label_type",
        "expected_direction",
        "spearman_ic_u",
        "utility_top_bottom_spread_aligned",
        "p10_u_high_minus_low",
        "bad_mae_spread_aligned",
        "timeout_spread_aligned",
        "month_positive_spread_rate",
        "side_positive_spread_rate",
        "spread_positive_spread_rate",
        "promotion_status",
    ]
    lines = [
        "# Target/Label Validation Dashboard",
        "",
        f"- Input: `{input_path}`",
        f"- Rows: `{len(frame)}`",
        "",
    ]
    if not dashboard.empty:
        lines.append(dashboard[[col for col in display_cols if col in dashboard.columns]].head(40).to_markdown(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", default="")
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--min-slice-rows", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns = [part.strip() for part in str(args.columns).split(",") if part.strip()] or None
    manifest = build_report(
        args.input,
        args.output_dir,
        columns=columns,
        quantile=float(args.quantile),
        min_slice_rows=int(args.min_slice_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
