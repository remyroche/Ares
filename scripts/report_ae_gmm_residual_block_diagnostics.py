#!/usr/bin/env python3
"""Report residual utility and path-risk relevance for AE/GMM feature blocks."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINE_SCORE_COLUMNS = (
    "selector_score",
    "selector_rank_pct",
    "selector_ts_rank_pct",
    "selector_ts_side_rank_pct",
    "base_model_score",
    "lgbm_ranker_score",
    "lgbm_path_ranker_score",
    "lgbm_bad_mae_pred",
    "lgbm_timeout_pred",
    "lgbm_clean_path_pred",
    "lgbm_dirty_positive_bad_mae_pred",
)


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


def _feature_blocks(name: str) -> set[str]:
    lower = str(name).lower()
    out: set[str] = set()
    if not lower.startswith("ctx_"):
        return out
    if lower.startswith("ctx_long_"):
        out.add("long")
    elif lower.startswith("ctx_short_"):
        out.add("short")
    elif any(token in lower for token in ("gmm", "cluster", "posterior", "mahal", "reconstruction", "latent")):
        out.add("global")
    else:
        out.add("market")
    if any(token in lower for token in ("gmm_prob_", "posterior_")):
        out.add("soft_prob")
    if any(token in lower for token in ("dist_center", "mahal", "density", "nll", "likelihood")):
        out.add("distance")
    if any(token in lower for token in ("delta_", "accel", "speed", "time_since", "stability", "flip_count")):
        out.add("transition")
    if "entropy" in lower:
        out.add("entropy")
    if "reconstruction" in lower:
        out.add("reconstruction")
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 20:
        return float("nan")
    return float(x.loc[mask].corr(y.loc[mask], method="spearman"))


def _zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    means = x.mean(axis=0)
    stds = x.std(axis=0).replace(0.0, np.nan)
    return ((x - means) / stds).fillna(0.0)


def _residual_after_baseline(frame: pd.DataFrame, y_col: str) -> pd.Series:
    y = pd.to_numeric(frame[y_col], errors="coerce")
    score_cols = [col for col in BASELINE_SCORE_COLUMNS if col in frame.columns]
    if not score_cols:
        return y - float(y.mean(skipna=True))
    x = frame[score_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = pd.concat([y.rename("__y__"), x], axis=1).dropna()
    if len(work) < max(30, len(score_cols) + 5):
        return y - float(y.mean(skipna=True))
    xw = work[score_cols].to_numpy(dtype=np.float64)
    xw = np.column_stack([np.ones(len(xw)), (xw - np.nanmean(xw, axis=0)) / (np.nanstd(xw, axis=0) + 1e-9)])
    yw = work["__y__"].to_numpy(dtype=np.float64)
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    pred = np.full(len(frame), np.nan, dtype=np.float64)
    all_x = x.to_numpy(dtype=np.float64)
    col_mean = np.nanmean(work[score_cols].to_numpy(dtype=np.float64), axis=0)
    col_std = np.nanstd(work[score_cols].to_numpy(dtype=np.float64), axis=0) + 1e-9
    valid = np.isfinite(all_x).all(axis=1)
    pred[valid] = np.column_stack([np.ones(int(valid.sum())), (all_x[valid] - col_mean) / col_std]) @ beta
    residual = y.to_numpy(dtype=np.float64) - pred
    out = pd.Series(residual, index=frame.index)
    return out.fillna(y - float(y.mean(skipna=True)))


def _top_bottom_spread(score: pd.Series, target: pd.Series, quantile: float = 0.20) -> float:
    s = pd.to_numeric(score, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    mask = s.notna() & y.notna()
    if int(mask.sum()) < 50:
        return float("nan")
    s = s.loc[mask]
    y = y.loc[mask]
    lo = float(s.quantile(quantile))
    hi = float(s.quantile(1.0 - quantile))
    return float(y.loc[s.ge(hi)].mean() - y.loc[s.le(lo)].mean())


def _slice_positive_rate(frame: pd.DataFrame, score: pd.Series, target: pd.Series, slice_cols: list[str]) -> float:
    signs: list[float] = []
    local = frame.copy()
    local["__score__"] = score
    local["__target__"] = target
    for _key, group in local.groupby(slice_cols, dropna=False):
        if len(group) < 50:
            continue
        spread = _top_bottom_spread(group["__score__"], group["__target__"])
        if math.isfinite(spread):
            signs.append(1.0 if spread > 0.0 else 0.0)
    return float(np.mean(signs)) if signs else float("nan")


def build_report(input_path: Path, output_dir: Path, *, quantile: float) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    if "u_policy_net" not in frame.columns:
        raise ValueError("candidate ledger must contain u_policy_net")
    residual_u = _residual_after_baseline(frame, "u_policy_net")
    side = pd.to_numeric(frame.get("side", pd.Series(1.0, index=frame.index)), errors="coerce").fillna(1.0)
    frame["__side_name__"] = np.where(side < 0.0, "short", "long")
    if "period" not in frame.columns:
        frame["period"] = "unknown"
    block_names = ["market", "global", "long", "short", "soft_prob", "distance", "transition", "entropy", "reconstruction"]
    rows: list[dict[str, Any]] = []
    columns = [str(col) for col in frame.columns if str(col).startswith("ctx_")]
    for block in block_names:
        block_cols = [col for col in columns if block in _feature_blocks(col)]
        if not block_cols:
            continue
        z = _zscore_frame(frame[block_cols])
        score = z.mean(axis=1)
        feature_ics = [_spearman(frame[col], residual_u) for col in block_cols]
        abs_ics = [abs(v) for v in feature_ics if math.isfinite(v)]
        rows.append(
            {
                "block": block,
                "feature_count": int(len(block_cols)),
                "rows": int(len(frame)),
                "score_ic_u": _spearman(score, frame["u_policy_net"]),
                "score_ic_residual_u": _spearman(score, residual_u),
                "score_ic_bad_mae": _spearman(score, frame["bad_mae_1r"]) if "bad_mae_1r" in frame.columns else float("nan"),
                "score_ic_timeout": _spearman(score, frame["is_timeout"]) if "is_timeout" in frame.columns else float("nan"),
                "score_ic_clean_positive": _spearman(score, frame["clean_positive"]) if "clean_positive" in frame.columns else float("nan"),
                "top_bottom_u_spread": _top_bottom_spread(score, frame["u_policy_net"], quantile=quantile),
                "top_bottom_residual_u_spread": _top_bottom_spread(score, residual_u, quantile=quantile),
                "top_bottom_bad_mae_spread": _top_bottom_spread(score, frame["bad_mae_1r"], quantile=quantile)
                if "bad_mae_1r" in frame.columns
                else float("nan"),
                "top_bottom_timeout_spread": _top_bottom_spread(score, frame["is_timeout"], quantile=quantile)
                if "is_timeout" in frame.columns
                else float("nan"),
                "side_month_positive_residual_spread_rate": _slice_positive_rate(
                    frame,
                    score,
                    residual_u,
                    ["__side_name__", "period"],
                ),
                "max_abs_feature_residual_ic": float(max(abs_ics)) if abs_ics else float("nan"),
                "median_abs_feature_residual_ic": float(np.median(abs_ics)) if abs_ics else float("nan"),
                "example_features": ",".join(block_cols[:8]),
            }
        )
    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ae_gmm_residual_block_diagnostics.csv"
    json_path = output_dir / "ae_gmm_residual_block_diagnostics.json"
    md_path = output_dir / "ae_gmm_residual_block_diagnostics.md"
    out.to_csv(csv_path, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "rows": int(len(frame)),
        "blocks": rows,
        "outputs": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    cols = [
        "block",
        "feature_count",
        "score_ic_u",
        "score_ic_residual_u",
        "score_ic_bad_mae",
        "score_ic_timeout",
        "top_bottom_u_spread",
        "top_bottom_residual_u_spread",
        "top_bottom_bad_mae_spread",
        "side_month_positive_residual_spread_rate",
        "max_abs_feature_residual_ic",
    ]
    lines = ["# AE/GMM Residual Block Diagnostics", "", f"- Input: `{input_path}`", ""]
    if not out.empty:
        lines.append(out[[col for col in cols if col in out.columns]].to_markdown(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_report(args.input, args.output_dir, quantile=float(args.quantile))
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
