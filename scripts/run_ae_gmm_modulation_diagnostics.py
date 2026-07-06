#!/usr/bin/env python3
"""Leakage-safe AE/GMM threshold modulation diagnostics.

This script does not create trades. It starts from a candidate ledger, learns a
small archetype block score on prior periods only, applies a bounded rank shift,
and reselects the same keep fraction with the same side cap. The goal is to test
whether AE/GMM descriptors are useful for threshold/rank modulation when direct
feature feeding is weak.
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


BLOCKS = ("market", "global", "long", "short", "soft_prob", "distance", "transition", "entropy", "reconstruction")
OBJECTIVES = ("utility", "balanced", "path_risk")


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
    out = pd.to_numeric(values, errors="coerce").quantile(q)
    return float(out) if pd.notna(out) else float("nan")


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


def _block_columns(columns: list[str], block: str) -> list[str]:
    return [col for col in columns if block in _feature_blocks(col)]


def _zscore(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_x = valid[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0).replace(0.0, np.nan)
    return ((train_x - means) / stds).fillna(0.0), ((valid_x - means) / stds).fillna(0.0)


def _fit_block_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    cols: list[str],
    *,
    max_features: int,
    objective_name: str,
) -> pd.Series:
    if len(cols) == 0 or len(train) < 80:
        return pd.Series(0.0, index=valid.index, dtype=np.float32)
    train_z, valid_z = _zscore(train, valid, cols)
    utility = pd.to_numeric(train["u_policy_net"], errors="coerce").fillna(0.0)
    bad = pd.to_numeric(train.get("bad_mae_1r", 0.0), errors="coerce").fillna(0.0)
    timeout = pd.to_numeric(train.get("is_timeout", 0.0), errors="coerce").fillna(0.0)
    # Normalize each component before combining so a binary risk label does not
    # dominate purely because of scale.
    def norm(s: pd.Series) -> pd.Series:
        return (s - float(s.mean())) / (float(s.std()) + 1e-9)

    if objective_name == "utility":
        objective = norm(utility)
    elif objective_name == "path_risk":
        objective = -0.60 * norm(bad) - 0.40 * norm(timeout)
    elif objective_name == "balanced":
        objective = norm(utility) - 0.35 * norm(bad) - 0.25 * norm(timeout)
    else:
        raise ValueError(f"unknown modulation objective {objective_name!r}; expected one of {OBJECTIVES}")
    weights: list[tuple[str, float]] = []
    for col in cols:
        corr = train_z[col].corr(objective, method="spearman")
        if pd.notna(corr) and math.isfinite(float(corr)):
            weights.append((col, float(corr)))
    weights = sorted(weights, key=lambda item: abs(item[1]), reverse=True)[: int(max_features)]
    if not weights:
        return pd.Series(0.0, index=valid.index, dtype=np.float32)
    denom = sum(abs(weight) for _col, weight in weights) or 1.0
    score = pd.Series(0.0, index=valid.index, dtype=np.float64)
    for col, weight in weights:
        score += valid_z[col] * (weight / denom)
    centered = score - float(score.mean())
    scaled = centered / (float(centered.std()) + 1e-9)
    return scaled.clip(-2.0, 2.0).astype(np.float32)


def _side_capped_indices(score: pd.Series, side: pd.Series, *, keep_frac: float, max_side_share: float) -> np.ndarray:
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    side_s = pd.to_numeric(side.reset_index(drop=True), errors="coerce").fillna(0.0)
    valid_idx = np.flatnonzero(score_s.notna().to_numpy())
    if len(valid_idx) == 0:
        return np.asarray([], dtype=np.int64)
    target_n = max(1, int(math.ceil(len(valid_idx) * float(keep_frac))))
    max_per_side = max(1, int(math.floor(target_n * float(max_side_share))))
    order = valid_idx[np.argsort(-score_s.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")]
    selected: list[int] = []
    side_counts = {"long": 0, "short": 0, "flat": 0}
    for idx in order:
        side_name = "long" if float(side_s.iloc[idx]) > 0.0 else "short" if float(side_s.iloc[idx]) < 0.0 else "flat"
        if side_name in {"long", "short"} and side_counts[side_name] >= max_per_side:
            other = "short" if side_name == "long" else "long"
            if side_counts[other] < max_per_side and len(selected) < target_n:
                continue
        selected.append(int(idx))
        side_counts[side_name] = side_counts.get(side_name, 0) + 1
        if len(selected) >= target_n:
            break
    if len(selected) < target_n:
        selected_set = set(selected)
        for idx in order:
            if int(idx) not in selected_set:
                selected.append(int(idx))
                if len(selected) >= target_n:
                    break
    return np.asarray(selected, dtype=np.int64)


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    side = pd.to_numeric(frame.get("side", pd.Series(dtype=float)), errors="coerce")
    rows = int(len(frame))
    long_rows = int((side > 0.0).sum()) if rows else 0
    short_rows = int((side < 0.0).sum()) if rows else 0
    return {
        "selected_rows": rows,
        "mean_u": _safe_mean(frame["u_policy_net"]) if rows else float("nan"),
        "median_u": _safe_quantile(frame["u_policy_net"], 0.50) if rows else float("nan"),
        "p10_u": _safe_quantile(frame["u_policy_net"], 0.10) if rows else float("nan"),
        "bad_mae_1r_rate": _safe_mean(frame["bad_mae_1r"]) if rows and "bad_mae_1r" in frame.columns else float("nan"),
        "timeout_rate": _safe_mean(frame["is_timeout"]) if rows and "is_timeout" in frame.columns else float("nan"),
        "clean_positive_rate": _safe_mean(frame["clean_positive"]) if rows and "clean_positive" in frame.columns else float("nan"),
        "dirty_positive_rate": _safe_mean(frame["dirty_positive"]) if rows and "dirty_positive" in frame.columns else float("nan"),
        "oracle_hit_rows": int(frame["oracle_top"].astype(bool).sum()) if rows and "oracle_top" in frame.columns else 0,
        "clean_oracle_hit_rows": int(frame["clean_oracle_top"].astype(bool).sum()) if rows and "clean_oracle_top" in frame.columns else 0,
        "long_rows": long_rows,
        "short_rows": short_rows,
        "max_side_share": (max(long_rows, short_rows) / float(rows)) if rows else float("nan"),
    }


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in monthly.groupby(["selector_variant", "policy", "objective", "block", "max_rank_shift", "keep_frac"], dropna=False):
        selector, policy, objective, block, max_rank_shift, keep_frac = keys
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        month_u = pd.to_numeric(group["mean_u"], errors="coerce")
        weights = selected_rows.where(selected_rows > 0.0)
        row = {
            "selector_variant": selector,
            "policy": policy,
            "objective": objective,
            "block": block,
            "max_rank_shift": float(max_rank_shift),
            "keep_frac": float(keep_frac),
            "periods": int(group["period"].nunique()),
            "selected_rows": int(selected_rows.sum()),
            "mean_u": float(np.average(month_u.fillna(0.0), weights=weights.fillna(0.0))) if float(weights.fillna(0.0).sum()) > 0.0 else float("nan"),
            "worst_month_mean_u": float(month_u.min(skipna=True)),
            "positive_months": int((month_u > 0.0).sum()),
            "bad_mae_1r_rate": float(np.average(pd.to_numeric(group["bad_mae_1r_rate"], errors="coerce").fillna(0.0), weights=weights.fillna(0.0))) if float(weights.fillna(0.0).sum()) > 0.0 else float("nan"),
            "timeout_rate": float(np.average(pd.to_numeric(group["timeout_rate"], errors="coerce").fillna(0.0), weights=weights.fillna(0.0))) if float(weights.fillna(0.0).sum()) > 0.0 else float("nan"),
            "clean_positive_rate": float(np.average(pd.to_numeric(group["clean_positive_rate"], errors="coerce").fillna(0.0), weights=weights.fillna(0.0))) if float(weights.fillna(0.0).sum()) > 0.0 else float("nan"),
            "dirty_positive_rate": float(np.average(pd.to_numeric(group["dirty_positive_rate"], errors="coerce").fillna(0.0), weights=weights.fillna(0.0))) if float(weights.fillna(0.0).sum()) > 0.0 else float("nan"),
            "oracle_hit_rows": int(pd.to_numeric(group["oracle_hit_rows"], errors="coerce").sum()),
            "clean_oracle_hit_rows": int(pd.to_numeric(group["clean_oracle_hit_rows"], errors="coerce").sum()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    baseline = out[out["policy"].eq("baseline")].copy()
    if not baseline.empty:
        base_cols = ["selector_variant", "keep_frac"]
        base = baseline.set_index(base_cols)
        for idx, row in out.iterrows():
            key = (row["selector_variant"], row["keep_frac"])
            if key not in base.index:
                continue
            b = base.loc[key]
            if isinstance(b, pd.DataFrame):
                b = b.iloc[0]
            for col in ("mean_u", "worst_month_mean_u", "bad_mae_1r_rate", "timeout_rate", "selected_rows", "oracle_hit_rows", "clean_oracle_hit_rows"):
                out.loc[idx, f"{col}_delta_vs_baseline"] = float(row[col]) - float(b[col])
    return out.sort_values(["mean_u", "worst_month_mean_u"], ascending=[False, False], na_position="last")


def run_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    blocks: list[str],
    objectives: list[str],
    keep_fracs: list[float],
    max_rank_shifts: list[float],
    max_side_share: float,
    base_score_col: str,
    max_features_per_block: int,
) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    if "period" not in frame.columns:
        raise ValueError("candidate ledger must contain period")
    if base_score_col not in frame.columns:
        raise ValueError(f"candidate ledger is missing base score column {base_score_col!r}")
    ctx_cols = [str(col) for col in frame.columns if str(col).startswith("ctx_")]
    periods = sorted(frame["period"].dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    for selector, selector_rows in frame.groupby("selector_variant", sort=False):
        for period in periods[1:]:
            train = selector_rows[selector_rows["period"] < period].copy()
            valid = selector_rows[selector_rows["period"].eq(period)].copy().reset_index(drop=True)
            if valid.empty:
                continue
            base_score = pd.to_numeric(valid[base_score_col], errors="coerce")
            base_rank = base_score.rank(method="average", pct=True).fillna(0.0)
            for keep_frac in keep_fracs:
                base_idx = _side_capped_indices(base_rank, valid["side"], keep_frac=keep_frac, max_side_share=max_side_share)
                selected = valid.iloc[base_idx] if len(base_idx) else valid.iloc[:0]
                monthly_rows.append(
                    {
                        "selector_variant": selector,
                        "period": period,
                        "policy": "baseline",
                        "objective": "none",
                        "block": "none",
                        "max_rank_shift": 0.0,
                        "keep_frac": float(keep_frac),
                        "feature_count": 0,
                        **_metrics(selected),
                    }
                )
                for objective_name in objectives:
                    if objective_name not in OBJECTIVES:
                        raise ValueError(f"unknown objective {objective_name!r}; expected one of {OBJECTIVES}")
                    for block in blocks:
                        cols = _block_columns(ctx_cols, block)
                        block_score = _fit_block_score(
                            train,
                            valid,
                            cols,
                            max_features=max_features_per_block,
                            objective_name=objective_name,
                        )
                        for max_shift in max_rank_shifts:
                            adjustment = (block_score.clip(-2.0, 2.0) / 2.0) * float(max_shift)
                            adjusted_rank = (base_rank + adjustment).clip(0.0, 1.0)
                            idx = _side_capped_indices(
                                adjusted_rank,
                                valid["side"],
                                keep_frac=keep_frac,
                                max_side_share=max_side_share,
                            )
                            selected = valid.iloc[idx] if len(idx) else valid.iloc[:0]
                            monthly_rows.append(
                                {
                                    "selector_variant": selector,
                                    "period": period,
                                    "policy": "rank_modulation",
                                    "objective": objective_name,
                                    "block": block,
                                    "max_rank_shift": float(max_shift),
                                    "keep_frac": float(keep_frac),
                                    "feature_count": int(len(cols)),
                                    **_metrics(selected),
                                }
                            )
    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = output_dir / "ae_gmm_modulation_monthly.csv"
    aggregate_path = output_dir / "ae_gmm_modulation_aggregate.csv"
    json_path = output_dir / "ae_gmm_modulation_diagnostics.json"
    md_path = output_dir / "ae_gmm_modulation_diagnostics.md"
    monthly.to_csv(monthly_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "rows": int(len(frame)),
        "base_score_col": str(base_score_col),
        "blocks": blocks,
        "objectives": objectives,
        "keep_fracs": keep_fracs,
        "max_rank_shifts": max_rank_shifts,
        "max_side_share": float(max_side_share),
        "max_features_per_block": int(max_features_per_block),
        "outputs": {"monthly_csv": str(monthly_path), "aggregate_csv": str(aggregate_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    display_cols = [
        "policy",
        "objective",
        "block",
        "max_rank_shift",
        "keep_frac",
        "selected_rows",
        "mean_u",
        "worst_month_mean_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "mean_u_delta_vs_baseline",
        "bad_mae_1r_rate_delta_vs_baseline",
        "timeout_rate_delta_vs_baseline",
        "oracle_hit_rows_delta_vs_baseline",
    ]
    lines = ["# AE/GMM Modulation Diagnostics", "", f"- Input: `{input_path}`", ""]
    if not aggregate.empty:
        lines.append(aggregate[[col for col in display_cols if col in aggregate.columns]].head(40).to_markdown(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", default="global,long,short,soft_prob,distance,transition,entropy,reconstruction")
    parser.add_argument("--objectives", default="utility,balanced,path_risk")
    parser.add_argument("--keep-fracs", default="0.50,0.60,0.70,0.80")
    parser.add_argument("--max-rank-shifts", default="0.01,0.02,0.05")
    parser.add_argument("--max-side-share", type=float, default=0.70)
    parser.add_argument("--base-score-col", default="selector_score")
    parser.add_argument("--max-features-per-block", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostics(
        input_path=args.input,
        output_dir=args.output_dir,
        blocks=[part.strip() for part in str(args.blocks).split(",") if part.strip()],
        objectives=[part.strip() for part in str(args.objectives).split(",") if part.strip()],
        keep_fracs=[float(part.strip()) for part in str(args.keep_fracs).split(",") if part.strip()],
        max_rank_shifts=[float(part.strip()) for part in str(args.max_rank_shifts).split(",") if part.strip()],
        max_side_share=float(args.max_side_share),
        base_score_col=str(args.base_score_col),
        max_features_per_block=int(args.max_features_per_block),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
