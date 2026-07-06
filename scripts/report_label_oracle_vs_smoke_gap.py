#!/usr/bin/env python3
"""Compare label-target oracle quality with cheap feature-smoke selections.

This is a pre-training diagnostic. It checks whether a label target is
economically sane when ranked directly, then compares that upper bound with the
rows selected by the causal feature-store smoke.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_labels,
    _path_metrics,
    _selection_metrics,
    _safe_mean,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_oracle_vs_smoke_gap_v1")
DEFAULT_LABEL_ARMS = (
    "S32_econ_limited_broad_policy",
    "S41_lowmae_timeout_safe_tail",
    "S43_lowbarrier_dirty_capped_broad",
    "S44_clean_masked_lowmae_rank",
    "S45_strict_clean_tail_rank",
    "S46_badmae_contrast_margin",
    "S47_dirty_capped_s41",
)


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    lowered = str(value).strip().lower()
    if lowered == "all":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _oracle_monthly(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    label_arms: list[str],
    top_fracs: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    for month in sorted(month_period.dropna().unique()):
        mask = month_period.eq(month)
        if int(mask.sum()) < 50:
            continue
        valid = frame.loc[mask].copy().reset_index(drop=True)
        valid_metrics = metrics.loc[mask].copy().reset_index(drop=True)
        for label_arm in label_arms:
            target = targets[label_arm].loc[mask].copy().reset_index(drop=True)
            score = target["target_soft"].copy().reset_index(drop=True)
            for top_frac in top_fracs:
                row = _selection_metrics(
                    frame=valid,
                    metrics=valid_metrics,
                    target=target,
                    score=score,
                    arm=f"oracle::{label_arm}",
                    selector="target_soft_oracle",
                    period=str(month),
                    top_frac=float(top_frac),
                )
                row["label_arm"] = label_arm
                rows.append(row)
    return pd.DataFrame(rows)


def _join_smoke(oracle: pd.DataFrame, smoke_dir: Path | None) -> pd.DataFrame:
    if smoke_dir is None:
        return pd.DataFrame()
    smoke_path = smoke_dir / "label_feature_store_model_smoke_monthly.csv"
    if not smoke_path.exists():
        raise FileNotFoundError(smoke_path)
    smoke = pd.read_csv(smoke_path)
    keys = ["period", "label_arm", "top_frac"]
    oracle_cols = [
        "period",
        "label_arm",
        "top_frac",
        "selected_rows",
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "top_symbol_share",
    ]
    smoke_cols = [
        "period",
        "label_arm",
        "weight_arm",
        "top_frac",
        "selected_rows",
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "top_symbol_share",
        "score_ic_u",
        "score_ic_label",
    ]
    joined = smoke[[col for col in smoke_cols if col in smoke.columns]].merge(
        oracle[[col for col in oracle_cols if col in oracle.columns]].rename(
            columns={
                "selected_rows": "oracle_selected_rows",
                "mean_u": "oracle_mean_u",
                "hit_u": "oracle_hit_u",
                "q10_u": "oracle_q10_u",
                "bad_mae_1r_rate": "oracle_bad_mae_1r_rate",
                "wide_barrier_25bps_rate": "oracle_wide_barrier_25bps_rate",
                "timeout_rate": "oracle_timeout_rate",
                "top_symbol_share": "oracle_top_symbol_share",
            }
        ),
        on=keys,
        how="left",
    )
    joined["mean_u_gap_vs_oracle"] = joined["mean_u"] - joined["oracle_mean_u"]
    joined["bad_mae_gap_vs_oracle"] = joined["bad_mae_1r_rate"] - joined["oracle_bad_mae_1r_rate"]
    return joined


def _aggregate_gap(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in joined.groupby(["label_arm", "weight_arm", "top_frac"], dropna=False, observed=True):
        label_arm, weight_arm, top_frac = key
        rows.append(
            {
                "label_arm": label_arm,
                "weight_arm": weight_arm,
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((pd.to_numeric(group["mean_u"], errors="coerce") > 0.0).sum()),
                "oracle_mean_u": _safe_mean(group["oracle_mean_u"]),
                "smoke_mean_u": _safe_mean(group["mean_u"]),
                "mean_u_gap_vs_oracle": _safe_mean(group["mean_u_gap_vs_oracle"]),
                "oracle_bad_mae_1r_rate": _safe_mean(group["oracle_bad_mae_1r_rate"]),
                "smoke_bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "bad_mae_gap_vs_oracle": _safe_mean(group["bad_mae_gap_vs_oracle"]),
                "oracle_wide_barrier_25bps_rate": _safe_mean(group["oracle_wide_barrier_25bps_rate"]),
                "smoke_wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "oracle_timeout_rate": _safe_mean(group["oracle_timeout_rate"]),
                "smoke_timeout_rate": _safe_mean(group["timeout_rate"]),
                "score_ic_u": _safe_mean(group.get("score_ic_u", pd.Series(dtype=float))),
                "score_ic_label": _safe_mean(group.get("score_ic_label", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_frac", "smoke_mean_u", "mean_u_gap_vs_oracle"],
        ascending=[True, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    oracle: pd.DataFrame,
    joined_aggregate: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_oracle_vs_smoke_gap.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    oracle_cols = [
        "period",
        "label_arm",
        "top_frac",
        "selected_rows",
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "top_symbol_share",
    ]
    gap_cols = [
        "label_arm",
        "weight_arm",
        "top_frac",
        "months",
        "positive_months",
        "oracle_mean_u",
        "smoke_mean_u",
        "mean_u_gap_vs_oracle",
        "oracle_bad_mae_1r_rate",
        "smoke_bad_mae_1r_rate",
        "bad_mae_gap_vs_oracle",
        "oracle_wide_barrier_25bps_rate",
        "smoke_wide_barrier_25bps_rate",
        "oracle_timeout_rate",
        "smoke_timeout_rate",
        "score_ic_u",
        "score_ic_label",
    ]
    lines = [
        "# Label Oracle Vs Smoke Gap",
        "",
        "Scope: pre-training diagnostic. Oracle rows are selected by the label target itself; smoke rows are selected by the cheap causal feature-store model.",
        "",
        "## Oracle Top Rows",
        "",
        table(
            oracle.sort_values(["top_frac", "period", "mean_u"], ascending=[True, True, False]),
            oracle_cols,
            limit=80,
        ),
        "",
        "## Smoke Gap",
        "",
        table(joined_aggregate, gap_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Oracle monthly: `{manifest['outputs']['oracle_monthly']}`",
        f"- Joined monthly: `{manifest['outputs']['joined_monthly']}`",
        f"- Joined aggregate: `{manifest['outputs']['joined_aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    output_dir: Path,
    smoke_dir: Path | None,
    label_arms: list[str],
    top_fracs: list[float],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    metrics = _path_metrics(frame)
    targets = _label_targets(frame, metrics)
    if not label_arms:
        label_arms = list(LABEL_ARMS)
    missing = sorted(set(label_arms) - set(targets))
    if missing:
        raise ValueError(f"Unknown label arms: {missing}")

    oracle = _oracle_monthly(
        frame=frame,
        metrics=metrics,
        targets=targets,
        label_arms=label_arms,
        top_fracs=top_fracs,
    )
    joined = _join_smoke(oracle, smoke_dir)
    joined_aggregate = _aggregate_gap(joined)

    paths = {
        "oracle_monthly": output_dir / "label_oracle_monthly.csv",
        "joined_monthly": output_dir / "label_oracle_vs_smoke_joined_monthly.csv",
        "joined_aggregate": output_dir / "label_oracle_vs_smoke_joined_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    oracle.to_csv(paths["oracle_monthly"], index=False)
    joined.to_csv(paths["joined_monthly"], index=False)
    joined_aggregate.to_csv(paths["joined_aggregate"], index=False)
    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "smoke_dir": str(smoke_dir) if smoke_dir is not None else "",
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "label_arms": label_arms,
        "top_fracs": [float(v) for v in top_fracs],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        oracle=oracle,
        joined_aggregate=joined_aggregate,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke-dir", type=Path, default=None)
    parser.add_argument(
        "--label-arms",
        type=str,
        default=",".join(DEFAULT_LABEL_ARMS),
        help="Comma-separated label arms, or 'all'.",
    )
    parser.add_argument("--top-fracs", type=str, default="0.0025,0.005,0.01")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        smoke_dir=args.smoke_dir,
        label_arms=_parse_csv(args.label_arms, DEFAULT_LABEL_ARMS),
        top_fracs=_parse_float_csv(args.top_fracs, (0.0025, 0.005, 0.01)),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
