#!/usr/bin/env python3
"""S52 OOF consensus/path-quality selector diagnostic.

This report asks whether existing OOF path-aware scores can reduce selected
full-path bad-MAE when used as an admission layer before a primary opportunity
score. It preserves top-k exposure by keeping rejected rows finite but ranked
below admitted rows, so top10 remains top10 of the original row universe when
the admission fraction is at least 10%.
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

from scripts.report_s52_score_blend_ablation import (  # noqa: E402
    DEFAULT_LEDGER,
    _blend_scores,
    _evaluate_score,
    _json_safe,
    _normalization_groups,
    _parse_weights,
    _safe_zscore,
    _wide_scores,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/s52_consensus_selector_learnability_features_noae_20260705_v1"
)
DEFAULT_GATE_FRACTIONS = "0.15,0.20,0.30,0.40,0.50,0.70"
DEFAULT_GATE_SCORES = (
    "ranker_timestamp_side_fullpath_evpath",
    "ranker_timestamp_side_soft_ordered_ev",
)


def _parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"value must be in (0, 1]: {value}")
        out.append(value)
    return sorted(set(out))


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _top_fraction_mask(score: pd.Series, fraction: float, groups: pd.Series | None) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce").astype(float)
    mask = pd.Series(False, index=values.index)
    if groups is None:
        valid = values[np.isfinite(values.to_numpy(dtype=np.float64))]
        if len(valid) == 0:
            return mask
        threshold = valid.rank(method="first", ascending=False)
        keep = max(1, int(math.ceil(float(fraction) * len(valid))))
        mask.loc[valid.index] = threshold.le(keep).to_numpy(dtype=bool)
        return mask
    for _group_key, idx in values.groupby(groups, observed=True, dropna=False).groups.items():
        group_values = values.loc[idx]
        valid = group_values[np.isfinite(group_values.to_numpy(dtype=np.float64))]
        if len(valid) == 0:
            continue
        keep = max(1, int(math.ceil(float(fraction) * len(valid))))
        ranks = valid.rank(method="first", ascending=False)
        mask.loc[valid.index] = ranks.le(keep).to_numpy(dtype=bool)
    return mask


def _admitted_score(primary: pd.Series, gate: pd.Series, *, gate_fraction: float, groups: pd.Series | None) -> tuple[pd.Series, pd.Series]:
    primary_z = _safe_zscore(primary, groups)
    gate_z = _safe_zscore(gate, groups)
    admit = _top_fraction_mask(gate_z, float(gate_fraction), groups)
    penalty = max(10.0, float(np.nanmax(np.abs(primary_z.to_numpy(dtype=np.float64)))) + 10.0)
    score = primary_z.where(admit, primary_z - penalty)
    return score.astype(float), admit


def _summarize_admission(base: pd.DataFrame, admit: pd.Series, name: str, gate_fraction: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": name,
        "gate_fraction": float(gate_fraction),
        "admitted_rows": int(admit.sum()),
        "admitted_share": float(admit.mean()) if len(admit) else float("nan"),
    }
    tmp = base.assign(__admit__=admit.to_numpy(dtype=bool))
    for (month, side), group in tmp.groupby(["month", "side_name"], observed=True, dropna=False):
        key = f"admit_share_{month}_{side}".replace("-", "")
        row[key] = float(group["__admit__"].mean()) if len(group) else float("nan")
    return row


def _write_report(output_dir: Path, summary: pd.DataFrame, folds: pd.DataFrame, admission: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def fmt(df: pd.DataFrame, cols: list[str], n: int = 40) -> str:
        if df.empty:
            return "No rows."
        view = df[[col for col in cols if col in df.columns]].head(n).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    top_cols = [
        "variant",
        "objective",
        "gate_score",
        "gate_fraction",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_first_pass_good_rate",
        "mean_top10_first_pass_bad_rate",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_p90_first_touch_full_path_mae_norm",
        "mean_top10_mfe_1r_before_mae_1r_rate",
        "mean_top10_mae_1r_before_mfe_1r_rate",
        "mean_top10_timeout_rate",
        "mean_long_top10_mean_first_touch_net",
        "mean_short_top10_mean_first_touch_net",
    ]
    fold_cols = [
        "variant",
        "month",
        "top10_ev_weighted_first_touch_precision",
        "top10_mean_first_touch_net",
        "top10_first_touch_full_path_bad_mae_1r_rate",
        "top10_p90_first_touch_full_path_mae_norm",
        "top10_timeout_rate",
    ]
    admission_cols = ["variant", "gate_fraction", "admitted_rows", "admitted_share"]
    lines = [
        "# S52 Consensus Selector",
        "",
        "This is an OOF diagnostic. It ranks with a primary score after admitting rows with a path-aware score.",
        "",
        f"Ledger: `{manifest['ledger']}`",
        f"Rows: `{manifest['rows']}`",
        f"Normalization: `{manifest['normalization']}`",
        f"Round-trip cost: `{manifest['round_trip_cost']:.6f}`",
        "",
        "## Best Rows",
        "",
        fmt(summary.sort_values("objective", ascending=False), top_cols, n=40),
        "",
        "## Fold Metrics For Top 10 Variants",
        "",
        fmt(folds[folds["variant"].isin(summary.sort_values("objective", ascending=False)["variant"].head(10))], fold_cols, n=200),
        "",
        "## Admission Shares",
        "",
        fmt(admission.sort_values("admitted_share"), admission_cols, n=40),
        "",
    ]
    output_dir.joinpath("s52_consensus_selector.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    ledger_path: Path,
    output_dir: Path,
    round_trip_cost: float,
    weights: list[float],
    normalization: str,
    gate_fractions: list[float],
    gate_scores: list[str],
    max_primary_variants: int,
) -> None:
    ledger = pd.read_parquet(ledger_path)
    base, score_frame = _wide_scores(ledger)
    blends = _blend_scores(base, score_frame, weights=weights, normalization=normalization)
    groups = _normalization_groups(base, normalization)

    primary_summaries: list[dict[str, Any]] = []
    for name, score in blends.items():
        summary, _rows = _evaluate_score(base, name, score, round_trip_cost=float(round_trip_cost))
        primary_summaries.append(summary)
    primary_rank = pd.DataFrame(primary_summaries).sort_values("objective", ascending=False)
    primary_names = [str(v) for v in primary_rank["variant"].head(int(max_primary_variants)).tolist()]

    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    admission_rows: list[dict[str, Any]] = []
    for primary_name in primary_names:
        primary_score = blends[primary_name]
        for gate_name in gate_scores:
            if gate_name not in score_frame.columns:
                continue
            gate_score = score_frame[gate_name]
            for fraction in gate_fractions:
                score, admit = _admitted_score(primary_score, gate_score, gate_fraction=float(fraction), groups=groups)
                name = f"consensus::{primary_name}__gate_{gate_name}_top{int(round(fraction * 100)):02d}"
                summary, rows = _evaluate_score(base, name, score, round_trip_cost=float(round_trip_cost))
                summary["gate_score"] = gate_name
                summary["gate_fraction"] = float(fraction)
                summary["primary_variant"] = primary_name
                summary["normalization_scope"] = str(normalization)
                summaries.append(summary)
                for row in rows:
                    row["gate_score"] = gate_name
                    row["gate_fraction"] = float(fraction)
                    row["primary_variant"] = primary_name
                    fold_rows.append(row)
                admission_rows.append(_summarize_admission(base, admit, name, float(fraction)))

    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)
    admission_df = pd.DataFrame(admission_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "s52_consensus_selector_summary.csv"
    folds_path = output_dir / "s52_consensus_selector_folds.csv"
    admission_path = output_dir / "s52_consensus_selector_admission.csv"
    manifest_path = output_dir / "manifest.json"
    summary_df.to_csv(summary_path, index=False)
    folds_df.to_csv(folds_path, index=False)
    admission_df.to_csv(admission_path, index=False)
    manifest = {
        "ledger": str(ledger_path),
        "output_dir": str(output_dir),
        "rows": int(len(base)),
        "round_trip_cost": float(round_trip_cost),
        "normalization": str(normalization),
        "weights": [float(w) for w in weights],
        "gate_fractions": [float(v) for v in gate_fractions],
        "gate_scores": list(gate_scores),
        "primary_variants": primary_names,
        "outputs": {
            "summary": str(summary_path),
            "folds": str(folds_path),
            "admission": str(admission_path),
            "report": str(output_dir / "s52_consensus_selector.md"),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_dir, summary_df, folds_df, admission_df, manifest)
    print(f"wrote {summary_path}")
    cols = [
        "variant",
        "objective",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_p90_first_touch_full_path_mae_norm",
        "mean_top10_timeout_rate",
    ]
    print(summary_df[cols].head(12).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--round-trip-cost", type=float, default=0.01)
    parser.add_argument("--weights", default="0,0.25,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument(
        "--normalization",
        choices=("global", "month", "month_side", "timestamp_side"),
        default="global",
    )
    parser.add_argument("--gate-fractions", default=DEFAULT_GATE_FRACTIONS)
    parser.add_argument("--gate-scores", default=",".join(DEFAULT_GATE_SCORES))
    parser.add_argument("--max-primary-variants", type=int, default=8)
    args = parser.parse_args()
    run(
        ledger_path=args.ledger,
        output_dir=args.output_dir,
        round_trip_cost=float(args.round_trip_cost),
        weights=_parse_weights(args.weights),
        normalization=str(args.normalization),
        gate_fractions=_parse_floats(args.gate_fractions),
        gate_scores=_parse_csv(args.gate_scores),
        max_primary_variants=int(args.max_primary_variants),
    )


if __name__ == "__main__":
    main()
