#!/usr/bin/env python3
"""Audit whether prior stability features predict next-month cell effects.

This script evaluates the explicit stability feature idea at the cell level.
It uses the month-cell promoted-vs-baseline diagnostics and tests whether
strictly prior cell effects have persistence into the next month.

It is diagnostic only.  It does not create deployable gates and does not tune
row-level train_meta on the evaluated month.
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

from scripts.audit_promoted_cross_asset_month_flip_attribution import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_FLIP_AUDIT_DIR,
    _json_safe,
)
from scripts.materialize_cross_asset_stability_meta_handoff import (  # noqa: E402
    CELL_COLUMNS,
    PRIOR_METRICS,
    STABILITY_KEEP_FRACS,
)


DEFAULT_MONTH_CELLS = DEFAULT_FLIP_AUDIT_DIR / "promoted_cross_asset_month_cell_effects.csv"
DEFAULT_OUT_DIR = DEFAULT_FLIP_AUDIT_DIR.parent / "cross_asset_stability_prior_persistence_v1"


def _num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _corr(x: pd.Series, y: pd.Series, *, method: str) -> float:
    pair = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return float(pair["x"].corr(pair["y"], method=method))


def _rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(mask.astype(bool).mean())


def _precision_recall(pred: pd.Series, actual: pd.Series) -> dict[str, float]:
    pred_b = pred.fillna(False).astype(bool)
    actual_b = actual.fillna(False).astype(bool)
    tp = int((pred_b & actual_b).sum())
    fp = int((pred_b & ~actual_b).sum())
    fn = int((~pred_b & actual_b).sum())
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _prior_rows(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cells = cells.copy()
    cells["month"] = cells["month"].astype(str)
    months = sorted(cells["month"].dropna().unique())
    key_cols = [*CELL_COLUMNS, "keep_frac"]
    for month in months[1:]:
        history_months = [m for m in months if m < month]
        history = cells[cells["month"].isin(history_months)]
        current = cells[cells["month"].eq(month)]
        if history.empty or current.empty:
            continue
        hist = (
            history.groupby(key_cols, dropna=False)
            .agg(
                prior_months=("month", "nunique"),
                prior_effect_mean=("effect_value_score", "mean"),
                prior_effect_last=("effect_value_score", "last"),
                prior_effect_std=("effect_value_score", lambda x: float(pd.to_numeric(x, errors="coerce").std(ddof=0) or 0.0)),
                prior_delta_ev_mean=("delta_ev_after_1pct", "mean"),
                prior_delta_bad_mae_mean=("delta_full_path_bad_mae", "mean"),
                prior_delta_timeout_mean=("delta_timeout", "mean"),
                prior_beneficial_rate=("promoted_beneficial", "mean"),
                prior_damaged_rate=("promoted_damaged", "mean"),
            )
            .reset_index()
        )
        merged = current.merge(hist, on=key_cols, how="inner")
        if merged.empty:
            continue
        merged["eval_month"] = month
        merged["history_months"] = ",".join(history_months)
        rows.extend(merged.to_dict("records"))
    return pd.DataFrame(rows)


def _summary_rows(prior: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if prior.empty:
        return pd.DataFrame()
    for keep_frac, group in prior.groupby("keep_frac", dropna=False):
        supported = group[group["support_pass"].astype(bool)].copy() if "support_pass" in group.columns else group.copy()
        if supported.empty:
            supported = group.copy()
        actual_positive = _num(supported["effect_value_score"]).gt(0.35)
        actual_negative = _num(supported["effect_value_score"]).lt(-0.35)
        pred_positive = _num(supported["prior_effect_mean"]).gt(0.35)
        pred_negative = _num(supported["prior_effect_mean"]).lt(-0.35)
        positive = _precision_recall(pred_positive, actual_positive)
        negative = _precision_recall(pred_negative, actual_negative)
        prior_sign = np.sign(_num(supported["prior_effect_mean"]).fillna(0.0))
        current_sign = np.sign(_num(supported["effect_value_score"]).fillna(0.0))
        sign_known = prior_sign.ne(0) & current_sign.ne(0)
        false_positive = pred_positive & actual_negative
        false_negative = pred_negative & actual_positive
        rows.append(
            {
                "keep_frac": float(keep_frac),
                "evaluated_cells": int(len(supported)),
                "eval_months": int(supported["eval_month"].astype(str).nunique()),
                "pearson_prior_current_effect": _corr(supported["prior_effect_mean"], supported["effect_value_score"], method="pearson"),
                "spearman_prior_current_effect": _corr(supported["prior_effect_mean"], supported["effect_value_score"], method="spearman"),
                "pearson_prior_delta_ev_current_delta_ev": _corr(
                    supported["prior_delta_ev_mean"], supported["delta_ev_after_1pct"], method="pearson"
                ),
                "sign_accuracy": _rate(prior_sign[sign_known].eq(current_sign[sign_known])) if bool(sign_known.any()) else float("nan"),
                "prior_positive_precision": positive["precision"],
                "prior_positive_recall": positive["recall"],
                "prior_positive_false_positive_cells": int(positive["fp"]),
                "prior_negative_precision": negative["precision"],
                "prior_negative_recall": negative["recall"],
                "prior_negative_false_negative_cells": int(false_negative.sum()),
                "positive_to_negative_cells": int(false_positive.sum()),
                "actual_positive_rate": _rate(actual_positive),
                "actual_negative_rate": _rate(actual_negative),
                "prior_positive_rate": _rate(pred_positive),
                "prior_negative_rate": _rate(pred_negative),
                "mean_current_effect_when_prior_positive": float(_num(supported.loc[pred_positive, "effect_value_score"]).mean())
                if bool(pred_positive.any())
                else float("nan"),
                "mean_current_effect_when_prior_negative": float(_num(supported.loc[pred_negative, "effect_value_score"]).mean())
                if bool(pred_negative.any())
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("keep_frac")


def _worst_rows(prior: pd.DataFrame) -> pd.DataFrame:
    if prior.empty:
        return pd.DataFrame()
    supported = prior[prior["support_pass"].astype(bool)].copy() if "support_pass" in prior.columns else prior.copy()
    supported["effect_delta"] = _num(supported["effect_value_score"]) - _num(supported["prior_effect_mean"])
    supported["prior_positive_current_negative"] = _num(supported["prior_effect_mean"]).gt(0.35) & _num(
        supported["effect_value_score"]
    ).lt(-0.35)
    cols = [
        "eval_month",
        "history_months",
        "keep_frac",
        "side_name",
        "source_semantic_family",
        "rows",
        "prior_effect_mean",
        "effect_value_score",
        "effect_delta",
        "prior_delta_ev_mean",
        "delta_ev_after_1pct",
        "prior_delta_bad_mae_mean",
        "delta_full_path_bad_mae",
        "prior_positive_current_negative",
    ]
    return supported.sort_values(["prior_positive_current_negative", "effect_delta"], ascending=[False, True])[
        [col for col in cols if col in supported.columns]
    ].head(40)


def _status(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "no_prior_evidence"
    keep10 = summary[summary["keep_frac"].round(6).eq(0.10)]
    row = keep10.iloc[0] if not keep10.empty else summary.iloc[0]
    spearman = float(row.get("spearman_prior_current_effect", np.nan))
    sign_acc = float(row.get("sign_accuracy", np.nan))
    false_pos = int(row.get("positive_to_negative_cells", 0) or 0)
    if math.isfinite(spearman) and spearman > 0.25 and math.isfinite(sign_acc) and sign_acc >= 0.60 and false_pos <= 2:
        return "prior_stability_promising"
    if math.isfinite(spearman) and spearman < 0.0:
        return "prior_stability_unreliable_or_inverted"
    return "prior_stability_weak_or_insufficient"


def _write_markdown(out_dir: Path, manifest: dict[str, Any], summary: pd.DataFrame, worst: pd.DataFrame) -> Path:
    lines = [
        "# Cross-Asset Stability Prior Persistence",
        "",
        "## Verdict",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- evaluated prior rows: `{manifest.get('evaluated_prior_rows')}`",
        f"- source month-cell file: `{manifest.get('month_cells_path')}`",
        "",
        "## Summary",
        "",
    ]
    lines.append(summary.to_markdown(index=False) if not summary.empty else "_No prior rows._")
    lines.extend(["", "## Worst Prior-Positive Failures", ""])
    lines.append(worst.to_markdown(index=False) if not worst.empty else "_No failure rows._")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit tests persistence of prior side x archetype effects. Strong persistence would support explicit stability features as train_meta context once more OOF months are available.",
            "Weak or inverted persistence means prior-cell diagnostics should be treated as uncertainty/context, not as direct confidence or policy gates.",
        ]
    )
    path = out_dir / "cross_asset_stability_prior_persistence.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_audit(*, month_cells_path: Path, out_dir: Path) -> dict[str, Any]:
    if not month_cells_path.exists():
        raise FileNotFoundError(month_cells_path)
    cells = pd.read_csv(month_cells_path)
    required = {"month", "keep_frac", *CELL_COLUMNS, *PRIOR_METRICS, "support_pass", "promoted_beneficial", "promoted_damaged"}
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise ValueError(f"Month-cell diagnostics missing required columns: {missing}")
    prior = _prior_rows(cells)
    summary = _summary_rows(prior)
    worst = _worst_rows(prior)
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_path = out_dir / "cross_asset_stability_prior_rows.csv"
    summary_path = out_dir / "cross_asset_stability_prior_persistence_summary.csv"
    worst_path = out_dir / "cross_asset_stability_prior_persistence_worst.csv"
    prior.to_csv(prior_path, index=False)
    summary.to_csv(summary_path, index=False)
    worst.to_csv(worst_path, index=False)
    manifest = {
        "generated_by": "audit_cross_asset_stability_prior_persistence",
        "month_cells_path": str(month_cells_path),
        "evaluated_prior_rows": int(len(prior)),
        "status": _status(summary),
        "summary": _json_safe(summary.to_dict("records")),
        "leakage_contract": {
            "source": "OOF month-cell baseline-vs-promoted diagnostics",
            "prior_rule": "current month is evaluated only against strictly earlier month-cell diagnostics",
            "deployment_use": "diagnostic evidence for stability context features, not a deployable gate",
        },
        "outputs": {
            "prior_rows": str(prior_path),
            "summary": str(summary_path),
            "worst": str(worst_path),
            "json": str(out_dir / "cross_asset_stability_prior_persistence.json"),
            "markdown": str(out_dir / "cross_asset_stability_prior_persistence.md"),
        },
    }
    markdown = _write_markdown(out_dir, manifest, summary, worst)
    manifest["outputs"]["markdown"] = str(markdown)
    (out_dir / "cross_asset_stability_prior_persistence.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month-cells-path", type=Path, default=DEFAULT_MONTH_CELLS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_audit(month_cells_path=args.month_cells_path, out_dir=args.out_dir)
    print(json.dumps(_json_safe({"event": "cross_asset_stability_prior_persistence_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
