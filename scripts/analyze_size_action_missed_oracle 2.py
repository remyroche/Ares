#!/usr/bin/env python3
"""Diagnose size-action oracle interventions missed by a learned selector.

The diagnostic compares three strategy-timestamp populations for a selected
arm:

* selected_positive: selector intervened and the realized counterfactual delta
  was positive;
* missed_oracle_positive: selector did not intervene although the exact-state
  oracle found a positive non-baseline action;
* non_actionable: neither selected nor missed-oracle-positive.

Only live-available group features from the exact panel are used for feature
separation. Counterfactual labels and action-outcome fields are excluded by the
same feature-contract forbidden list used by the production scorer audit.
"""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_size_action_feature_contract import FORBIDDEN_FEATURES


DEFAULT_ARM = "C3ed_bagged_safety_c3ea_or_pressure_opportunity_action_tail_veto_union_gate"

GROUP_KEYS = ["fold_id", "timestamp", "strategy_id"]

SCORE_COLUMNS = [
    "selection_score",
    "p_intervene",
    "pred_delta_J",
    "cal_mean_delta_J",
    "cal_lcb_mean_delta_J",
    "cal_q25_delta_J",
    "cal_positive_rate",
    "p_action_positive",
    "p_action_value_positive",
    "p_action_economic_positive",
    "ranker_score",
    "ranker_score_margin",
    "pred_delta_margin",
    "eligible_action",
    "calibration_bin_n",
]

EXPLICIT_NON_FEATURE_COLUMNS = {
    *GROUP_KEYS,
    "split",
    "multiplier",
    "action_binds",
}

FORBIDDEN_SUBSTRINGS = (
    "delta_",
    "best_",
    "oracle_",
    "selected_delta",
    "zero_cut",
)


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _is_forbidden_feature(column: str) -> bool:
    if column in EXPLICIT_NON_FEATURE_COLUMNS:
        return True
    if column in FORBIDDEN_FEATURES:
        return True
    if column.endswith("_group"):
        base = column[: -len("_group")]
        if base in FORBIDDEN_FEATURES:
            return True
    lowered = column.lower()
    return any(token in lowered for token in FORBIDDEN_SUBSTRINGS)


def live_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Return numeric, live-clean columns from the exact-state panel."""

    columns: list[str] = []
    for column in panel.columns:
        if _is_forbidden_feature(str(column)):
            continue
        if pd.api.types.is_numeric_dtype(panel[column]):
            columns.append(str(column))
    return columns


def _baseline_group_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = live_feature_columns(panel)
    if "multiplier" in panel.columns:
        mult = pd.to_numeric(panel["multiplier"], errors="coerce")
        baseline = panel.loc[np.isclose(mult, 1.0, rtol=0.0, atol=1e-9)].copy()
        if baseline.empty:
            baseline = panel.copy()
    else:
        baseline = panel.copy()
    keep = GROUP_KEYS + feature_cols
    missing = [col for col in GROUP_KEYS if col not in baseline.columns]
    if missing:
        raise ValueError(f"panel is missing group keys: {missing}")
    frame = baseline[keep].copy()
    frame["timestamp"] = frame["timestamp"].astype(str)
    # Multiple baseline rows should not exist, but mean aggregation makes the
    # diagnostic robust to duplicated source rows without using labels.
    grouped = frame.groupby(GROUP_KEYS, as_index=False)[feature_cols].mean(numeric_only=True)
    return grouped, feature_cols


def _arm_diagnostics(diag: pd.DataFrame, arm: str) -> pd.DataFrame:
    if "arm" not in diag.columns:
        raise ValueError("selector diagnostics is missing 'arm'")
    subset = diag.loc[diag["arm"].astype(str) == arm].copy()
    if subset.empty:
        available = sorted(diag["arm"].dropna().astype(str).unique().tolist())
        raise ValueError(f"arm {arm!r} not found; available arms: {available[:20]}")
    for key in GROUP_KEYS:
        if key not in subset.columns:
            raise ValueError(f"selector diagnostics is missing group key {key!r}")
    subset["timestamp"] = subset["timestamp"].astype(str)
    selected = subset.get("selected", False)
    selected_bool = selected.astype(bool) if isinstance(selected, pd.Series) else bool(selected)
    selected_delta = pd.to_numeric(subset.get("selected_delta_full_J", np.nan), errors="coerce")
    missed = subset.get("missed_positive_oracle", False)
    missed_bool = missed.astype(bool) if isinstance(missed, pd.Series) else bool(missed)
    oracle_delta = pd.to_numeric(subset.get("oracle_best_delta_full_J", np.nan), errors="coerce")
    oracle_multiplier = pd.to_numeric(subset.get("oracle_best_multiplier", np.nan), errors="coerce")
    missed_oracle = missed_bool | (
        (~selected_bool)
        & (oracle_delta > 0.0)
        & (oracle_multiplier.notna())
        & (oracle_multiplier < 1.0)
    )
    subset["population"] = "non_actionable"
    subset.loc[missed_oracle, "population"] = "missed_oracle_positive"
    subset.loc[selected_bool & (selected_delta > 0.0), "population"] = "selected_positive"
    return subset


def _augment_with_oracle_action_scores(run_dir: Path, arm_diag: pd.DataFrame) -> pd.DataFrame:
    """Attach scored non-baseline action diagnostics for missed oracle groups.

    Selector schedules often only persist scores for selected rows. For missed
    positives, that makes bottleneck reports show zeros even though the exact
    action-score ledger contains live-clean Stage-1 and action-value scores.
    This diagnostic-only merge uses the oracle-best non-baseline multiplier as
    the action to inspect, so it can explain which model gate suppressed the
    action without changing any training or policy behavior.
    """

    score_path = run_dir / "size_action_eval_action_scores.csv"
    if not score_path.exists() or arm_diag.empty:
        return arm_diag
    scores = pd.read_csv(score_path)
    required = set(GROUP_KEYS + ["multiplier"])
    if not required.issubset(scores.columns):
        return arm_diag
    out = arm_diag.copy()
    out["timestamp"] = out["timestamp"].astype(str)
    out["diagnostic_action_multiplier"] = pd.to_numeric(out.get("selected_multiplier"), errors="coerce")
    missed_mask = out["population"].eq("missed_oracle_positive")
    out.loc[missed_mask, "diagnostic_action_multiplier"] = pd.to_numeric(
        out.loc[missed_mask].get("oracle_best_multiplier"), errors="coerce"
    )
    out["diagnostic_action_multiplier"] = out["diagnostic_action_multiplier"].fillna(1.0).astype(float)
    action_scores = scores.copy()
    action_scores["timestamp"] = action_scores["timestamp"].astype(str)
    action_scores["strategy_id"] = action_scores["strategy_id"].astype(str)
    action_scores["multiplier"] = pd.to_numeric(action_scores["multiplier"], errors="coerce").fillna(1.0).astype(float)
    score_cols = [
        c
        for c in SCORE_COLUMNS
        if c in action_scores.columns and c not in {"selection_score", "eligible_action"}
    ]
    keep = GROUP_KEYS + ["multiplier"] + score_cols
    action_scores = action_scores[keep].drop_duplicates(GROUP_KEYS + ["multiplier"])
    action_scores = action_scores.rename(columns={"multiplier": "diagnostic_action_multiplier"})
    rename = {c: f"oracle_action_{c}" for c in score_cols}
    action_scores = action_scores.rename(columns=rename)
    out = out.merge(
        action_scores,
        on=GROUP_KEYS + ["diagnostic_action_multiplier"],
        how="left",
        validate="many_to_one",
    )
    for col in score_cols:
        oracle_col = f"oracle_action_{col}"
        if oracle_col not in out.columns:
            continue
        if col not in out.columns:
            out[col] = np.nan
        fill_mask = out[col].isna() | (missed_mask & (pd.to_numeric(out[col], errors="coerce").fillna(0.0) == 0.0))
        out.loc[fill_mask, col] = out.loc[fill_mask, oracle_col]
    return out


def _standardized_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(b, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if a.empty or b.empty:
        return float("nan")
    pooled = math.sqrt((float(a.var(ddof=0)) + float(b.var(ddof=0))) / 2.0)
    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _feature_population_summary(merged: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, group in merged.groupby("population", dropna=False):
        row: dict[str, Any] = {
            "population": population,
            "groups": int(len(group)),
            "folds": int(group["fold_id"].nunique()) if "fold_id" in group.columns else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("population").reset_index(drop=True)


def _feature_differences(merged: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = merged.loc[merged["population"] == "selected_positive"]
    missed = merged.loc[merged["population"] == "missed_oracle_positive"]
    noop = merged.loc[merged["population"] == "non_actionable"]
    for feature in feature_cols:
        values = pd.to_numeric(merged[feature], errors="coerce")
        if values.notna().sum() == 0:
            continue
        rows.append(
            {
                "feature": feature,
                "overall_missing_share": float(values.isna().mean()),
                "selected_mean": float(pd.to_numeric(selected.get(feature), errors="coerce").mean())
                if not selected.empty
                else float("nan"),
                "missed_mean": float(pd.to_numeric(missed.get(feature), errors="coerce").mean())
                if not missed.empty
                else float("nan"),
                "non_actionable_mean": float(pd.to_numeric(noop.get(feature), errors="coerce").mean())
                if not noop.empty
                else float("nan"),
                "missed_vs_non_actionable_std_diff": _standardized_difference(
                    missed.get(feature, pd.Series(dtype=float)),
                    noop.get(feature, pd.Series(dtype=float)),
                ),
                "selected_vs_missed_std_diff": _standardized_difference(
                    selected.get(feature, pd.Series(dtype=float)),
                    missed.get(feature, pd.Series(dtype=float)),
                ),
                "selected_vs_non_actionable_std_diff": _standardized_difference(
                    selected.get(feature, pd.Series(dtype=float)),
                    noop.get(feature, pd.Series(dtype=float)),
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_missed_vs_non_actionable_std_diff"] = out[
        "missed_vs_non_actionable_std_diff"
    ].abs()
    out["abs_selected_vs_missed_std_diff"] = out["selected_vs_missed_std_diff"].abs()
    return out.sort_values(
        ["abs_missed_vs_non_actionable_std_diff", "abs_selected_vs_missed_std_diff"],
        ascending=False,
    ).reset_index(drop=True)


def _mann_whitney_auc(values: pd.Series, labels: pd.Series) -> float:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "label": labels.astype(int)})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    n_pos = int(frame["label"].sum())
    n_neg = int(len(frame) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = frame["value"].rank(method="average")
    sum_ranks_pos = float(ranks.loc[frame["label"] == 1].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _feature_auc(merged: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    pair = merged.loc[merged["population"].isin(["missed_oracle_positive", "non_actionable"])].copy()
    if pair.empty:
        return pd.DataFrame()
    labels = pair["population"].eq("missed_oracle_positive").astype(int)
    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        auc = _mann_whitney_auc(pair[feature], labels)
        if not np.isfinite(auc):
            continue
        rows.append(
            {
                "feature": feature,
                "auc_missed_vs_non_actionable": auc,
                "separation_auc": max(auc, 1.0 - auc),
                "direction": "higher_in_missed" if auc >= 0.5 else "lower_in_missed",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("separation_auc", ascending=False).reset_index(drop=True)


def _score_bottlenecks(arm_diag: pd.DataFrame) -> pd.DataFrame:
    missed = arm_diag.loc[arm_diag["population"] == "missed_oracle_positive"].copy()
    if missed.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for column in SCORE_COLUMNS:
        if column not in missed.columns:
            continue
        numeric = pd.to_numeric(missed[column], errors="coerce").astype(float)
        finite = numeric.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        row: dict[str, Any] = {
            "score": column,
            "available_share": float(numeric.notna().mean()),
            "count": int(finite.shape[0]),
        }
        if finite.empty:
            rows.append(row)
            continue
        row.update(
            {
                "mean": float(finite.mean()),
                "q05": float(finite.quantile(0.05)),
                "q25": float(finite.quantile(0.25)),
                "q50": float(finite.quantile(0.50)),
                "q75": float(finite.quantile(0.75)),
                "q95": float(finite.quantile(0.95)),
                "share_lt_0": float((finite < 0.0).mean()),
                "share_lt_0_2": float((finite < 0.2).mean()),
                "share_lt_0_4": float((finite < 0.4).mean()),
                "share_lt_0_5": float((finite < 0.5).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _multiplier_distribution(arm_diag: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, group in arm_diag.groupby("population", dropna=False):
        for column in ("oracle_best_multiplier", "selected_multiplier"):
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            counts = values.value_counts(dropna=False).sort_index()
            for multiplier, count in counts.items():
                rows.append(
                    {
                        "population": population,
                        "multiplier_source": column,
                        "multiplier": float(multiplier) if pd.notna(multiplier) else np.nan,
                        "groups": int(count),
                        "share": float(count / max(len(group), 1)),
                    }
                )
    return pd.DataFrame(rows)


def analyze_missed_oracle(*, run_dir: Path, arm: str, out_dir: Path) -> dict[str, Any]:
    panel = _read_csv(run_dir / "size_action_exact_panel.csv")
    diag = _read_csv(run_dir / "size_action_selector_transfer_diagnostics.csv")
    group_features, feature_cols = _baseline_group_features(panel)
    arm_diag = _arm_diagnostics(diag, arm)
    arm_diag = _augment_with_oracle_action_scores(run_dir, arm_diag)
    colliding_features = sorted(set(feature_cols) & (set(arm_diag.columns) - set(GROUP_KEYS)))
    if colliding_features:
        feature_cols = [col for col in feature_cols if col not in colliding_features]
        group_features = group_features[GROUP_KEYS + feature_cols]
    merged = arm_diag.merge(group_features, on=GROUP_KEYS, how="left", validate="one_to_one")
    matched_share = float(merged[feature_cols].notna().any(axis=1).mean()) if feature_cols else 0.0

    population_summary = _feature_population_summary(merged, feature_cols)
    feature_diffs = _feature_differences(merged, feature_cols)
    feature_auc = _feature_auc(merged, feature_cols)
    score_bottlenecks = _score_bottlenecks(arm_diag)
    multiplier_distribution = _multiplier_distribution(arm_diag)

    out_dir.mkdir(parents=True, exist_ok=True)
    population_summary.to_csv(out_dir / "missed_oracle_population_summary.csv", index=False)
    feature_diffs.to_csv(out_dir / "missed_oracle_feature_differences.csv", index=False)
    feature_auc.to_csv(out_dir / "missed_oracle_feature_auc.csv", index=False)
    score_bottlenecks.to_csv(out_dir / "missed_oracle_score_bottlenecks.csv", index=False)
    multiplier_distribution.to_csv(out_dir / "missed_oracle_multiplier_distribution.csv", index=False)

    payload = {
        "generated_by": "analyze_size_action_missed_oracle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "arm": arm,
        "out_dir": str(out_dir),
        "feature_count": int(len(feature_cols)),
        "matched_group_feature_share": matched_share,
        "population_counts": population_summary.to_dict(orient="records"),
        "top_missed_vs_non_actionable_features": feature_auc.head(20).to_dict(orient="records")
        if not feature_auc.empty
        else [],
        "top_selected_vs_missed_differences": feature_diffs.sort_values(
            "abs_selected_vs_missed_std_diff", ascending=False
        )
        .head(20)
        .to_dict(orient="records")
        if not feature_diffs.empty
        else [],
        "multiplier_distribution": multiplier_distribution.to_dict(orient="records")
        if not multiplier_distribution.empty
        else [],
    }
    (out_dir / "missed_oracle_diagnostic.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_markdown(
        out_dir / "missed_oracle_diagnostic.md",
        payload,
        feature_auc,
        feature_diffs,
        score_bottlenecks,
        multiplier_distribution,
    )
    return payload


def _write_markdown(
    path: Path,
    payload: dict[str, Any],
    feature_auc: pd.DataFrame,
    feature_diffs: pd.DataFrame,
    score_bottlenecks: pd.DataFrame,
    multiplier_distribution: pd.DataFrame,
) -> None:
    lines = [
        "# Size-Action Missed Oracle Diagnostic",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Run dir: `{payload['run_dir']}`",
        f"Arm: `{payload['arm']}`",
        "",
        f"- Live-clean feature count: `{payload['feature_count']}`",
        f"- Matched group feature share: `{payload['matched_group_feature_share']:.4f}`",
        "",
        "## Populations",
        "",
        "| population | groups | folds |",
        "|---|---:|---:|",
    ]
    for row in payload["population_counts"]:
        lines.append(f"| `{row['population']}` | {row['groups']} | {row['folds']} |")

    lines.extend(
        [
            "",
            "## Top Live Features Separating Missed Oracle From Non-Actionable Groups",
            "",
            "| feature | separation_auc | direction |",
            "|---|---:|---|",
        ]
    )
    for _, row in feature_auc.head(15).iterrows():
        lines.append(
            f"| `{row['feature']}` | {row['separation_auc']:.4f} | `{row['direction']}` |"
        )

    lines.extend(
        [
            "",
            "## Top Differences Between Selected Positives And Missed Oracle Groups",
            "",
            "| feature | selected_mean | missed_mean | std_diff_selected_minus_missed |",
            "|---|---:|---:|---:|",
        ]
    )
    diff_sorted = feature_diffs.sort_values("abs_selected_vs_missed_std_diff", ascending=False)
    for _, row in diff_sorted.head(15).iterrows():
        lines.append(
            f"| `{row['feature']}` | {row['selected_mean']:.6g} | {row['missed_mean']:.6g} | "
            f"{row['selected_vs_missed_std_diff']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Missed Oracle Score Bottlenecks",
            "",
            "| score | count | mean | q25 | q50 | q75 | share_lt_0_4 | share_lt_0_5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in score_bottlenecks.iterrows():
        lines.append(
            f"| `{row['score']}` | {int(row.get('count', 0))} | {row.get('mean', float('nan')):.6g} | "
            f"{row.get('q25', float('nan')):.6g} | {row.get('q50', float('nan')):.6g} | "
            f"{row.get('q75', float('nan')):.6g} | {row.get('share_lt_0_4', float('nan')):.4f} | "
            f"{row.get('share_lt_0_5', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Multiplier Distribution",
            "",
            "| population | source | multiplier | groups | share |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in multiplier_distribution.iterrows():
        multiplier = row.get("multiplier", float("nan"))
        multiplier_text = "nan" if pd.isna(multiplier) else f"{float(multiplier):.2f}"
        lines.append(
            f"| `{row['population']}` | `{row['multiplier_source']}` | {multiplier_text} | "
            f"{int(row['groups'])} | {float(row['share']):.4f} |"
        )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze_missed_oracle(run_dir=args.run_dir, arm=args.arm, out_dir=args.out_dir)
    print(
        {
            "out_dir": payload["out_dir"],
            "arm": payload["arm"],
            "feature_count": payload["feature_count"],
            "matched_group_feature_share": payload["matched_group_feature_share"],
            "population_counts": payload["population_counts"],
        }
    )


if __name__ == "__main__":
    main()
