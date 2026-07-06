#!/usr/bin/env python3
"""Audit which available fields separate replay-positive execution opportunities.

This report is intentionally diagnostic.  It compares actual replay top-net rows
against the rest of the selected candidate stream and classifies columns by
whether they are safe decision-time candidates, entry-delay observables, or
outcome/leakage labels.  The goal is to identify feature families worth wiring
into meta/execution models after the current oracle gap.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "meta_prefeature_regime_source_interaction_audit_v1/meta_regime_context_filter_oos_v1/"
    "meta_regime_handoff_candidates_v1/execution_replay_all_exec_keys_cost1pct_v1/"
    "regime_friction_attribution_v1/meta_handoff_replay_regime_friction_candidates.parquet"
)
DEFAULT_OUT_DIR = DEFAULT_INPUT.parent / "execution_edge_feature_gap_v1"
TOP_FRAC = 0.10

OUTCOME_TOKENS = (
    "net_return",
    "gross_return",
    "exit_",
    "simple_policy_exit_reason",
    "holding_bars",
    "mtm_path",
    "replay_",
    "archetype_label_",
)
ENTRY_DELAY_TOKENS = (
    "delay_window_",
    "delay_max_",
    "delay_close_gap",
    "delay_entry_ref_gap",
)
FRICTION_TOKENS = (
    "expected_friction_bps",
    "expected_spread_bps",
    "expected_half_spread_bps",
    "spread_cost_bps",
    "entry_reanchor_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "price_gap_bps",
    "liquidity_capacity_weight",
    "orderbook_slippage_bps",
)
DEPLOYABLE_CONTEXT_TOKENS = (
    "rank_pct",
    "calibrated_score",
    "meta_regime_score",
    "score_rank_pct_by_month",
    "side_name",
    "source_",
    "candidate_",
    "policy_overlay",
    "scenario",
    "path_len",
    "horizon_hours",
    "barrier_multiplier",
    "barrier_pct",
    "policy_sl_return",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _num(series: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce")


def _mean(series: Any) -> float:
    arr = _num(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(series: Any) -> float:
    arr = _num(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = _num(a)
    y = _num(b)
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 8:
        return float("nan")
    return float(x[mask].rank(method="average").corr(y[mask].rank(method="average")))


def _feature_role(column: str) -> str:
    lower = column.lower()
    if any(token in lower for token in OUTCOME_TOKENS):
        return "outcome_or_replay_label_exclude"
    if any(token in lower for token in ENTRY_DELAY_TOKENS):
        return "entry_delay_observable_after_signal"
    if any(token in lower for token in FRICTION_TOKENS):
        return "candidate_execution_friction_feature"
    if any(token in lower for token in DEPLOYABLE_CONTEXT_TOKENS):
        return "candidate_decision_feature"
    if lower in {"timestamp", "symbol", "month", "market_mode", "strategy_id", "side"}:
        return "identifier_or_group"
    return "unclassified_review_before_use"


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["month"] = out["timestamp"].dt.to_period("M").astype(str)
    out["net_return"] = _num(out.get("net_return"))
    if "side_name" not in out.columns:
        side = _num(out.get("side", pd.Series(1.0, index=out.index))).fillna(1.0)
        out["side_name"] = np.where(side.lt(0.0), "short", "long")
    return out.dropna(subset=["timestamp", "net_return"]).reset_index(drop=True)


def _oracle_flags(frame: pd.DataFrame, *, top_frac: float) -> pd.Series:
    flags = pd.Series(False, index=frame.index)
    group_cols = [col for col in ("scenario", "month") if col in frame.columns]
    if not group_cols:
        group_cols = ["month"]
    for _key, group in frame.groupby(group_cols, dropna=False, sort=False):
        if group.empty:
            continue
        top_n = max(1, int(math.ceil(float(top_frac) * len(group))))
        idx = group.sort_values("net_return", ascending=False).head(top_n).index
        flags.loc[idx] = True
    return flags


def _numeric_feature_rows(frame: pd.DataFrame, oracle: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = frame["net_return"]
    y_oracle = oracle.astype(float)
    for column in frame.columns:
        values = _num(frame[column]).replace([np.inf, -np.inf], np.nan)
        finite = values.notna()
        if int(finite.sum()) < 20 or int(values.loc[finite].nunique(dropna=True)) <= 2:
            continue
        good = values[oracle & finite]
        rest = values[(~oracle) & finite]
        if len(good) < 3 or len(rest) < 10:
            continue
        std = float(values.loc[finite].std(ddof=0))
        diff = float(good.mean() - rest.mean())
        rows.append(
            {
                "feature": column,
                "feature_role": _feature_role(column),
                "finite_rows": int(finite.sum()),
                "oracle_rows": int(len(good)),
                "rest_rows": int(len(rest)),
                "oracle_mean": float(good.mean()),
                "rest_mean": float(rest.mean()),
                "mean_diff": diff,
                "standardized_diff": diff / std if std > 1e-12 else float("nan"),
                "spearman_net": _spearman(values, y),
                "spearman_oracle": _spearman(values, y_oracle),
                "month_sign_stability": _month_sign_stability(frame, column, oracle),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_standardized_diff"] = _num(out["standardized_diff"]).abs()
    return out.sort_values(["feature_role", "abs_standardized_diff"], ascending=[True, False])


def _month_sign_stability(frame: pd.DataFrame, column: str, oracle: pd.Series) -> float:
    signs: list[float] = []
    values = _num(frame[column])
    for _month, idx in frame.groupby("month", dropna=False).groups.items():
        idx = pd.Index(idx)
        good = values.loc[idx[oracle.loc[idx].to_numpy()]]
        rest = values.loc[idx[(~oracle.loc[idx]).to_numpy()]]
        good = good.dropna()
        rest = rest.dropna()
        if len(good) < 2 or len(rest) < 5:
            continue
        signs.append(float(np.sign(good.mean() - rest.mean())))
    if not signs:
        return float("nan")
    majority = max(signs.count(1.0), signs.count(-1.0), signs.count(0.0))
    return float(majority / len(signs))


def _categorical_feature_rows(frame: pd.DataFrame, oracle: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    oracle_rate = float(oracle.mean()) if len(oracle) else float("nan")
    for column in frame.columns:
        role = _feature_role(column)
        if role.startswith("outcome"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].nunique(dropna=True) > 20:
            continue
        values = frame[column].fillna("missing").astype(str)
        if int(values.nunique(dropna=True)) < 2 or int(values.nunique(dropna=True)) > 200:
            continue
        for value, idx in values.groupby(values, dropna=False).groups.items():
            idx = pd.Index(idx)
            if len(idx) < 5:
                continue
            local_oracle = oracle.loc[idx]
            local_net = frame.loc[idx, "net_return"]
            rate = float(local_oracle.mean()) if len(local_oracle) else float("nan")
            rows.append(
                {
                    "feature": column,
                    "feature_role": role,
                    "value": str(value),
                    "rows": int(len(idx)),
                    "oracle_rows": int(local_oracle.sum()),
                    "oracle_rate": rate,
                    "oracle_lift": rate / oracle_rate if oracle_rate and math.isfinite(oracle_rate) else float("nan"),
                    "mean_net_return": _mean(local_net),
                    "hit_net_rate": _rate(local_net.gt(0.0)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_lift_from_1"] = (_num(out["oracle_lift"]) - 1.0).abs()
    return out.sort_values(["feature_role", "abs_lift_from_1", "rows"], ascending=[True, False, False])


def _summary(frame: pd.DataFrame, oracle: pd.Series) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "months": sorted(frame["month"].astype(str).unique()),
        "scenarios": sorted(frame["scenario"].astype(str).unique()) if "scenario" in frame.columns else [],
        "oracle_top_frac": float(TOP_FRAC),
        "oracle_rows": int(oracle.sum()),
        "oracle_rate": float(oracle.mean()) if len(oracle) else float("nan"),
        "mean_net_return": _mean(frame["net_return"]),
        "oracle_mean_net_return": _mean(frame.loc[oracle, "net_return"]),
        "rest_mean_net_return": _mean(frame.loc[~oracle, "net_return"]),
        "positive_net_rate": _rate(frame["net_return"].gt(0.0)),
        "oracle_positive_net_rate": _rate(frame.loc[oracle, "net_return"].gt(0.0)),
    }


def run_report(*, input_path: Path, out_dir: Path, top_frac: float) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _prepare(pd.read_parquet(input_path))
    oracle = _oracle_flags(frame, top_frac=top_frac)
    numeric = _numeric_feature_rows(frame, oracle)
    categorical = _categorical_feature_rows(frame, oracle)
    paths = {
        "numeric_feature_gap": out_dir / "execution_edge_numeric_feature_gap.csv",
        "categorical_feature_gap": out_dir / "execution_edge_categorical_feature_gap.csv",
        "manifest": out_dir / "manifest.json",
        "report": out_dir / "execution_edge_feature_gap_report.md",
    }
    numeric.to_csv(paths["numeric_feature_gap"], index=False)
    categorical.to_csv(paths["categorical_feature_gap"], index=False)
    summary = _summary(frame, oracle)
    manifest = {
        "generated_by": "report_execution_edge_feature_gap",
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "summary": summary,
        "feature_role_counts_numeric": numeric["feature_role"].value_counts(dropna=False).to_dict()
        if "feature_role" in numeric.columns
        else {},
        "feature_role_counts_categorical": categorical["feature_role"].value_counts(dropna=False).to_dict()
        if "feature_role" in categorical.columns
        else {},
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    lines = [
        "# Execution Edge Feature Gap",
        "",
        "Oracle rows are the top replay-net candidates within each scenario × month.",
        "Outcome/replay label columns are reported separately or excluded from candidate feature conclusions.",
        "",
        "## Summary",
        "",
        json.dumps(_json_safe(summary), indent=2),
        "",
        "## Candidate Decision/Friction Numeric Features",
        "",
    ]
    if numeric.empty:
        lines.append("No numeric feature rows.")
    else:
        safe_numeric = numeric[numeric["feature_role"].isin(["candidate_decision_feature", "candidate_execution_friction_feature"])]
        display = [
            "feature",
            "feature_role",
            "finite_rows",
            "oracle_mean",
            "rest_mean",
            "standardized_diff",
            "spearman_net",
            "month_sign_stability",
        ]
        lines.append(safe_numeric[display].head(30).to_markdown(index=False))
    lines.extend(["", "## Candidate Categorical Features", ""])
    if categorical.empty:
        lines.append("No categorical feature rows.")
    else:
        safe_cat = categorical[categorical["feature_role"].isin(["candidate_decision_feature", "candidate_execution_friction_feature"])]
        display = [
            "feature",
            "value",
            "rows",
            "oracle_rate",
            "oracle_lift",
            "mean_net_return",
            "hit_net_rate",
        ]
        lines.append(safe_cat[display].head(30).to_markdown(index=False))
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-frac", type=float, default=TOP_FRAC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(input_path=args.input, out_dir=args.out_dir, top_frac=float(args.top_frac))
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
