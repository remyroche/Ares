#!/usr/bin/env python3
"""Complete the paired base-IC versus execution-EV diagnostic on exact IDs.

The runner is diagnostic-only.  It freezes the monthly pooled-global base
top-10 book, emits a unified target bridge and economic decomposition, tests
same-ID fixed-time/oracle exits from native 1-minute paths, quantifies joint
book-composition effects, and reconciles month changes with an exact Shapley
decomposition.  It never changes the deployed exit label or admission policy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PANEL = ROOT / (
    "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/"
    "panel.parquet"
)
DEFAULT_PATHS = ROOT / (
    "data_perp/artifacts/febapr2025_native_first_touch_full_12h_paths_"
    "20260729_v2"
)
DEFAULT_LEARNABILITY = ROOT / (
    "data_perp/artifacts/canonical_base_conversion_prediction_attribution_"
    "20260729_v1"
)

BRIDGE_TARGETS = {
    "native_24h_alpha": "__first_touch_target_soft__",
    "exact_12h_mfe": "execution_mfe_return_12h",
    "exact_12h_gross": "execution_gross_ev_12h",
    "exact_cost": "execution_cost_return",
    "exact_12h_net": "execution_net_ev_12h",
}
EXIT_CLASSES = ("trailing", "timeout", "full_stop", "adverse_exit")
SHAPLEY_FACTORS = (
    "composition",
    "opportunity_prevalence",
    "exit_mix",
    "favorable_exit_payoff",
    "adverse_exit_payoff",
    "cost",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def corr(left: pd.Series, right: pd.Series) -> float:
    local = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(local) < 3 or local.left.nunique() < 2 or local.right.nunique() < 2:
        return np.nan
    result = spearmanr(local.left, local.right).statistic
    return float(result) if np.isfinite(result) else np.nan


def stable_top(frame: pd.DataFrame, fraction: float = 0.10) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * fraction)))
    order = np.lexsort(
        (
            frame["candidate_id"].astype(str).to_numpy(),
            -pd.to_numeric(frame["base_oof_score"], errors="raise").to_numpy(),
        )
    )
    return frame.iloc[order[:count]].copy()


def add_score_decile(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["score_decile"] = (
        result.groupby("candidate_month", observed=True)["base_oof_score"]
        .rank(method="first", pct=True)
        .mul(10.0)
        .sub(np.finfo(float).eps)
        .astype(int)
        .clip(0, 9)
    )
    return result


def bridge_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    deciles: list[dict[str, Any]] = []
    work = add_score_decile(panel)
    for month, month_rows in work.groupby("candidate_month", sort=True):
        for scope, local in [
            ("pooled_global", month_rows),
            *[
                (f"side_{side}", rows)
                for side, rows in month_rows.groupby("side_name", sort=True)
            ],
        ]:
            for target, column in BRIDGE_TARGETS.items():
                metrics.append(
                    {
                        "candidate_month": str(month),
                        "scope": scope,
                        "target": target,
                        "rows": int(len(local)),
                        "rank_ic": corr(local["base_oof_score"], local[column]),
                    }
                )
        for decile, local in month_rows.groupby("score_decile", sort=True):
            row: dict[str, Any] = {
                "candidate_month": str(month),
                "score_decile": int(decile),
                "rows": int(len(local)),
                "score_mean": float(local.base_oof_score.mean()),
            }
            for target, column in BRIDGE_TARGETS.items():
                row[f"{target}__mean"] = float(local[column].mean())
                row[f"{target}__rank_ic"] = corr(
                    local["base_oof_score"], local[column]
                )
            deciles.append(row)
    return pd.DataFrame(metrics), pd.DataFrame(deciles)


def _conditional_mean(
    frame: pd.DataFrame, value: str, condition: pd.Series
) -> float:
    values = pd.to_numeric(frame.loc[condition, value], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def economic_decile_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    work = add_score_decile(panel)
    rows: list[dict[str, Any]] = []
    for (month, decile), local in work.groupby(
        ["candidate_month", "score_decile"], sort=True, observed=True
    ):
        opportunity = local["opportunity_gross_above_cost_0bps"].astype(bool)
        net = pd.to_numeric(local.execution_net_ev_12h, errors="coerce")
        row: dict[str, Any] = {
            "candidate_month": str(month),
            "score_decile": int(decile),
            "rows": int(len(local)),
            "opportunity_rate": float(opportunity.mean()),
            "positive_net_rate": float(net.gt(0).mean()),
            "conditional_favorable_gross": _conditional_mean(
                local, "execution_gross_ev_12h", opportunity
            ),
            "conditional_nonopportunity_gross": _conditional_mean(
                local, "execution_gross_ev_12h", ~opportunity
            ),
            "mfe_mean": float(local.execution_mfe_return_12h.mean()),
            "mae_mean": float(local.execution_mae_return_12h.mean()),
            "exit_minute_mean": float(local.execution_exit_minute.mean()),
            "gross_mean": float(local.execution_gross_ev_12h.mean()),
            "cost_mean": float(local.execution_cost_return.mean()),
            "net_mean": float(local.execution_net_ev_12h.mean()),
        }
        for exit_class in EXIT_CLASSES:
            mask = local["execution_exit_class"].astype(str).eq(exit_class)
            row[f"exit_{exit_class}__rate"] = float(mask.mean())
            row[f"exit_{exit_class}__conditional_net"] = _conditional_mean(
                local, "execution_net_ev_12h", mask
            )
            row[f"exit_{exit_class}__conditional_gross"] = _conditional_mean(
                local, "execution_gross_ev_12h", mask
            )
            row[f"exit_{exit_class}__conditional_mfe"] = _conditional_mean(
                local, "execution_mfe_return_12h", mask
            )
            row[f"exit_{exit_class}__conditional_mae"] = _conditional_mean(
                local, "execution_mae_return_12h", mask
            )
            row[f"exit_{exit_class}__conditional_minute"] = _conditional_mean(
                local, "execution_exit_minute", mask
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _decode_fixed_returns(
    payload: str,
    decision_price: float,
    side: str,
) -> dict[str, float]:
    parsed = json.loads(payload)
    close = np.asarray(parsed["close"], dtype=float)
    if close.shape != (720,) or not np.isfinite(decision_price) or decision_price <= 0:
        raise ValueError("native path must be a finite 720x1m close path")
    sign = 1.0 if str(side).lower() == "long" else -1.0
    result = {}
    for hours, index in ((1, 59), (2, 119), (4, 239), (8, 479), (12, 719)):
        result[f"fixed_{hours}h_gross"] = float(
            sign * (close[index] / decision_price - 1.0)
        )
    return result


def load_selected_path_counterfactuals(
    paths_root: Path,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    selected_ids = set(selected["candidate_id"].astype(str))
    pieces: list[pd.DataFrame] = []
    for path in sorted((paths_root / "shards").glob("*_paths.parquet")):
        identity = pd.read_parquet(
            path, columns=["candidate_id", "side_name", "native_future_ohlc_path", "decision_price"]
        )
        local = identity.loc[identity.candidate_id.astype(str).isin(selected_ids)]
        if local.empty:
            continue
        records = []
        for row in local.itertuples(index=False):
            records.append(
                {
                    "candidate_id": row.candidate_id,
                    **_decode_fixed_returns(
                        row.native_future_ohlc_path,
                        float(row.decision_price),
                        str(row.side_name),
                    ),
                }
            )
        pieces.append(pd.DataFrame(records))
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if (
        len(result) != len(selected)
        or result["candidate_id"].astype(str).duplicated().any()
        or set(result["candidate_id"].astype(str)) != selected_ids
    ):
        raise ValueError("native paths do not exactly cover frozen selected IDs")
    return result


def exit_counterfactuals(
    selected: pd.DataFrame,
    paths: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = selected.merge(paths, on="candidate_id", how="inner", validate="one_to_one")
    cost = pd.to_numeric(joined.execution_cost_return, errors="raise")
    joined["deployed_net"] = joined.execution_net_ev_12h
    joined["oracle_mfe_net"] = joined.execution_mfe_return_12h - cost
    for hours in (1, 2, 4, 8, 12):
        joined[f"fixed_{hours}h_net"] = joined[f"fixed_{hours}h_gross"] - cost
    rows: list[dict[str, Any]] = []
    score_columns = [
        "deployed_net",
        "fixed_1h_net",
        "fixed_2h_net",
        "fixed_4h_net",
        "fixed_8h_net",
        "fixed_12h_net",
        "oracle_mfe_net",
    ]
    for month, local in joined.groupby("candidate_month", sort=True):
        deployed = float(local.deployed_net.mean())
        for column in score_columns:
            rows.append(
                {
                    "candidate_month": str(month),
                    "exit_counterfactual": column,
                    "rows": int(len(local)),
                    "mean_net_bps": float(local[column].mean() * 1e4),
                    "positive_rate": float(local[column].gt(0).mean()),
                    "delta_vs_deployed_bps": float(
                        (local[column].mean() - deployed) * 1e4
                    ),
                    "diagnostic_only": column != "deployed_net",
                }
            )
    audit_columns = [
        "candidate_id",
        "candidate_month",
        "side_name",
        "deployed_net",
        *[f"fixed_{hours}h_net" for hours in (1, 2, 4, 8, 12)],
        "oracle_mfe_net",
    ]
    return pd.DataFrame(rows), joined.loc[:, audit_columns]


def add_composition_strata(selected: pd.DataFrame) -> pd.DataFrame:
    result = add_score_decile(selected)
    top_assets = set(result["__symbol__"].value_counts().head(20).index.astype(str))
    result["asset_bucket"] = np.where(
        result["__symbol__"].astype(str).isin(top_assets),
        result["__symbol__"].astype(str),
        "__other__",
    )
    result["candidate_group_bin"] = pd.cut(
        result["base_group_rows_timestamp_global"],
        bins=[-np.inf, 25, 50, 100, np.inf],
        labels=["le25", "26_50", "51_100", "gt100"],
    ).astype(str)
    result["opportunity_binary"] = result[
        "opportunity_gross_above_cost_0bps"
    ].astype(int)
    exit_class = result["execution_exit_class"].astype(str)
    result["exit_bucket"] = np.where(
        exit_class.isin(EXIT_CLASSES), exit_class, "adverse_exit"
    )
    result["stratum"] = (
        result["side_name"].astype(str)
        + "|"
        + result["asset_bucket"].astype(str)
        + "|d"
        + result["score_decile"].astype(str)
        + "|"
        + result["candidate_group_bin"].astype(str)
    )
    return result


def joint_composition_reweight(selected: pd.DataFrame) -> pd.DataFrame:
    work = add_composition_strata(selected)
    rows = []
    months = sorted(work.candidate_month.astype(str).unique())
    for source_month, target_month in zip(months[:-1], months[1:]):
        source = work.loc[work.candidate_month.astype(str).eq(source_month)]
        target = work.loc[work.candidate_month.astype(str).eq(target_month)]
        source_cell = source.groupby("stratum", observed=True).agg(
            source_rows=("candidate_id", "size"),
            source_net=("execution_net_ev_12h", "mean"),
        )
        target_cell = target.groupby("stratum", observed=True).agg(
            target_rows=("candidate_id", "size"),
            target_net=("execution_net_ev_12h", "mean"),
        )
        common = source_cell.join(target_cell, how="inner")
        source_common_mass = common.source_rows.sum() / len(source)
        target_common_mass = common.target_rows.sum() / len(target)
        source_weight = common.source_rows / common.source_rows.sum()
        target_weight = common.target_rows / common.target_rows.sum()
        source_common = float(np.sum(source_weight * common.source_net))
        target_common = float(np.sum(target_weight * common.target_net))
        source_under_target_mix = float(np.sum(target_weight * common.source_net))
        rows.append(
            {
                "from_month": source_month,
                "to_month": target_month,
                "strata": int(len(common)),
                "from_common_mass": float(source_common_mass),
                "to_common_mass": float(target_common_mass),
                "from_common_net_bps": source_common * 1e4,
                "to_common_net_bps": target_common * 1e4,
                "composition_effect_bps": (
                    source_under_target_mix - source_common
                )
                * 1e4,
                "within_cell_payoff_effect_bps": (
                    target_common - source_under_target_mix
                )
                * 1e4,
                "common_delta_bps": (target_common - source_common) * 1e4,
            }
        )
    return pd.DataFrame(rows)


def _build_month_parameters(frame: pd.DataFrame) -> dict[str, Any]:
    total = float(len(frame))
    strata = sorted(frame["stratum"].unique())
    composition = frame["stratum"].value_counts().div(total).to_dict()
    opportunity = (
        frame.groupby("stratum", observed=True)["opportunity_binary"].mean().to_dict()
    )
    global_opportunity = float(frame.opportunity_binary.mean())
    exit_mix = (
        frame.groupby(["stratum", "opportunity_binary", "exit_bucket"], observed=True)
        .size()
        .groupby(level=[0, 1])
        .transform(lambda values: values / values.sum())
        .to_dict()
    )
    global_exit = (
        frame.groupby(["opportunity_binary", "exit_bucket"], observed=True)
        .size()
        .groupby(level=0)
        .transform(lambda values: values / values.sum())
        .to_dict()
    )
    payoff = (
        frame.groupby(
            ["stratum", "opportunity_binary", "exit_bucket"], observed=True
        )["execution_gross_ev_12h"]
        .mean()
        .to_dict()
    )
    global_payoff = (
        frame.groupby(["opportunity_binary", "exit_bucket"], observed=True)[
            "execution_gross_ev_12h"
        ]
        .mean()
        .to_dict()
    )
    cost = (
        frame.groupby(
            ["stratum", "opportunity_binary", "exit_bucket"], observed=True
        )["execution_cost_return"]
        .mean()
        .to_dict()
    )
    global_cost = (
        frame.groupby(["opportunity_binary", "exit_bucket"], observed=True)[
            "execution_cost_return"
        ]
        .mean()
        .to_dict()
    )
    return {
        "strata": strata,
        "composition": composition,
        "opportunity": opportunity,
        "global_opportunity": global_opportunity,
        "exit_mix": exit_mix,
        "global_exit": global_exit,
        "payoff": payoff,
        "global_payoff": global_payoff,
        "cost": cost,
        "global_cost": global_cost,
        "actual": float(frame.execution_net_ev_12h.mean()),
    }


def _evaluate_mix(
    source: dict[str, Any],
    target: dict[str, Any],
    target_factors: frozenset[str],
) -> float:
    choose = lambda factor: target if factor in target_factors else source
    comp = choose("composition")
    opp = choose("opportunity_prevalence")
    exits = choose("exit_mix")
    cost_source = choose("cost")
    value = 0.0
    for stratum, weight in comp["composition"].items():
        p_opp = opp["opportunity"].get(stratum, opp["global_opportunity"])
        for opportunity, p_state in ((1, p_opp), (0, 1.0 - p_opp)):
            payoff_source = choose(
                "favorable_exit_payoff"
                if opportunity == 1
                else "adverse_exit_payoff"
            )
            exit_probabilities = {
                exit_class: exits["exit_mix"].get(
                    (stratum, opportunity, exit_class),
                    exits["global_exit"].get((opportunity, exit_class), 0.0),
                )
                for exit_class in EXIT_CLASSES
            }
            total_exit = sum(exit_probabilities.values())
            if total_exit <= 0:
                continue
            for exit_class, probability in exit_probabilities.items():
                probability /= total_exit
                key = (stratum, opportunity, exit_class)
                fallback = (opportunity, exit_class)
                gross = payoff_source["payoff"].get(
                    key, payoff_source["global_payoff"].get(fallback, 0.0)
                )
                cost = cost_source["cost"].get(
                    key, cost_source["global_cost"].get(fallback, 0.0)
                )
                value += weight * p_state * probability * (gross - cost)
    return float(value)


def unified_shapley_attribution(selected: pd.DataFrame) -> pd.DataFrame:
    work = add_composition_strata(selected)
    parameters = {
        str(month): _build_month_parameters(local)
        for month, local in work.groupby("candidate_month", sort=True)
    }
    months = sorted(parameters)
    rows: list[dict[str, Any]] = []
    factorial = math.factorial
    factor_count = len(SHAPLEY_FACTORS)
    for from_month, to_month in zip(months[:-1], months[1:]):
        source = parameters[from_month]
        target = parameters[to_month]
        cache: dict[frozenset[str], float] = {}
        for size in range(factor_count + 1):
            for subset in itertools.combinations(SHAPLEY_FACTORS, size):
                key = frozenset(subset)
                cache[key] = _evaluate_mix(source, target, key)
        contributions: dict[str, float] = {}
        for factor in SHAPLEY_FACTORS:
            contribution = 0.0
            others = [name for name in SHAPLEY_FACTORS if name != factor]
            for size in range(factor_count):
                coefficient = (
                    factorial(size)
                    * factorial(factor_count - size - 1)
                    / factorial(factor_count)
                )
                for subset in itertools.combinations(others, size):
                    key = frozenset(subset)
                    contribution += coefficient * (
                        cache[key | {factor}] - cache[key]
                    )
            contributions[factor] = contribution
        actual_delta = target["actual"] - source["actual"]
        modeled_delta = sum(contributions.values())
        for factor, contribution in contributions.items():
            rows.append(
                {
                    "from_month": from_month,
                    "to_month": to_month,
                    "component": factor,
                    "contribution_bps": contribution * 1e4,
                    "actual_delta_bps": actual_delta * 1e4,
                }
            )
        rows.append(
            {
                "from_month": from_month,
                "to_month": to_month,
                "component": "interaction_or_fallback_remainder",
                "contribution_bps": (actual_delta - modeled_delta) * 1e4,
                "actual_delta_bps": actual_delta * 1e4,
            }
        )
    return pd.DataFrame(rows)


def learnability_linkage(root: Path) -> pd.DataFrame:
    attribution = pd.read_parquet(root / "monthly_base_tail_attribution.parquet")
    bootstrap = pd.read_parquet(root / "high_low_daily_block_bootstrap.parquet")
    rows = []
    for head, local in attribution.groupby("head", sort=True):
        top10 = local.loc[np.isclose(local.fraction, 0.10)]
        uncertainty = bootstrap.loc[bootstrap["head"].eq(head)]
        rows.append(
            {
                "head": head,
                "months": "|".join(sorted(top10.candidate_month.astype(str).unique())),
                "minimum_prediction_coverage": float(top10.prediction_coverage.min()),
                "mean_net_rank_ic": float(
                    top10.conversion_prediction_net_rank_ic.mean()
                ),
                "mean_opportunity_rank_ic": float(
                    top10.conversion_prediction_opportunity_rank_ic.mean()
                ),
                "mean_upside_rank_ic": float(
                    top10.conversion_prediction_upside_rank_ic.mean()
                ),
                "mean_loss_rank_ic": float(
                    top10.conversion_prediction_loss_rank_ic.mean()
                ),
                "tail_ci_excludes_zero_all_months": bool(
                    len(uncertainty)
                    and (
                        uncertainty.ci95_low_bps.gt(0)
                        | uncertainty.ci95_high_bps.lt(0)
                    ).all()
                ),
                "promotion_eligible": False,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_path = Path(args.panel)
    paths_root = Path(args.paths_root)
    learnability_root = Path(args.learnability)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    panel = pd.read_parquet(panel_path)
    required = {
        "candidate_id",
        "candidate_month",
        "side_name",
        "__symbol__",
        "base_oof_score",
        "base_group_rows_timestamp_global",
        *BRIDGE_TARGETS.values(),
        "execution_mae_return_12h",
        "execution_exit_minute",
        "execution_exit_class",
        "opportunity_gross_above_cost_0bps",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"canonical panel lacks {missing}")
    if panel.candidate_id.duplicated().any():
        raise ValueError("canonical candidate IDs must be unique")
    selected = pd.concat(
        [stable_top(local, 0.10) for _, local in panel.groupby("candidate_month", sort=True)],
        ignore_index=True,
    )
    bridge_ic, bridge_deciles = bridge_tables(panel)
    decomposition = economic_decile_decomposition(panel)
    native_paths = load_selected_path_counterfactuals(paths_root, selected)
    exit_summary, exit_candidates = exit_counterfactuals(selected, native_paths)
    composition = joint_composition_reweight(selected)
    attribution = unified_shapley_attribution(selected)
    learnability = learnability_linkage(learnability_root)
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "bridge_ic": output / "unified_target_bridge_ic.parquet",
        "bridge_deciles": output / "unified_target_bridge_deciles.parquet",
        "economic_deciles": output / "per_decile_economic_decomposition.parquet",
        "exit_summary": output / "same_id_exit_counterfactual_summary.parquet",
        "exit_candidates": output / "same_id_exit_counterfactual_candidates.parquet",
        "composition": output / "joint_composition_reweighting.parquet",
        "attribution": output / "unified_change_attribution.parquet",
        "learnability": output / "decision_time_learnability_linkage.parquet",
        "selected": output / "frozen_monthly_global_top10_candidates.parquet",
    }
    frames = {
        "bridge_ic": bridge_ic,
        "bridge_deciles": bridge_deciles,
        "economic_deciles": decomposition,
        "exit_summary": exit_summary,
        "exit_candidates": exit_candidates,
        "composition": composition,
        "attribution": attribution,
        "learnability": learnability,
        "selected": selected,
    }
    for key, frame in frames.items():
        frame.to_parquet(outputs[key], index=False, compression="zstd")
    report = {
        "schema": "base_ic_execution_ev_paired_completion_v1",
        "status": "DIAGNOSTIC_COMPLETE_NO_POLICY_CHANGE",
        "population": {
            "canonical_rows": int(len(panel)),
            "selected_rows": int(len(selected)),
            "months": {
                str(month): int(len(local))
                for month, local in selected.groupby("candidate_month", sort=True)
            },
            "selection": (
                "one pooled-global monthly base-score top10 with candidate-ID "
                "tie break; never per timestamp or side"
            ),
        },
        "contracts": {
            "deployed_exit_is_canonical_label": True,
            "fixed_time_and_oracle_are_diagnostic_only": True,
            "same_selected_candidate_ids": True,
            "native_paths": "exact decision+12h 720x1m OHLC",
            "composition_strata": (
                "side x top20/other asset x score decile x fixed candidate-group bin"
            ),
            "attribution_identity": (
                "composition x opportunity prevalence x exit mix x "
                "favorable/adverse exit-conditional gross payoff x cost"
            ),
        },
        "sources": {
            "panel": {"path": str(panel_path.resolve()), "sha256": sha256(panel_path)},
            "path_manifest": {
                "path": str((paths_root / "manifest.json").resolve()),
                "sha256": sha256(paths_root / "manifest.json"),
            },
            "learnability_manifest": {
                "path": str((learnability_root / "manifest.json").resolve()),
                "sha256": sha256(learnability_root / "manifest.json"),
            },
        },
        "outputs": {
            key: {
                "path": str(path.resolve()),
                "rows": int(len(frames[key])),
                "sha256": sha256(path),
            }
            for key, path in outputs.items()
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "promotion_eligible": False,
    }
    write_json(output / "manifest.json", report)
    (output / "manifest.sha256").write_text(
        sha256(output / "manifest.json") + "\n", encoding="utf-8"
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    result.add_argument("--paths-root", type=Path, default=DEFAULT_PATHS)
    result.add_argument("--learnability", type=Path, default=DEFAULT_LEARNABILITY)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    print(json.dumps(safe(run(parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
