"""Immutable Round-3 joint-stack shortlist for the Stage-I target funnel.

Base-layer economics are diagnostics only. They may choose one deterministic
configuration inside each S/O target family, but may neither promote nor reject
R3, S, or O. All three representatives must receive their matching same-side
direct-FQ3 meta layer before any terminal comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA = "stage_i_base_target_round3_joint_shortlist_v2"
R3_ARM = "R3_frozen_control"


class StageITargetPromotionError(ValueError):
    """Raised when a Round-3 scorecard cannot support a deterministic gate."""


def canonical_sha(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PromotionGateContract:
    """Predeclared, intentionally conservative diagnostic comparisons.

    All comparisons are values in common expected-net bps after the side-local
    causal map, followed by a *single pooled-global* rank.  A challenger must
    provide a strictly better primary economic result and must not trade away
    a tail or robustness diagnostic to obtain it. These checks are descriptive
    only and cannot remove a family from joint meta evaluation.
    """

    primary_metric: str = "pooled_top10_net_bps"
    required_strict_improvement_bps: float = 0.0
    non_regression_metrics: tuple[str, ...] = (
        "pooled_top1_net_bps",
        "pooled_top5_net_bps",
        "robust_top10_lift_score",
        "worst_era_top10_net_bps",
        "worst_side_top10_net_bps",
        "worst_regime_top10_net_bps",
        "latest_era_top10_net_bps",
    )
    maximum_metrics: tuple[str, ...] = ("mapped_ev_monotonicity_violations",)
    scorecard_scope: str = "round3_identical_single_holdout_common_bps_pooled_global"
    evidence_status: str = "development_target_selection_not_final_oos_or_production_promotion"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contract_sha256"] = canonical_sha(value)
        return value


DEFAULT_GATE = PromotionGateContract()

_REQUIRED = {
    "arm", "weight_mode", "pooled_top10_net_bps", "pooled_top1_net_bps",
    "pooled_top5_net_bps", "robust_top10_lift_score", "worst_era_top10_net_bps",
    "worst_side_top10_net_bps", "worst_regime_top10_net_bps",
    "latest_era_top10_net_bps", "mapped_ev_monotonicity_violations",
}


def _require_one(frame: pd.DataFrame, *, arm: str) -> pd.Series:
    rows = frame.loc[frame.arm.astype(str).eq(arm)].copy()
    if rows.empty:
        raise StageITargetPromotionError(f"Round-3 scorecard lacks {arm}")
    # Same deterministic ordering used by the funnel family scorecard.  This
    # breaks ties before the gate rather than depending on source row order.
    ordered = rows.sort_values(
        [
            "pooled_top10_net_bps", "pooled_top1_net_bps", "pooled_top5_net_bps",
            "robust_top10_lift_score", "worst_era_top10_net_bps",
            "worst_side_top10_net_bps", "worst_regime_top10_net_bps",
            "latest_era_top10_net_bps", "mapped_ev_monotonicity_violations", "weight_mode",
        ],
        ascending=[False, False, False, False, False, False, False, False, True, True],
        kind="mergesort",
    )
    return ordered.iloc[0]


def _family_winner(frame: pd.DataFrame, *, prefix: str) -> pd.Series:
    choices = frame.loc[frame.arm.astype(str).str.startswith(prefix)].copy()
    if choices.empty:
        raise StageITargetPromotionError(f"Round-3 scorecard lacks a {prefix} challenger")
    choices = choices.sort_values(
        [
            "pooled_top10_net_bps", "pooled_top1_net_bps", "pooled_top5_net_bps",
            "robust_top10_lift_score", "worst_era_top10_net_bps",
            "worst_side_top10_net_bps", "worst_regime_top10_net_bps",
            "latest_era_top10_net_bps", "mapped_ev_monotonicity_violations", "arm", "weight_mode",
        ],
        ascending=[False, False, False, False, False, False, False, False, True, True, True],
        kind="mergesort",
    )
    return choices.iloc[0]


def _finite_row(row: pd.Series, fields: set[str]) -> bool:
    return all(np.isfinite(pd.to_numeric(pd.Series([row[field]]), errors="coerce").iloc[0]) for field in fields)


def _scorecard_row(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if key in {"arm", "weight_mode"}:
            result[str(key)] = str(value)
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        result[str(key)] = float(numeric) if np.isfinite(numeric) else None
    return result


def _gate_challenger(
    *, baseline: pd.Series, challenger: pd.Series, contract: PromotionGateContract,
) -> dict[str, Any]:
    fields = {contract.primary_metric, *contract.non_regression_metrics, *contract.maximum_metrics}
    finite = _finite_row(baseline, fields) and _finite_row(challenger, fields)
    checks: list[dict[str, Any]] = [{
        "gate": "finite_and_comparable_scorecard_metrics", "passed": bool(finite),
        "baseline": None, "challenger": None,
        "reason": "both rows must contain finite values for every predeclared gate",
    }]
    primary_delta = float(challenger[contract.primary_metric]) - float(baseline[contract.primary_metric]) if finite else np.nan
    checks.append({
        "gate": "pooled_global_common_bps_top10_strict_improvement",
        "passed": bool(finite and primary_delta > contract.required_strict_improvement_bps),
        "baseline": float(baseline[contract.primary_metric]) if finite else None,
        "challenger": float(challenger[contract.primary_metric]) if finite else None,
        "delta_bps": float(primary_delta) if finite else None,
        "reason": "challenger must strictly beat frozen R3 on pooled global top-10% net bps",
    })
    for metric in contract.non_regression_metrics:
        baseline_value = float(baseline[metric]) if finite else None
        challenger_value = float(challenger[metric]) if finite else None
        checks.append({
            "gate": f"non_regression__{metric}",
            "passed": bool(finite and challenger_value >= baseline_value),
            "baseline": baseline_value, "challenger": challenger_value,
            "delta_bps": (challenger_value - baseline_value) if finite else None,
            "reason": "challenger may not obtain its primary gain by degrading a declared tail/robustness metric",
        })
    for metric in contract.maximum_metrics:
        baseline_value = float(baseline[metric]) if finite else None
        challenger_value = float(challenger[metric]) if finite else None
        checks.append({
            "gate": f"non_regression_maximum__{metric}",
            "passed": bool(finite and challenger_value <= baseline_value),
            "baseline": baseline_value, "challenger": challenger_value,
            "delta": (challenger_value - baseline_value) if finite else None,
            "reason": "challenger may not increase common-bps mapped-EV monotonicity violations",
        })
    passed = all(bool(item["passed"]) for item in checks)
    return {
        "arm": str(challenger.arm), "weight_mode": str(challenger.weight_mode),
        "passed": passed, "checks": checks,
        "primary_delta_bps": float(primary_delta) if finite else None,
    }


def decide_round3_promotion(
    scorecard: pd.DataFrame, *, source_contract: Mapping[str, Any],
    gate: PromotionGateContract = DEFAULT_GATE,
) -> dict[str, Any]:
    """Return immutable R3 + best-S + best-O joint-meta shortlist evidence.

    ``source_contract`` is intentionally carried verbatim into the signed
    decision.  The runner supplies selector, label-grid, source-code and
    scorecard hashes so downstream MDA cannot detach a selected target from
    the materialisation that justified it.
    """

    missing = sorted(_REQUIRED.difference(scorecard.columns))
    if missing:
        raise StageITargetPromotionError(f"Round-3 scorecard lacks required gate columns: {missing}")
    if scorecard.empty:
        raise StageITargetPromotionError("Round-3 scorecard is empty")
    baseline = _require_one(scorecard, arm=R3_ARM)
    s_winner = _family_winner(scorecard, prefix="S__")
    o_winner = _family_winner(scorecard, prefix="O_")
    decisions = [_gate_challenger(baseline=baseline, challenger=row, contract=gate) for row in (s_winner, o_winner)]
    finalist_rows = (baseline, s_winner, o_winner)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "promotion_scope": "base_layer_diagnostic_and_joint_stack_shortlist_only",
        "gate_contract": gate.to_dict(),
        "source_contract": dict(source_contract),
        "baseline_r3": _scorecard_row(baseline),
        "shortlist_disposition": "ALL_THREE_FAMILIES_REQUIRE_MATCHING_DIRECT_FQ3_META",
        "base_diagnostic_comparisons": decisions,
        "finalists": [
            {
                "arm": str(row.arm),
                "family": (
                    "R3_control" if str(row.arm) == R3_ARM
                    else ("scalar_S" if str(row.arm).startswith("S__") else "ordinal_O")
                ),
                "weight_mode": str(row.weight_mode),
                "base_diagnostic_scorecard_row": _scorecard_row(row),
                "must_advance_to_joint_base_meta_evaluation": True,
            }
            for row in finalist_rows
        ],
        "joint_stack_requirement": {
            "terminal_selection_layer": "reconstructed_base_plus_direct_three_class_meta",
            "base_only_economics_are_diagnostic": True,
            "base_only_gate_may_not_promote_a_finalist": True,
            "base_only_gate_may_not_eliminate_a_finalist": True,
            "comparison_scope": "identical_rows_after_causal_common_bps_mapping",
        },
        "rationale": (
            "R3 and the deterministic best S and O configurations are a three-stack shortlist. "
            "The base-only comparison is diagnostic; no target is promoted or rejected until its matching direct three-class meta layer is evaluated on identical rows."
        ),
    }
    payload["decision_sha256"] = canonical_sha(payload)
    return payload
