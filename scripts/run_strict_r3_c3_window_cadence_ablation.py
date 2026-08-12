#!/usr/bin/env python3
"""Long-only C3 downstream window/cadence ablation with executable replay.

The upstream strict-prequential R3 base and ten-head consensus ledger is held
fixed.  Every downstream fit receives exactly one newly fitted, base-independent
raw K9 bundle.  Its supervised rows begin after the bundle burn-in, so one model
never consumes geometry fields from two bundle definitions.

Arms are evaluated twice:

* one pooled-global score ranking, including Top-2% economics and monthly
  contribution stability;
* causal prior-resolved 21-day EV admission followed by the canonical portfolio
  auction, using the pre-2025 SimplePolicyOptimiser winner outcomes.

This is a sequential screen.  Screening caps are explicit in the manifest;
finalists must be rerun with --full-caps before promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_causal_geometry_k9_c3_ablation as c3  # noqa: E402
from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from scripts.replay_strict_r3_forward_portfolio import _auction_candidates  # noqa: E402
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run  # noqa: E402
from scripts.run_tp6_sl4_exact170_canonical_consensus import _load_contract  # noqa: E402


SEED = 20260810
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
LEDGER = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_prequential_ledger_targetfree_long_2024_2026_20260809_v1/"
    "prequential_stack_ledger.parquet"
)
SOURCE_PANEL = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_source_panel_targetfree_long_2023_aug7_2026_20260809_v2/"
    "canonical_source_panel.parquet"
)
OPTIMISED_POLICY = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_optimised_policy_replay_targetfree_long_2025_aug7_2026_20260809_v1/"
    "candidate_policy_outcomes.parquet"
)
POLICY_WINNER = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/"
    "winner.json"
)

OVERLAY_CATALOG: dict[str, OverlaySpec] = {}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    training_months: int
    burnin_months: int = 3
    cadence_weeks: int = 0
    schedule: str = "monthly"

    def validate(self) -> None:
        if self.training_months < 1:
            raise ValueError("training_months must be positive")
        if self.burnin_months not in {2, 3}:
            raise ValueError("C3 burn-in is restricted to two or three months")
        if self.schedule not in {"monthly", "weeks"}:
            raise ValueError("schedule must be monthly or weeks")
        if self.schedule == "weeks" and self.cadence_weeks not in {2, 4, 8}:
            raise ValueError("weekly cadence must be 2, 4, or 8 weeks")


@dataclass(frozen=True)
class OverlaySpec:
    """One matched conversion-overlay arm on a fixed upstream/K9 contract."""

    name: str
    severe_target: str = "h12"
    severe_alpha: float = 0.5
    use_correctness: bool = True
    k9_soft_memberships: bool = True
    correctness_training_fraction: float = 1.0
    reliability_target: str = "positive_residual_gt100"
    reliability_integration: str = "positive_multiplier"
    reliability_alpha: float = 0.75
    include_global_recent: bool = False
    include_covariance_break: bool = False
    include_cross_model_state: bool = False
    k9_temperature_scale: float = 1.0

    def validate(self) -> None:
        if self.severe_target not in {"none", "h12"}:
            raise ValueError(
                "Severe-200 target is frozen to TP6/SL4 H12; use none or h12",
            )
        if not 0.0 <= self.severe_alpha <= 1.0:
            raise ValueError("severe_alpha must lie in [0, 1]")
        if not 0.0 < self.correctness_training_fraction <= 1.0:
            raise ValueError(
                "correctness_training_fraction must lie in (0, 1]",
            )
        if self.reliability_target not in {
            "positive_residual_gt100", "risk_residual_le_neg100",
            "risk_residual_le_neg200",
        }:
            raise ValueError(f"unknown reliability_target {self.reliability_target!r}")
        if self.reliability_integration not in {
            "positive_multiplier", "risk_demote",
        }:
            raise ValueError(
                f"unknown reliability_integration {self.reliability_integration!r}",
            )
        if not 0.0 <= self.reliability_alpha <= 1.0:
            raise ValueError("reliability_alpha must lie in [0, 1]")
        if self.k9_temperature_scale not in {0.25, 0.50, 0.75, 1.0}:
            raise ValueError("k9_temperature_scale must be 0.25, 0.50, 0.75, or 1")
        if (
            self.reliability_target.startswith("risk_")
            != (self.reliability_integration == "risk_demote")
        ):
            raise ValueError("risk targets require risk_demote integration")


OVERLAY_CATALOG.update({
    "upstream_only": OverlaySpec(
        "upstream_only", severe_target="none", severe_alpha=0.0,
        use_correctness=False, k9_soft_memberships=False,
    ),
    "correctness_only_no_k9": OverlaySpec(
        "correctness_only_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
    ),
    "correctness_top05_no_k9": OverlaySpec(
        "correctness_top05_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
        correctness_training_fraction=0.05,
    ),
    "correctness_top10_no_k9": OverlaySpec(
        "correctness_top10_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
        correctness_training_fraction=0.10,
    ),
    "correctness_top15_no_k9": OverlaySpec(
        "correctness_top15_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
        correctness_training_fraction=0.15,
    ),
    "correctness_top20_no_k9": OverlaySpec(
        "correctness_top20_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
        correctness_training_fraction=0.20,
    ),
    "correctness_top25_no_k9": OverlaySpec(
        "correctness_top25_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
        correctness_training_fraction=0.25,
    ),
    "correctness_top30_no_k9": OverlaySpec(
        "correctness_top30_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
        correctness_training_fraction=0.30,
    ),
    "correctness_top30_recent_no_k9": OverlaySpec(
        "correctness_top30_recent_no_k9", severe_target="none",
        severe_alpha=0.0, use_correctness=True,
        k9_soft_memberships=False, correctness_training_fraction=0.30,
        include_global_recent=True,
    ),
    "correctness_top30_cross_no_k9": OverlaySpec(
        "correctness_top30_cross_no_k9", severe_target="none",
        severe_alpha=0.0, use_correctness=True,
        k9_soft_memberships=False, correctness_training_fraction=0.30,
        include_cross_model_state=True,
    ),
    "correctness_top30_covariance_no_k9": OverlaySpec(
        "correctness_top30_covariance_no_k9", severe_target="none",
        severe_alpha=0.0, use_correctness=True,
        k9_soft_memberships=False, correctness_training_fraction=0.30,
        include_covariance_break=True,
    ),
    "correctness_top30_all_reliability_no_k9": OverlaySpec(
        "correctness_top30_all_reliability_no_k9", severe_target="none",
        severe_alpha=0.0, use_correctness=True,
        k9_soft_memberships=False, correctness_training_fraction=0.30,
        include_global_recent=True, include_covariance_break=True,
        include_cross_model_state=True,
    ),
    **{
        f"correctness_top30_k9temp{int(scale * 100):03d}_no_memberships": OverlaySpec(
            f"correctness_top30_k9temp{int(scale * 100):03d}_no_memberships",
            severe_target="none",
            severe_alpha=0.0,
            use_correctness=True,
            k9_soft_memberships=False,
            correctness_training_fraction=0.30,
            k9_temperature_scale=scale,
        )
        for scale in (0.75, 0.50, 0.25)
    },
    **{
        f"risk_top30_residual_le_neg{hurdle}_a{int(alpha * 100):03d}_no_k9": OverlaySpec(
            f"risk_top30_residual_le_neg{hurdle}_a{int(alpha * 100):03d}_no_k9",
            severe_target="none",
            severe_alpha=0.0,
            use_correctness=True,
            k9_soft_memberships=False,
            correctness_training_fraction=0.30,
            reliability_target=f"risk_residual_le_neg{hurdle}",
            reliability_integration="risk_demote",
            reliability_alpha=alpha,
            include_global_recent=True,
            include_covariance_break=True,
            include_cross_model_state=True,
        )
        for hurdle in (100, 200)
        for alpha in (0.25, 0.50, 0.75)
    },
    "correctness_top35_no_k9": OverlaySpec(
        "correctness_top35_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
        correctness_training_fraction=0.35,
    ),
    "correctness_top40_no_k9": OverlaySpec(
        "correctness_top40_no_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=False,
        correctness_training_fraction=0.40,
    ),
    "correctness_only_k9": OverlaySpec(
        "correctness_only_k9", severe_target="none", severe_alpha=0.0,
        use_correctness=True, k9_soft_memberships=True,
    ),
    "h12_no_k9_a010": OverlaySpec(
        "h12_no_k9_a010", severe_target="h12", severe_alpha=0.10,
        use_correctness=True, k9_soft_memberships=False,
    ),
    "h12_no_k9_a025": OverlaySpec(
        "h12_no_k9_a025", severe_target="h12", severe_alpha=0.25,
        use_correctness=True, k9_soft_memberships=False,
    ),
    "h12_no_k9_a050": OverlaySpec(
        "h12_no_k9_a050", severe_target="h12", severe_alpha=0.50,
        use_correctness=True, k9_soft_memberships=False,
    ),
    "h12_k9_a010": OverlaySpec(
        "h12_k9_a010", severe_target="h12", severe_alpha=0.10,
        use_correctness=True, k9_soft_memberships=True,
    ),
    "h12_k9_a025": OverlaySpec(
        "h12_k9_a025", severe_target="h12", severe_alpha=0.25,
        use_correctness=True, k9_soft_memberships=True,
    ),
    "h12_k9_a050": OverlaySpec(
        "h12_k9_a050", severe_target="h12", severe_alpha=0.50,
        use_correctness=True, k9_soft_memberships=True,
    ),
})


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _month_add(value: pd.Timestamp, offset: int) -> pd.Timestamp:
    return (value.tz_convert(None).to_period("M") + offset).to_timestamp().tz_localize("UTC")


def _fields() -> list[str]:
    fields = [str(value) for value in _load_contract()["long"]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("expected the frozen 120-field long contract")
    for path in (LEDGER, SOURCE_PANEL):
        names = set(pq.ParquetFile(path).schema.names)
        missing = sorted(set(fields) - names)
        if missing:
            raise ValueError(f"{path} lacks frozen fields: {missing[:10]}")
    return fields


def _read_ledger(start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "r3_class", "r3_label_available_ts", "policy_path_valid",
        "policy_label_available_ts", "policy_gross_bps", "policy_net_bps",
        "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        "prequential_base_score", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_residual_rank", "prequential_upstream", "stack_is_prequential",
        *fields,
    ]
    frame = pd.read_parquet(
        LEDGER,
        columns=columns,
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    for column in (
        "__ts__", "__decision_ts__", "r3_label_available_ts",
        "policy_label_available_ts", "h12_label_available_ts",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.loc[
        frame["side_name"].astype(str).str.lower().eq("long")
        & frame["stack_is_prequential"].fillna(False).astype(bool)
    ].copy()
    frame["r3_label_valid"] = frame["r3_class"].notna()
    frame["r3_clear"] = frame["r3_class"].eq(2).astype(np.int8)
    frame["__label_available_at__"] = frame["r3_label_available_ts"]
    frame["label_available_ts"] = frame["policy_label_available_ts"]
    frame["base_score"] = frame["prequential_base_score"]
    frame["base_rank"] = frame["prequential_base_rank42"]
    frame["base_anchor_bps"] = frame["prequential_base_anchor_bps"]
    frame["consensus_rank"] = frame["prequential_consensus_rank"]
    frame["final_score"] = frame["prequential_upstream"]
    frame["exact_h12_net_bps"] = frame["h12_tp6_sl4_net_bps"]
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("strict-prequential long ledger is empty or duplicated")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _apply_score_overrides(
    ledger: pd.DataFrame,
    *,
    path: Path,
    arm: str,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    overrides = pd.read_parquet(path)
    if "arm" in overrides:
        overrides = overrides.loc[overrides["arm"].astype(str).eq(arm)].copy()
    if overrides.empty or overrides["candidate_id"].duplicated().any():
        raise ValueError(f"score override {arm!r} is empty or duplicated")
    score_fields = [
        "prequential_p_adverse", "prequential_p_weak", "prequential_p_clear",
        "prequential_base_score", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_residual_rank", "prequential_upstream", "stack_is_prequential",
    ]
    available = [field for field in score_fields if field in overrides]
    available.extend(
        column for column in overrides.columns
        if column.startswith("conditional_head__") and column.endswith("__rank")
    )
    available = list(dict.fromkeys(available))
    required = {
        "prequential_base_rank42", "prequential_base_anchor_bps",
        "prequential_consensus_rank", "prequential_upstream",
    }
    if not required.issubset(available):
        raise ValueError(f"score override lacks required fields: {sorted(required - set(available))}")
    override_frame = overrides[["candidate_id", *available]].rename(
        columns={field: f"{field}__override" for field in available},
    )
    output = ledger.merge(
        override_frame, on="candidate_id", how="left",
        validate="one_to_one", indicator="__override_join__",
    )
    matched = output["__override_join__"].eq("both")
    for field in available:
        override = f"{field}__override"
        if field in output:
            output.loc[matched, field] = output.loc[matched, override]
        else:
            output[field] = output[override]
        output = output.drop(columns=override)
    output = output.drop(columns="__override_join__")
    # ``_read_ledger`` exposes convenient aliases before optional score
    # overrides are joined.  Refresh every downstream alias here so the C3
    # fit, Severe demotion, tail metrics and admission layer all consume the
    # selected distilled stack rather than the original ledger values.
    alias_map = {
        "base_score": "prequential_base_score",
        "base_rank": "prequential_base_rank42",
        "base_anchor_bps": "prequential_base_anchor_bps",
        "consensus_rank": "prequential_consensus_rank",
        "final_score": "prequential_upstream",
    }
    for alias, source in alias_map.items():
        if source in output:
            output[alias] = output[source]
    held = output["__decision_ts__"].ge(evaluation_start) & output["__decision_ts__"].lt(evaluation_end)
    held_matched = matched.loc[held]
    if not len(held_matched) or not held_matched.all():
        raise ValueError(
            f"score override {arm!r} covers {int(held_matched.sum())}/{len(held_matched)} held rows"
        )
    if not output.loc[held, "stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("score override introduced non-prequential held rows")
    return output, {
        "score_override": str(path), "score_override_arm": arm,
        "override_rows": int(matched.sum()), "held_override_rows": int(held_matched.sum()),
        "held_override_coverage": float(held_matched.mean()),
    }


def _read_geometry_source(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(
        SOURCE_PANEL,
        columns=["candidate_id", "__decision_ts__", *fields],
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError(f"geometry source is empty or duplicated for {start} to {end}")
    return frame


def _read_evaluation_policy(
    start: pd.Timestamp, end: pd.Timestamp, *, path: Path = OPTIMISED_POLICY,
) -> pd.DataFrame:
    requested = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
        "policy_exit_price", "policy_label_available_ts",
        "policy_outcome_source", "policy_proxy_resolution_minutes",
        "policy_market_data_source", "policy_market_data_quality",
    ]
    available = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in requested if column in available]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    for column in ("__ts__", "__decision_ts__", "policy_label_available_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("optimized-policy evaluation ledger is empty or duplicated")
    return frame


def _apply_training_policy_overrides(
    ledger: pd.DataFrame,
    *,
    path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replace historical policy labels by identity, retaining older rows."""
    available = set(pq.ParquetFile(path).schema.names)
    requested = [
        "candidate_id", "policy_path_valid", "policy_gross_bps",
        "policy_net_bps", "policy_label_available_ts",
        "policy_outcome_source", "policy_proxy_resolution_minutes",
    ]
    columns = [column for column in requested if column in available]
    required = {
        "candidate_id", "policy_path_valid", "policy_gross_bps",
        "policy_net_bps", "policy_label_available_ts",
    }
    if not required.issubset(columns):
        raise ValueError(
            f"training policy override lacks {sorted(required - set(columns))}",
        )
    override = pd.read_parquet(path, columns=columns)
    override["policy_label_available_ts"] = pd.to_datetime(
        override["policy_label_available_ts"], utc=True, errors="raise",
    )
    if override.empty or override["candidate_id"].duplicated().any():
        raise ValueError("training policy override is empty or duplicated")
    rename = {
        column: f"{column}__policy_override"
        for column in columns if column != "candidate_id"
    }
    output = ledger.merge(
        override.rename(columns=rename),
        on="candidate_id",
        how="left",
        validate="one_to_one",
        indicator="__policy_override_join__",
    )
    matched = output["__policy_override_join__"].eq("both")
    for column, renamed in rename.items():
        output.loc[matched, column] = output.loc[matched, renamed]
        output = output.drop(columns=renamed)
    output = output.drop(columns="__policy_override_join__")
    valid = (
        output["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(output["policy_net_bps"], errors="coerce"))
    )
    return output, {
        "training_policy_override": str(path),
        "training_policy_override_rows": int(matched.sum()),
        "training_policy_valid_rows_after_override": int(valid.sum()),
    }


def _overlay_fields(
    base_fields: Sequence[str],
    state_columns: Sequence[str],
    *,
    include_k9_soft_memberships: bool,
) -> list[str]:
    """Return a matched context contract with an explicit membership toggle."""
    memberships = {
        f"k09__cluster_{cluster:02d}__membership"
        for cluster in range(c3.K)
    }
    cluster_specific = {
        column for column in state_columns
        if column.startswith("k09__cluster_")
    }
    common = [
        column for column in state_columns if column not in cluster_specific
    ]
    selected = [*base_fields, *common]
    if include_k9_soft_memberships:
        selected.extend(
            column for column in state_columns if column in memberships
        )
    if len(selected) != len(set(selected)):
        raise ValueError("overlay feature contract contains duplicates")
    return selected


def _blocks(spec: ArmSpec, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if spec.schedule == "monthly":
        cutoffs = list(pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left"))
        return [(cutoff, min(cutoff + pd.offsets.MonthBegin(1), end)) for cutoff in cutoffs if cutoff < end]
    cutoffs = list(pd.date_range(start, end, freq=f"{spec.cadence_weeks * 7}D", inclusive="left"))
    return [
        (cutoff, min(cutoff + pd.Timedelta(weeks=spec.cadence_weeks), end))
        for cutoff in cutoffs if cutoff < end
    ]


def _admission_scoring_start(
    spec: ArmSpec,
    evaluation_start: pd.Timestamp,
    warmup_days: int,
) -> pd.Timestamp:
    """Keep evaluation cutoffs fixed while materialising prior admission rows."""
    if warmup_days < 0:
        raise ValueError("admission warm-up days cannot be negative")
    if warmup_days == 0:
        return evaluation_start
    if spec.schedule == "weeks":
        block_days = int(spec.cadence_weeks) * 7
        blocks = int(math.ceil(float(warmup_days) / float(block_days)))
        return evaluation_start - pd.Timedelta(days=blocks * block_days)
    months = int(math.ceil(float(warmup_days) / 28.0))
    return _month_add(evaluation_start, -months)


def _model_windows(spec: ArmSpec, cutoff: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    train_start = _month_add(cutoff, -spec.training_months)
    geometry_end = train_start
    geometry_start = _month_add(geometry_end, -spec.burnin_months)
    return geometry_start, geometry_end, train_start


def _cap_equal_month(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    return c3._equal_month_sample(frame, cap, seed=seed)


def _correctness_training_gate(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    retained_fraction: float,
) -> tuple[pd.Series, pd.Series, float]:
    """Select a causal pooled-global upper-rank domain for reliability fitting.

    The cutoff is one quantile of the complete *training* population, not a
    per-timestamp rank and not a percentile of the held window.  The resulting
    scalar is frozen before scoring reference/held rows, so future candidates
    cannot alter an earlier decision.
    """

    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must lie in (0, 1]")
    train_score = pd.to_numeric(train["final_score"], errors="coerce")
    finite = train_score[np.isfinite(train_score)]
    if finite.empty:
        raise ValueError("correctness training scores contain no finite values")
    if retained_fraction >= 1.0:
        floor = float("-inf")
    else:
        floor = float(
            np.quantile(
                finite.to_numpy(float),
                1.0 - retained_fraction,
                method="higher",
            )
        )
    train_mask = train_score.ge(floor)
    score_mask = pd.to_numeric(score["final_score"], errors="coerce").ge(floor)
    return train_mask, score_mask, floor


def _strict_asof_state(
    frame: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    """Join availability-indexed state strictly before each decision row."""

    left = frame[["__decision_ts__"]].copy()
    left["__row__"] = np.arange(len(left), dtype=np.int64)
    left["__decision_ts__"] = pd.to_datetime(
        left["__decision_ts__"], utc=True,
    )
    lookup = state.reset_index().rename(
        columns={state.index.name or "index": "__available_ts__"},
    )
    lookup["__available_ts__"] = pd.to_datetime(
        lookup["__available_ts__"], utc=True,
    )
    joined = pd.merge_asof(
        left.sort_values("__decision_ts__", kind="stable"),
        lookup.sort_values("__available_ts__", kind="stable"),
        left_on="__decision_ts__", right_on="__available_ts__",
        direction="backward", allow_exact_matches=False,
    )
    return (
        joined.sort_values("__row__", kind="stable")
        .drop(columns=["__decision_ts__", "__available_ts__", "__row__"])
        .reset_index(drop=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )


def _causal_reliability_context(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Materialise compact prior-resolved reliability state.

    Outcome-bearing fields are bucketed by ``policy_label_available_ts`` and
    joined with ``allow_exact_matches=False``.  Per-row disagreement is
    target-free and observable with the two upstream predictions.
    """

    output = pd.DataFrame(index=frame.index)
    base = pd.to_numeric(frame["base_rank"], errors="coerce").fillna(0.5)
    consensus = pd.to_numeric(
        frame["consensus_rank"], errors="coerce",
    ).fillna(0.5)
    output["reliability_base_consensus_gap"] = base - consensus
    output["reliability_base_consensus_abs_gap"] = (base - consensus).abs()
    output["reliability_base_consensus_mean"] = 0.5 * (base + consensus)
    output["reliability_upstream_rank"] = pd.to_numeric(
        frame["final_score"], errors="coerce",
    ).fillna(0.5)
    cross_fields = output.columns.tolist()

    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    anchor = pd.to_numeric(frame["base_anchor_bps"], errors="coerce")
    available = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    valid = (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & net.notna() & anchor.notna() & available.notna()
    )
    residual = (net - anchor).clip(-1_000.0, 1_000.0)
    events = pd.DataFrame(
        {
            "__available_ts__": available.loc[valid],
            "support": 1.0,
            "residual_sum": residual.loc[valid],
            "positive_sum": residual.loc[valid].gt(100.0).astype(float),
            "approx_sum": residual.loc[valid].abs().le(50.0).astype(float),
            "adverse100_sum": residual.loc[valid].le(-100.0).astype(float),
            "adverse200_sum": residual.loc[valid].le(-200.0).astype(float),
        }
    )
    recent_fields: list[str] = []
    covariance_fields: list[str] = []
    if events.empty:
        return output.astype(np.float32), {
            "cross_model": cross_fields,
            "global_recent": recent_fields,
            "covariance": covariance_fields,
        }

    bucket = events.groupby("__available_ts__", sort=True).sum().sort_index()
    state_parts: list[pd.DataFrame] = []
    for days in (3, 7, 14):
        rolled = bucket.rolling(f"{days}D", min_periods=1).sum()
        support = rolled["support"].replace(0.0, np.nan)
        part = pd.DataFrame(index=rolled.index)
        prefix = f"reliability_recent_{days}d_"
        part[prefix + "support"] = rolled["support"]
        part[prefix + "mean_residual_bps"] = rolled["residual_sum"] / support
        part[prefix + "positive_rate"] = rolled["positive_sum"] / support
        part[prefix + "approx_rate"] = rolled["approx_sum"] / support
        part[prefix + "adverse100_rate"] = rolled["adverse100_sum"] / support
        part[prefix + "adverse200_rate"] = rolled["adverse200_sum"] / support
        recent_fields.extend(part.columns.tolist())
        state_parts.append(part)
    recent_state = pd.concat(state_parts, axis=1)
    output = pd.concat(
        [output.reset_index(drop=True), _strict_asof_state(frame, recent_state)],
        axis=1,
    )

    covariance_events = pd.DataFrame(
        {
            "__available_ts__": available.loc[valid],
            "n": 1.0,
            "y": residual.loc[valid],
            "y2": residual.loc[valid] ** 2,
            "x_upstream": pd.to_numeric(
                frame.loc[valid, "final_score"], errors="coerce",
            ).fillna(0.5),
            "x_gap": (base - consensus).loc[valid],
        }
    )
    for field in ("x_upstream", "x_gap"):
        covariance_events[field + "2"] = covariance_events[field] ** 2
        covariance_events[field + "y"] = (
            covariance_events[field] * covariance_events["y"]
        )
    covariance_bucket = (
        covariance_events.groupby("__available_ts__", sort=True)
        .sum().sort_index()
    )

    def moments(days: int, field: str) -> tuple[pd.Series, pd.Series]:
        rolled = covariance_bucket.rolling(f"{days}D", min_periods=4).sum()
        n = rolled["n"].replace(0.0, np.nan)
        cov = rolled[field + "y"] / n - (rolled[field] / n) * (rolled["y"] / n)
        vx = rolled[field + "2"] / n - (rolled[field] / n) ** 2
        vy = rolled["y2"] / n - (rolled["y"] / n) ** 2
        corr = cov / np.sqrt((vx.clip(lower=0.0) * vy.clip(lower=0.0))).replace(0.0, np.nan)
        return cov, corr

    covariance_state = pd.DataFrame(index=covariance_bucket.index)
    for field, short_name in (("x_upstream", "upstream"), ("x_gap", "disagreement")):
        cov7, corr7 = moments(7, field)
        cov28, corr28 = moments(28, field)
        covariance_state[f"reliability_cov_break_{short_name}_7v28"] = cov7 - cov28
        covariance_state[f"reliability_corr_break_{short_name}_7v28"] = corr7 - corr28
    covariance_fields = covariance_state.columns.tolist()
    output = pd.concat(
        [output, _strict_asof_state(frame, covariance_state)], axis=1,
    )
    return output.astype(np.float32), {
        "cross_model": cross_fields,
        "global_recent": recent_fields,
        "covariance": covariance_fields,
    }


def _fit_reliability_overlay(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
    *,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if target_mode == "positive_residual_gt100":
        raw, rank, audit = c3._fit_correctness(train, score, fields)
        return raw, rank, {**audit, "reliability_target": target_mode}

    threshold = -100.0 if target_mode.endswith("neg100") else -200.0
    residual = (
        pd.to_numeric(train["policy_net_bps"], errors="coerce")
        - pd.to_numeric(train["base_anchor_bps"], errors="coerce")
    )
    target = residual.le(threshold).astype(np.int8).to_numpy()
    x_train, x_score = c3._impute_pair(train, score, fields)
    model = c3.LGBMClassifier(
        objective="binary", n_estimators=100, learning_rate=0.035,
        max_depth=3, num_leaves=7,
        min_child_samples=max(120, int(0.03 * len(train))),
        colsample_bytree=0.80, subsample=0.82, subsample_freq=1,
        reg_alpha=0.05, reg_lambda=5.0, max_bin=127,
        random_state=SEED + 223, n_jobs=4, verbosity=-1,
    ).fit(x_train, target)
    raw_train = model.predict_proba(x_train)[:, 1]
    raw_score = model.predict_proba(x_score)[:, 1]
    rank = c3._pct(raw_train, raw_score)
    return raw_score.astype(np.float32), rank, {
        "correctness_fit_rows": int(len(train)),
        "correctness_positive_rate": float(target.mean()),
        "correctness_features": int(len(fields)),
        "correctness_query": "none; shallow binary downside classifier",
        "reliability_target": target_mode,
        "reliability_threshold_bps": threshold,
    }


def _fit_one_block(
    spec: ArmSpec,
    overlay: OverlaySpec,
    *,
    cutoff: pd.Timestamp,
    held_end: pd.Timestamp,
    ledger: pd.DataFrame,
    geometry_source: pd.DataFrame,
    fields: Sequence[str],
    previous: c3.RawK9Bundle | None,
    model_cap: int,
    geometry_cap: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], c3.RawK9Bundle]:
    geometry_start, geometry_end, train_start = _model_windows(spec, cutoff)
    geometry_fit = geometry_source.loc[
        geometry_source["__decision_ts__"].ge(geometry_start)
        & geometry_source["__decision_ts__"].lt(geometry_end)
    ].copy()
    held = ledger.loc[
        ledger["__decision_ts__"].ge(cutoff) & ledger["__decision_ts__"].lt(held_end)
    ].copy()
    reference = ledger.loc[
        ledger["__decision_ts__"].ge(cutoff - pd.Timedelta(days=42))
        & ledger["__decision_ts__"].lt(cutoff)
    ].copy()
    meta_mask = (
        ledger["__decision_ts__"].ge(train_start)
        & ledger["__decision_ts__"].lt(cutoff)
        & ledger["policy_label_available_ts"].lt(cutoff)
        & ledger["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(ledger["policy_net_bps"], errors="coerce"))
    )
    if overlay.severe_target == "h12":
        meta_mask &= (
            ledger["h12_label_available_ts"].lt(cutoff)
            & ledger["h12_label_valid"].fillna(False).astype(bool)
            & np.isfinite(
                pd.to_numeric(ledger["h12_tp6_sl4_net_bps"], errors="coerce"),
            )
        )
    meta_train = ledger.loc[meta_mask].copy()
    leaf_train = ledger.loc[
        ledger["__decision_ts__"].ge(train_start)
        & ledger["__decision_ts__"].lt(cutoff)
        & ledger["r3_label_available_ts"].lt(cutoff)
        & ledger["r3_label_valid"].fillna(False).astype(bool)
    ].copy()
    if held.empty or len(reference) < 100 or len(meta_train) < 1_000 or len(leaf_train) < 1_000:
        raise ValueError(
            f"{spec.name} {cutoff}: insufficient held/reference/meta/leaf support "
            f"{len(held)}/{len(reference)}/{len(meta_train)}/{len(leaf_train)}"
        )
    if not meta_train["__decision_ts__"].ge(geometry_end).all():
        raise AssertionError("one downstream fit consumed rows from before its geometry definition")
    if not leaf_train["__decision_ts__"].ge(geometry_end).all():
        raise AssertionError("leaf support fit consumed rows from the geometry definition window")
    if not reference["__decision_ts__"].lt(cutoff).all() or not held["__decision_ts__"].ge(cutoff).all():
        raise AssertionError("geometry reference or held population crosses the model cutoff")

    old_geometry_cap, old_model_cap = c3.GEOMETRY_CAP, c3.MODEL_CAP
    c3.GEOMETRY_CAP, c3.MODEL_CAP = int(geometry_cap), int(model_cap)
    try:
        bundle_id = (
            f"{spec.name}__g{geometry_start:%Y%m%d}_{geometry_end:%Y%m%d}"
            f"__fit{cutoff:%Y%m%d}"
        )
        bundle, bundle_audit = c3._fit_raw_k9(
            geometry_fit,
            fields,
            bundle_id=bundle_id,
            fit_start=geometry_start,
            fit_end=geometry_end,
            source_kind="raw_complete_point_in_time_market_burnin",
            previous=previous,
            temperature_scale=overlay.k9_temperature_scale,
        )
        leaf_reference, leaf_audit = c3._fit_leaf_reference(leaf_train, fields)
    finally:
        c3.GEOMETRY_CAP, c3.MODEL_CAP = old_geometry_cap, old_model_cap

    score_population = (
        pd.concat([meta_train, reference, held], ignore_index=True)
        .drop_duplicates("candidate_id", keep="last")
        .sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        .reset_index(drop=True)
    )
    k9_values = bundle.transform(score_population)
    leaf_values = c3._leaf_state_from_reference(leaf_reference, score_population)
    base_state = c3._state_features(score_population, k9_values, leaf_values)
    reliability_context, reliability_groups = _causal_reliability_context(
        score_population,
    )
    state = pd.concat(
        [base_state.reset_index(drop=True), reliability_context.reset_index(drop=True)],
        axis=1,
    )
    state.index = score_population["candidate_id"].to_numpy()
    upstream_fields = [
        "base_score", "base_anchor_bps", "base_rank", "consensus_rank", "final_score",
    ]
    state_fields = _overlay_fields(
        upstream_fields,
        base_state.columns.tolist(),
        include_k9_soft_memberships=overlay.k9_soft_memberships,
    )
    if overlay.include_global_recent:
        state_fields.extend(reliability_groups["global_recent"])
    if overlay.include_covariance_break:
        state_fields.extend(reliability_groups["covariance"])
    if overlay.include_cross_model_state:
        state_fields.extend(reliability_groups["cross_model"])
    state_fields = list(dict.fromkeys(state_fields))

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        aligned = frame[["candidate_id"]].join(state, on="candidate_id")
        return frame.join(aligned.drop(columns="candidate_id"))

    meta_fit = attach(meta_train)
    reference_fit = attach(reference)
    held_fit = attach(held)
    meta_fit = _cap_equal_month(meta_fit, model_cap, SEED + 241)
    scored = pd.concat(
        [reference_fit.assign(__score_role__="reference"), held_fit.assign(__score_role__="held")],
        ignore_index=True,
    )
    if overlay.severe_target == "none":
        severe = np.zeros(len(scored), dtype=np.float32)
        severe_audit = {
            "safety_fit_rows": 0,
            "safety_positive_rate": np.nan,
            "safety_features": int(len(state_fields)),
            "safety_target": "disabled",
            "safety_threshold_bps": -200.0,
        }
    else:
        severe, severe_audit = c3._fit_safety(
            meta_fit,
            scored,
            state_fields,
        )
    scored["severe200_probability"] = severe
    scored["raw_severe"] = scored["final_score"].to_numpy(float) * (
        1.0 - float(overlay.severe_alpha) * severe
    )
    correctness_train_mask, correctness_score_mask, correctness_gate = (
        _correctness_training_gate(
            meta_fit,
            scored,
            retained_fraction=overlay.correctness_training_fraction,
        )
    )
    correctness_fit = meta_fit.loc[correctness_train_mask].copy()
    if overlay.use_correctness:
        if len(correctness_fit) < 1_000:
            raise ValueError(
                f"{spec.name} {cutoff}: correctness top-"
                f"{overlay.correctness_training_fraction:.0%} gate has only "
                f"{len(correctness_fit)} rows",
            )
        correctness_raw, correctness_rank, correctness_audit = (
            _fit_reliability_overlay(
                correctness_fit,
                scored,
                state_fields,
                target_mode=overlay.reliability_target,
            )
        )
    else:
        correctness_raw = np.zeros(len(scored), dtype=np.float32)
        correctness_rank = np.ones(len(scored), dtype=np.float32)
        correctness_audit = {
            "correctness_fit_rows": 0,
            "correctness_positive_rate": np.nan,
            "correctness_features": int(len(state_fields)),
            "correctness_query": "disabled",
        }
    scored["correctness_raw"] = correctness_raw
    scored["correctness_rank"] = correctness_rank
    scored["correctness_gate_active"] = correctness_score_mask
    if overlay.reliability_integration == "positive_multiplier":
        active_multiplier = (
            1.0 - overlay.reliability_alpha * (1.0 - correctness_rank)
        )
    else:
        active_multiplier = 1.0 - overlay.reliability_alpha * correctness_rank
    correctness_multiplier = np.where(
        scored["correctness_gate_active"].to_numpy(bool),
        active_multiplier,
        1.0,
    )
    scored["raw_correctness_demote"] = (
        scored["raw_severe"].to_numpy(float) * correctness_multiplier
    )
    reference_mask = scored["__score_role__"].eq("reference").to_numpy()
    scored["final_score"] = c3._pct(
        scored.loc[reference_mask, "raw_correctness_demote"].to_numpy(float),
        scored["raw_correctness_demote"].to_numpy(float),
    )
    output = scored.loc[~reference_mask].copy()
    output["arm"] = f"{spec.name}__{overlay.name}"
    output["window_arm"] = spec.name
    output["overlay_arm"] = overlay.name
    output["model_cutoff"] = cutoff
    output["model_held_end_exclusive"] = held_end
    output["geometry_bundle_sha256"] = bundle_audit["bundle_sha256"]
    output["geometry_bundle_id"] = bundle.bundle_id
    output["geometry_fit_start"] = geometry_start
    output["geometry_fit_end_exclusive"] = geometry_end
    output["meta_training_start"] = train_start
    output["training_months"] = spec.training_months
    output["burnin_months"] = spec.burnin_months
    output["cadence_weeks"] = spec.cadence_weeks
    audit = {
        "arm": spec.name,
        "overlay_arm": overlay.name,
        "model_cutoff": cutoff,
        "held_end_exclusive": held_end,
        "held_rows": int(len(output)),
        "reference_rows": int(len(reference)),
        "meta_fit_rows": int(len(meta_fit)),
        "leaf_fit_source_rows": int(len(leaf_train)),
        "geometry_source_rows": int(len(geometry_fit)),
        "geometry_bundle_sha256": bundle_audit["bundle_sha256"],
        "geometry_fit_start": geometry_start,
        "geometry_fit_end_exclusive": geometry_end,
        "meta_training_start": train_start,
        "single_geometry_per_downstream_fit": True,
        "geometry_lineage_rule": (
            "geometry/K9 fit rows end before downstream meta and leaf-support training; "
            "one frozen bundle transforms train, reference, and held rows for this fit"
        ),
        "held_outcomes_consumed": False,
        "overlay_refit_at_cutoff": True,
        "overlay_refit_cadence_weeks": (
            spec.cadence_weeks if spec.schedule == "weeks" else 4
        ),
        "severe_target_mode": overlay.severe_target,
        "severe_alpha": overlay.severe_alpha,
        "correctness_enabled": overlay.use_correctness,
        "correctness_training_fraction": overlay.correctness_training_fraction,
        "correctness_training_rank_floor": correctness_gate,
        "correctness_training_rows_after_rank_gate": int(len(correctness_fit)),
        "correctness_scored_active_fraction": float(
            scored["correctness_gate_active"].mean()
        ),
        "k9_soft_memberships_enabled": overlay.k9_soft_memberships,
        "reliability_target": overlay.reliability_target,
        "reliability_integration": overlay.reliability_integration,
        "reliability_alpha": overlay.reliability_alpha,
        "include_global_recent": overlay.include_global_recent,
        "include_covariance_break": overlay.include_covariance_break,
        "include_cross_model_state": overlay.include_cross_model_state,
        "global_recent_feature_count": len(reliability_groups["global_recent"]),
        "covariance_feature_count": len(reliability_groups["covariance"]),
        "cross_model_feature_count": len(reliability_groups["cross_model"]),
        "overlay_feature_count": len(state_fields),
        **leaf_audit,
        **severe_audit,
        **correctness_audit,
    }
    bundle_audit.update(
        {
            "arm": spec.name,
            "overlay_arm": overlay.name,
            "model_cutoff": cutoff,
            "training_months": spec.training_months,
            "burnin_months": spec.burnin_months,
        }
    )
    return output, audit, bundle_audit, bundle


def _global_tail_metrics(frame: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_population = frame.loc[
        np.isfinite(pd.to_numeric(frame["final_score"], errors="coerce"))
    ].copy()
    rows: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    for tail in TAILS:
        count = max(1, int(math.ceil(tail * len(score_population))))
        selected_score = score_population.nlargest(count, "final_score", keep="first").copy()
        valid = selected_score.loc[
            selected_score["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(selected_score["policy_net_bps"], errors="coerce"))
        ].copy()
        rows.append(
            {
                "arm": arm,
                "tail": tail,
                "population_rows": int(len(score_population)),
                "selected_score_rows": int(len(selected_score)),
                "valid_outcomes": int(len(valid)),
                "outcome_coverage": float(len(valid) / max(len(selected_score), 1)),
                "trades": int(len(valid)),
                "gross_bps_per_trade": float(valid["policy_gross_bps"].mean()),
                "net_bps_per_trade": float(valid["policy_net_bps"].mean()),
                "positive_rate": float(valid["policy_net_bps"].gt(0).mean()),
                "trades_per_calendar_day": float(
                    len(selected_score)
                    / max(
                        (
                            selected_score["__decision_ts__"].max().normalize()
                            - selected_score["__decision_ts__"].min().normalize()
                        ).days
                        + 1,
                        1,
                    )
                ),
            }
        )
        for month, selected_month in selected_score.groupby(
            selected_score["__decision_ts__"].dt.to_period("M").astype(str), sort=True,
        ):
            block = selected_month.loc[
                selected_month["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected_month["policy_net_bps"], errors="coerce"))
            ]
            monthly.append(
                {
                    "arm": arm,
                    "tail": tail,
                    "month": month,
                    "selected_score_rows": int(len(selected_month)),
                    "valid_outcomes": int(len(block)),
                    "outcome_coverage": float(len(block) / max(len(selected_month), 1)),
                    "trades": int(len(block)),
                    "gross_bps_per_trade": float(block["policy_gross_bps"].mean()),
                    "net_bps_per_trade": float(block["policy_net_bps"].mean()),
                    "positive_rate": float(block["policy_net_bps"].gt(0).mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def _global_tail_outcome_source_metrics(
    frame: pd.DataFrame,
    arm: str,
) -> pd.DataFrame:
    population = frame.loc[
        np.isfinite(pd.to_numeric(frame["final_score"], errors="coerce"))
    ].copy()
    rows: list[dict[str, object]] = []
    for tail in TAILS:
        count = max(1, int(math.ceil(tail * len(population))))
        selected = population.nlargest(count, "final_score", keep="first")
        valid = selected.loc[
            selected["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
        ].copy()
        if "policy_outcome_source" not in valid:
            valid["policy_outcome_source"] = "unspecified"
        for source, block in valid.groupby("policy_outcome_source", dropna=False):
            rows.append({
                "arm": arm,
                "tail": tail,
                "policy_outcome_source": str(source),
                "trades": int(len(block)),
                "share_of_valid_tail": float(len(block) / max(len(valid), 1)),
                "gross_bps_per_trade": float(block["policy_gross_bps"].mean()),
                "net_bps_per_trade": float(block["policy_net_bps"].mean()),
                "positive_rate": float(block["policy_net_bps"].gt(0.0).mean()),
            })
    return pd.DataFrame(rows)


def _compact_prediction_artifact(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the 120 raw model inputs after scoring.

    The immutable replay artifact needs identities, score decomposition, C3 state,
    geometry lineage, and realised policy outcomes.  Persisting the full upstream
    feature matrix once per ablation arm multiplies a source contract that already
    lives in the prequential ledger and made six-arm screens unnecessarily large.
    """

    exact = {
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "prequential_base_score", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_residual_rank", "prequential_upstream", "stack_is_prequential",
        "base_score", "base_rank", "base_anchor_bps", "consensus_rank",
        "severe200_probability", "raw_severe", "correctness_raw",
        "correctness_rank", "correctness_gate_active",
        "raw_correctness_demote", "final_score",
        "arm", "window_arm", "overlay_arm",
        "model_cutoff", "model_held_end_exclusive",
        "geometry_bundle_sha256", "geometry_bundle_id", "geometry_fit_start",
        "geometry_fit_end_exclusive", "meta_training_start", "training_months",
        "burnin_months", "cadence_weeks", "policy_path_valid",
        "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_exit_reason", "policy_entry_price", "policy_exit_price",
        "policy_label_available_ts", "policy_outcome_source",
        "policy_proxy_resolution_minutes", "policy_market_data_source",
        "policy_market_data_quality",
        "is_admission_warmup",
    }
    prefixes = ("k09__", "k9_", "leaf_", "geometry_", "conditional_head__")
    keep = [name for name in frame.columns if name in exact or name.startswith(prefixes)]
    missing = sorted(
        name
        for name in (
            "candidate_id", "__decision_ts__", "__symbol__", "final_score",
            "policy_path_valid", "policy_net_bps",
        )
        if name not in keep
    )
    if missing:
        raise ValueError(f"compact prediction artifact missing required fields: {missing}")
    return frame.loc[:, keep].copy()


def _persist_structural_baseline(
    bundle: c3.RawK9Bundle,
    path: Path,
) -> None:
    """Persist the exact training-frozen state used by structural diagnostics."""

    required = (
        "structural_projection", "structural_mean", "structural_covariance",
        "structural_correlation", "cluster_structural_mean",
        "cluster_structural_covariance", "cluster_structural_correlation",
        "cluster_structural_support",
    )
    missing = [field for field in required if getattr(bundle, field, None) is None]
    if missing:
        raise ValueError(f"geometry bundle lacks structural baseline state: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"structural baseline already exists: {path}")
    np.savez_compressed(
        path,
        **{field: np.asarray(getattr(bundle, field)) for field in required},
        fields=np.asarray(bundle.fields, dtype=str),
        fit_start=np.asarray([bundle.fit_start.isoformat()]),
        fit_end=np.asarray([bundle.fit_end.isoformat()]),
        bundle_id=np.asarray([bundle.bundle_id]),
    )


def _portability(values: Iterable[float]) -> tuple[float, float, float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("-inf"), np.nan, np.nan, np.nan
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    worst = float(np.min(x))
    return median - 0.5 * mad - max(0.0, -worst), median, mad, worst


def _stability(monthly: pd.DataFrame, global_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = ["arm"]
    if "source_artifact" in monthly and "source_artifact" in global_metrics:
        grouping = ["source_artifact", "arm"]
    for key, block in monthly.loc[monthly["tail"].eq(0.02)].groupby(grouping, sort=True):
        if len(grouping) == 2:
            source_artifact, arm = key
        else:
            source_artifact = None
            arm = key[0] if isinstance(key, tuple) else key
        score, median, mad, worst = _portability(block["net_bps_per_trade"])
        mask = global_metrics["arm"].eq(arm) & global_metrics["tail"].eq(0.02)
        if source_artifact is not None:
            mask &= global_metrics["source_artifact"].eq(source_artifact)
        pooled = global_metrics.loc[mask, "net_bps_per_trade"]
        row = {
                "arm": arm,
                "top2_portability_score": score,
                "top2_pooled_net_bps": float(pooled.iloc[0]) if len(pooled) else np.nan,
                "top2_month_median_net_bps": median,
                "top2_month_mad_bps": mad,
                "top2_worst_month_net_bps": worst,
                "top2_positive_months": int(block["net_bps_per_trade"].gt(0).sum()),
                "top2_months": int(len(block)),
        }
        if source_artifact is not None:
            row["source_artifact"] = source_artifact
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["top2_portability_score", "top2_pooled_net_bps"],
        ascending=False,
        kind="stable",
    )


def _portfolio(
    frame: pd.DataFrame,
    *,
    arm: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    admitted, admission_audit = apply_causal_21d_side_admission(
        frame,
        score_column="final_score",
        net_column="policy_net_bps",
        decision_column="__decision_ts__",
        label_available_column="policy_label_available_ts",
        spec=Causal21dAdmissionSpec(
            mode="hierarchical_tail_side_shrinkage_v2",
        ),
    )
    evaluation = admitted.loc[
        admitted["__decision_ts__"].ge(start) & admitted["__decision_ts__"].lt(end)
    ].copy()
    try:
        candidates = _auction_candidates(evaluation)
    except ValueError:
        return (
            pd.DataFrame(
                [
                    {
                        "arm": arm,
                        "accepted_trades": 0,
                        "accepted_outcome_unavailable": 0,
                        "trades_per_calendar_day": 0.0,
                        "gross_bps_per_trade": np.nan,
                        "net_bps_per_trade": np.nan,
                        "positive_rate": np.nan,
                        "max_drawdown": np.nan,
                    }
                ]
            ),
            admission_audit,
            pd.DataFrame(),
        )
    decisions, _, monthly, summary = _run(
        candidates,
        0.0,
        arm,
        initial_wallet=1_000.0,
        perp_leverage=7.0,
        margin_slot_wallet_fraction=0.10,
    )
    days = max((end - start).total_seconds() / 86_400.0, 1.0)
    replay_summary = summary.get("replay_metric_summary", {})
    if isinstance(replay_summary, str):
        replay_summary = json.loads(replay_summary)
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)]
    accepted_unavailable = (
        ~accepted["policy_outcome_available"].fillna(False).astype(bool)
        if "policy_outcome_available" in accepted
        else pd.Series(False, index=accepted.index)
    )
    result = pd.DataFrame(
        [
            {
                "arm": arm,
                "accepted_trades": int(summary["accepted_trades"]),
                "accepted_outcome_unavailable": int(accepted_unavailable.sum()),
                "trades_per_calendar_day": float(summary["accepted_trades"] / days),
                "gross_bps_per_trade": float(summary["gross_bps_per_trade"]),
                "net_bps_per_trade": float(summary["net_bps_per_trade"]),
                "positive_rate": float(summary["positive_rate"]),
                "max_drawdown": float(replay_summary.get("max_drawdown", np.nan)),
            }
        ]
    )
    monthly = monthly.assign(arm=arm)
    return result, admission_audit.assign(arm=arm), monthly


def _arm_specs(args: argparse.Namespace) -> list[ArmSpec]:
    if args.phase == "window":
        return [
            ArmSpec(name=f"window_{months}m_burn{args.burnin_months}m_monthly", training_months=months, burnin_months=args.burnin_months)
            for months in args.training_months
        ]
    if args.phase == "cadence":
        if len(args.training_months) != 1:
            raise ValueError("cadence phase requires exactly one training length")
        return [
            ArmSpec(
                name=f"window_{args.training_months[0]}m_burn{args.burnin_months}m_cadence{weeks}w",
                training_months=args.training_months[0],
                burnin_months=args.burnin_months,
                cadence_weeks=weeks,
                schedule="weeks",
            )
            for weeks in args.cadence_weeks
        ]
    if len(args.training_months) != 1:
        raise ValueError("burnin phase requires exactly one training length")
    return [
        ArmSpec(
            name=f"window_{args.training_months[0]}m_burn{burnin}m_cadence{args.cadence_weeks[0]}w",
            training_months=args.training_months[0],
            burnin_months=burnin,
            cadence_weeks=args.cadence_weeks[0],
            schedule="weeks",
        )
        for burnin in (2, 3)
    ]


def _overlay_specs(value: str) -> list[OverlaySpec]:
    names = [token.strip() for token in value.split(",") if token.strip()]
    unknown = sorted(set(names) - set(OVERLAY_CATALOG))
    if unknown:
        raise ValueError(f"unknown overlay arms: {unknown}")
    overlays = [OVERLAY_CATALOG[name] for name in names]
    for overlay in overlays:
        overlay.validate()
    return overlays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("window", "cadence", "burnin"), default="window")
    parser.add_argument("--training-months", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--burnin-months", type=int, default=3)
    parser.add_argument("--cadence-weeks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--evaluation-start", default="2025-01-01")
    parser.add_argument("--evaluation-end", default="2025-08-01")
    parser.add_argument("--geometry-cap", type=int, default=30_000)
    parser.add_argument("--model-cap", type=int, default=80_000)
    parser.add_argument(
        "--admission-warmup-days", type=int, default=84,
        help=(
            "Score prior rows before the evaluation boundary so the 21/42/84-day "
            "causal EV map never starts from an artificial empty reference."
        ),
    )
    parser.add_argument("--full-caps", action="store_true")
    parser.add_argument(
        "--skip-portfolio", action="store_true",
        help=(
            "Skip the expensive causal admission/portfolio pass during a "
            "score-tail screening funnel. Finalists must be replayed without it."
        ),
    )
    parser.add_argument("--policy-outcomes", type=Path, default=OPTIMISED_POLICY)
    parser.add_argument(
        "--training-policy-outcomes", type=Path, default=None,
        help="Optional repaired policy labels joined into downstream training by candidate ID.",
    )
    parser.add_argument(
        "--overlay-arms",
        default=(
            "correctness_only_no_k9,correctness_only_k9,"
            "h12_no_k9_a025,h12_k9_a010,h12_k9_a025,h12_k9_a050"
        ),
    )
    parser.add_argument("--score-overrides", type=Path, default=None)
    parser.add_argument("--score-arm", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    if args.full_caps:
        args.geometry_cap, args.model_cap = 100_000, 240_000
    specs = _arm_specs(args)
    for spec in specs:
        spec.validate()
    overlays = _overlay_specs(args.overlay_arms)
    start, end = _utc(args.evaluation_start), _utc(args.evaluation_end)
    scoring_starts = {
        spec.name: _admission_scoring_start(
            spec, start, int(args.admission_warmup_days),
        )
        for spec in specs
    }
    earliest = min(
        _model_windows(spec, scoring_starts[spec.name])[0] for spec in specs
    )
    earliest_scoring_start = min(scoring_starts.values())
    fields = _fields()
    ledger = _read_ledger(earliest, end, fields)
    policy_override_audit: dict[str, object] = {}
    if args.training_policy_outcomes is not None:
        ledger, policy_override_audit = _apply_training_policy_overrides(
            ledger, path=args.training_policy_outcomes,
        )
    override_audit: dict[str, object] = {}
    if args.score_overrides is not None:
        if not args.score_arm:
            raise ValueError("--score-arm is required with --score-overrides")
        ledger, override_audit = _apply_score_overrides(
            ledger, path=args.score_overrides, arm=args.score_arm,
            evaluation_start=start, evaluation_end=end,
        )
        override_audit["admission_warmup_score_source"] = (
            "strict prequential ledger where the selected override has no prior rows"
        )
    geometry_source = _read_geometry_source(earliest, end, fields)
    evaluation_policy = _read_evaluation_policy(
        earliest_scoring_start, end, path=args.policy_outcomes,
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)

    best_prediction: pd.DataFrame | None = None
    best_prediction_key = (float("-inf"), float("-inf"))
    all_fold_audits: list[dict[str, object]] = []
    all_bundle_audits: list[dict[str, object]] = []
    global_metrics: list[pd.DataFrame] = []
    monthly_metrics: list[pd.DataFrame] = []
    outcome_source_metrics: list[pd.DataFrame] = []
    portfolio_metrics: list[pd.DataFrame] = []
    admission_audits: list[pd.DataFrame] = []
    portfolio_monthly: list[pd.DataFrame] = []

    for spec in specs:
      for overlay in overlays:
        arm_name = f"{spec.name}__{overlay.name}"
        scoring_start = scoring_starts[spec.name]
        print(json.dumps({
            "event": "arm_start", **asdict(spec), "overlay": asdict(overlay),
        }), flush=True)
        previous: c3.RawK9Bundle | None = None
        arm_parts: list[pd.DataFrame] = []
        for cutoff, held_end in _blocks(spec, scoring_start, end):
            part, fold_audit, bundle_audit, previous = _fit_one_block(
                spec,
                overlay,
                cutoff=cutoff,
                held_end=held_end,
                ledger=ledger,
                geometry_source=geometry_source,
                fields=fields,
                previous=previous,
                model_cap=args.model_cap,
                geometry_cap=args.geometry_cap,
            )
            _persist_structural_baseline(
                previous,
                args.out_dir / "structural_baselines" / arm_name
                / f"{cutoff:%Y%m%d}.npz",
            )
            arm_parts.append(part)
            all_fold_audits.append(fold_audit)
            all_bundle_audits.append(bundle_audit)
            print(json.dumps({
                "event": "fold_complete", "arm": arm_name,
                "cutoff": cutoff.isoformat(), "rows": len(part),
            }), flush=True)
        arm_prediction = pd.concat(arm_parts, ignore_index=True)
        expected_rows = len(arm_prediction)
        arm_prediction = arm_prediction.merge(
            evaluation_policy,
            on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"],
            how="left",
            validate="one_to_one",
            suffixes=("_train_policy", ""),
            indicator="__evaluation_policy_join__",
        )
        unmatched = arm_prediction["__evaluation_policy_join__"].ne("both")
        arm_prediction = arm_prediction.drop(columns="__evaluation_policy_join__")
        if len(arm_prediction) != expected_rows:
            raise AssertionError("evaluation policy join changed held score population")
        warmup_unmatched = unmatched & arm_prediction["__decision_ts__"].lt(start)
        evaluation_unmatched = unmatched & ~warmup_unmatched
        policy_payload = [
            "policy_path_valid", "policy_gross_bps", "policy_net_bps",
            "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
            "policy_exit_price", "policy_label_available_ts",
            "policy_outcome_source", "policy_proxy_resolution_minutes",
            "policy_market_data_source", "policy_market_data_quality",
        ]
        for column in policy_payload:
            prior_column = f"{column}_train_policy"
            if prior_column not in arm_prediction:
                continue
            fill = warmup_unmatched & arm_prediction[prior_column].notna()
            arm_prediction.loc[fill, column] = arm_prediction.loc[fill, prior_column]
        if evaluation_unmatched.any():
            arm_prediction.loc[evaluation_unmatched, "policy_path_valid"] = False
            arm_prediction.loc[evaluation_unmatched, "policy_outcome_source"] = "unavailable"
        arm_prediction["is_admission_warmup"] = arm_prediction[
            "__decision_ts__"
        ].lt(start)
        if arm_prediction["candidate_id"].duplicated().any():
            raise ValueError(f"{arm_name}: duplicate held prediction identities")
        evaluation_prediction = arm_prediction.loc[
            arm_prediction["__decision_ts__"].ge(start)
            & arm_prediction["__decision_ts__"].lt(end)
        ].copy()
        global_part, monthly_part = _global_tail_metrics(
            evaluation_prediction, arm_name,
        )
        global_metrics.append(global_part)
        monthly_metrics.append(monthly_part)
        outcome_source_metrics.append(
            _global_tail_outcome_source_metrics(evaluation_prediction, arm_name),
        )
        if args.skip_portfolio:
            portfolio_part = pd.DataFrame([{
                "arm": arm_name,
                "status": "SKIPPED_SCREEN_ONLY",
                "accepted_trades": np.nan,
                "accepted_outcome_unavailable": np.nan,
                "trades_per_calendar_day": np.nan,
                "gross_bps_per_trade": np.nan,
                "net_bps_per_trade": np.nan,
                "positive_rate": np.nan,
                "max_drawdown": np.nan,
            }])
            admission_part = pd.DataFrame()
            portfolio_month_part = pd.DataFrame()
        else:
            portfolio_part, admission_part, portfolio_month_part = _portfolio(
                arm_prediction,
                arm=arm_name,
                start=start,
                end=end,
            )
        portfolio_metrics.append(portfolio_part)
        if not admission_part.empty:
            admission_audits.append(admission_part)
        portfolio_monthly.append(portfolio_month_part)
        arm_stability = _stability(monthly_part, global_part).iloc[0]
        arm_key = (
            float(arm_stability["top2_portability_score"]),
            float(arm_stability["top2_pooled_net_bps"]),
        )
        if arm_key > best_prediction_key:
            best_prediction = _compact_prediction_artifact(arm_prediction)
            best_prediction_key = arm_key
        print(json.dumps({
            "event": "arm_complete", "arm": arm_name,
            "rows": len(evaluation_prediction),
            "admission_warmup_rows": int(arm_prediction["is_admission_warmup"].sum()),
        }), flush=True)

    if best_prediction is None:
        raise RuntimeError("no overlay arm produced a prediction artifact")
    predictions = best_prediction
    globals_frame = pd.concat(global_metrics, ignore_index=True)
    monthly_frame = pd.concat(monthly_metrics, ignore_index=True)
    stability = _stability(monthly_frame, globals_frame)
    portfolio_frame = pd.concat(portfolio_metrics, ignore_index=True)
    predictions.to_parquet(args.out_dir / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(all_fold_audits).to_parquet(args.out_dir / "fold_audit.parquet", index=False)
    pd.DataFrame(all_bundle_audits).to_parquet(args.out_dir / "geometry_bundle_audit.parquet", index=False)
    globals_frame.to_parquet(args.out_dir / "metrics_global.parquet", index=False)
    monthly_frame.to_parquet(args.out_dir / "metrics_monthly_global_tail_contribution.parquet", index=False)
    pd.concat(outcome_source_metrics, ignore_index=True).to_parquet(
        args.out_dir / "metrics_global_tail_by_outcome_source.parquet",
        index=False,
    )
    stability.to_parquet(args.out_dir / "top2_stability.parquet", index=False)
    portfolio_frame.to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    if admission_audits:
        pd.concat(admission_audits, ignore_index=True).to_parquet(
            args.out_dir / "causal_21d_admission_audit.parquet", index=False,
        )
    nonempty_monthly = [frame for frame in portfolio_monthly if not frame.empty]
    if nonempty_monthly:
        pd.concat(nonempty_monthly, ignore_index=True).to_parquet(
            args.out_dir / "portfolio_monthly_metrics.parquet", index=False,
        )
    winner = stability.iloc[0].to_dict()
    manifest = {
        "schema": "strict_r3_c3_window_cadence_ablation_v2",
        "side": "long",
        "phase": args.phase,
        "arms": [asdict(spec) for spec in specs],
        "overlay_arms": [asdict(overlay) for overlay in overlays],
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "admission_warmup_days": int(args.admission_warmup_days),
        "admission_scoring_start_by_arm": {
            name: value.isoformat() for name, value in scoring_starts.items()
        },
        "screening_caps": not args.full_caps,
        "portfolio_skipped_for_screen": bool(args.skip_portfolio),
        "geometry_cap": args.geometry_cap,
        "model_cap": args.model_cap,
        "upstream": str(LEDGER),
        "geometry_source": str(SOURCE_PANEL),
        "policy_outcomes": str(args.policy_outcomes),
        **policy_override_audit,
        **override_audit,
        "evaluation_policy": str(args.policy_outcomes),
        "simple_policy_optimiser_winner": str(POLICY_WINNER),
        "ranking": "one pooled global ranking; monthly rows are contributions from that selected set",
        "prediction_artifact_scope": "winning overlay arm only; metrics retain every tested arm",
        "selection": "Top-2 portability score, then pooled Top-2 net EV",
        "admission": "causal prior-resolved hierarchical 21/42/84-day uneven-tail EV map >= +50 bps; fail closed",
        "portfolio": "8 concurrent; 2 new per 15m bar; 1 per asset; 80% margin; 7x",
        "geometry_membership_temperature": "fit population only; frozen per bundle",
        "single_geometry_per_downstream_fit": True,
        "winner": winner,
        "seed": SEED,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "winner": winner, "out": str(args.out_dir)}, default=str))


if __name__ == "__main__":
    main()
