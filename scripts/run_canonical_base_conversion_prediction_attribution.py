#!/usr/bin/env python3
"""Join causal OOF conversion predictions to frozen base-score economics.

This is an explanatory March/April attribution.  It preserves the canonical
monthly pooled-global base top-k selections, joins only OOF conversion
predictions by hour/side/frozen score decile, and tests whether those predicted
states stratify realized opportunity, upside, downside and net economics.  It
does not rerank candidates or authorize an admission interaction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scripts.materialize_canonical_economic_conversion_transition_labels import (
        add_frozen_causal_score_deciles,
        sha256,
    )
    from scripts.run_canonical_base_ic_ev_tail_diagnostic import _corr
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from materialize_canonical_economic_conversion_transition_labels import (
        add_frozen_causal_score_deciles,
        sha256,
    )
    from run_canonical_base_ic_ev_tail_diagnostic import _corr


ROOT = Path(__file__).resolve().parents[1]
PANEL_SOURCE = (
    ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
)
FEATURE_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_economic_conversion_transition_feature_group_ablation_20260729_v1"
)
TARGET_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_economic_conversion_transition_target_ablation_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_base_conversion_prediction_attribution_20260729_v1"
)
SCHEMA = "canonical_base_conversion_prediction_attribution_v1"
COHORT_KEY = ("cohort_anchor_utc", "side_name", "frozen_base_score_decile")
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
REFERENCE_MONTH = "2025-03"
EVALUATION_MONTH = "2025-04"
HEADS = {
    "opportunity_full_context": (
        "feature",
        "full_context",
        "opportunity_probability_0bps",
    ),
    "direct_market_and_regime": (
        "feature",
        "market_and_regime",
        "direct_mean_net",
    ),
    "direct_score_and_regime": (
        "feature",
        "score_and_regime",
        "direct_mean_net",
    ),
    "adverse_market_and_regime": (
        "feature",
        "market_and_regime",
        "adverse_severity_robust_mean",
    ),
    "upside_robust_full_context": (
        "target",
        "B1R_robust_positive_contribution",
        "",
    ),
    "upside_raw_full_context": (
        "target",
        "B1_unconditional_positive_contribution",
        "",
    ),
    "downside_raw_full_context": (
        "target",
        "B3_unconditional_loss_contribution",
        "",
    ),
}


def _verify_artifact(
    root: Path, schema: str, material_names: Iterable[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"immutable artifact is incomplete: {root}")
    if sidecar_path.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError(f"manifest checksum mismatch: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema:
        raise ValueError(f"unexpected schema at {root}: {manifest.get('schema')}")
    paths = [manifest_path, sidecar_path, *(root / name for name in material_names)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"immutable artifact lacks material files: {missing}")
    return manifest, {str(path): sha256(path) for path in paths}


def _stable_top(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    score = pd.to_numeric(frame["score_raw"], errors="raise").to_numpy(float)
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), -score))
    return frame.iloc[order[:count]].copy()


def _load_predictions(feature_source: Path, target_source: Path) -> pd.DataFrame:
    feature = pd.read_parquet(
        feature_source / "oof_feature_group_predictions.parquet"
    )
    target = pd.read_parquet(target_source / "oof_target_predictions.parquet")
    parts: list[pd.DataFrame] = []
    for head, (kind, arm, component) in HEADS.items():
        if kind == "feature":
            selected = feature.loc[
                feature["horizon_hours"].eq(12)
                & feature["feature_group"].eq(arm)
                & feature["target"].eq(component)
                & feature["target_valid"].astype(bool)
            ].copy()
        else:
            selected = target.loc[
                target["horizon_hours"].eq(12)
                & target["target_arm"].eq(arm)
                & target["target_valid"].astype(bool)
            ].copy()
        selected = selected.loc[
            :, [*COHORT_KEY, "fold_id", "delta_prediction", "sign_probability"]
        ]
        if selected.duplicated(list(COHORT_KEY)).any():
            raise ValueError(f"OOF head has duplicate cohort predictions: {head}")
        selected["head"] = head
        parts.append(selected)
    predictions = pd.concat(parts, ignore_index=True)
    predictions["cohort_anchor_utc"] = pd.to_datetime(
        predictions["cohort_anchor_utc"], utc=True, errors="raise"
    )
    return predictions


def _load_panel(panel_source: Path) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "candidate_month",
        "side_name",
        "__symbol__",
        "__ts__",
        "base_oof_score",
        "__first_touch_target_soft__",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "opportunity_gross_above_cost_0bps",
        "execution_label_end_utc",
    ]
    panel = pd.read_parquet(panel_source / "panel.parquet", columns=columns)
    panel = add_frozen_causal_score_deciles(panel)
    panel = panel.rename(
        columns={"__ts__": "cohort_anchor_utc", "base_oof_score": "score_raw"}
    )
    panel["candidate_month"] = panel["candidate_month"].astype(str)
    panel["candidate_positive_contribution"] = np.maximum(
        panel["execution_net_ev_12h"].to_numpy(float), 0.0
    )
    panel["candidate_loss_contribution"] = np.maximum(
        -panel["execution_net_ev_12h"].to_numpy(float), 0.0
    )
    return panel


def _monthly_tail_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (head, month), local in joined.groupby(
        ["head", "candidate_month"], observed=True, sort=True
    ):
        for fraction in FRACTIONS:
            selected = _stable_top(local, fraction)
            records.append(
                {
                    "head": head,
                    "candidate_month": month,
                    "fraction": fraction,
                    "candidate_rows": int(len(local)),
                    "selected_rows": int(len(selected)),
                    "prediction_coverage": float(
                        selected["delta_prediction"].notna().mean()
                    ),
                    "prediction_mean": float(selected["delta_prediction"].mean()),
                    "prediction_std": float(selected["delta_prediction"].std()),
                    "base_native_rank_ic": _corr(
                        selected["score_raw"],
                        selected["__first_touch_target_soft__"],
                    ),
                    "base_net_rank_ic": _corr(
                        selected["score_raw"], selected["execution_net_ev_12h"]
                    ),
                    "conversion_prediction_net_rank_ic": _corr(
                        selected["delta_prediction"],
                        selected["execution_net_ev_12h"],
                    ),
                    "conversion_prediction_opportunity_rank_ic": _corr(
                        selected["delta_prediction"],
                        selected["opportunity_gross_above_cost_0bps"],
                    ),
                    "conversion_prediction_upside_rank_ic": _corr(
                        selected["delta_prediction"],
                        selected["candidate_positive_contribution"],
                    ),
                    "conversion_prediction_loss_rank_ic": _corr(
                        selected["delta_prediction"],
                        selected["candidate_loss_contribution"],
                    ),
                    "mean_net_bps": float(
                        selected["execution_net_ev_12h"].mean() * 1e4
                    ),
                    "opportunity_rate": float(
                        selected["opportunity_gross_above_cost_0bps"].mean()
                    ),
                    "positive_contribution_bps": float(
                        selected["candidate_positive_contribution"].mean() * 1e4
                    ),
                    "loss_contribution_bps": float(
                        selected["candidate_loss_contribution"].mean() * 1e4
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _reference_edges(values: pd.Series, bins: int = 5) -> np.ndarray:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(finite) < bins:
        raise ValueError("insufficient reference predictions for fixed bins")
    interior = np.unique(np.quantile(finite, np.arange(1, bins) / bins))
    return np.concatenate(([-np.inf], interior, [np.inf]))


def _two_state_shapley(
    probability_a: np.ndarray,
    value_a: np.ndarray,
    probability_b: np.ndarray,
    value_b: np.ndarray,
) -> tuple[float, float]:
    composition = 0.5 * (
        np.dot(probability_b - probability_a, value_a)
        + np.dot(probability_b - probability_a, value_b)
    )
    conversion = 0.5 * (
        np.dot(probability_a, value_b - value_a)
        + np.dot(probability_b, value_b - value_a)
    )
    return float(composition), float(conversion)


def _fixed_bin_attribution(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells: list[dict[str, Any]] = []
    attribution: list[dict[str, Any]] = []
    for head, head_rows in joined.groupby("head", observed=True, sort=True):
        reference = _stable_top(
            head_rows.loc[head_rows["candidate_month"].eq(REFERENCE_MONTH)],
            0.10,
        )
        evaluation = _stable_top(
            head_rows.loc[head_rows["candidate_month"].eq(EVALUATION_MONTH)],
            0.10,
        )
        edges = _reference_edges(reference["delta_prediction"])
        month_frames: dict[str, pd.DataFrame] = {}
        for month, local in (
            (REFERENCE_MONTH, reference),
            (EVALUATION_MONTH, evaluation),
        ):
            work = local.copy()
            work["reference_prediction_bin"] = pd.cut(
                work["delta_prediction"],
                bins=edges,
                labels=False,
                include_lowest=True,
            )
            grouped = (
                work.groupby("reference_prediction_bin", observed=False)
                .agg(
                    rows=("candidate_id", "size"),
                    prediction_mean=("delta_prediction", "mean"),
                    net_mean=("execution_net_ev_12h", "mean"),
                    opportunity_rate=("opportunity_gross_above_cost_0bps", "mean"),
                    positive_contribution_mean=(
                        "candidate_positive_contribution",
                        "mean",
                    ),
                    loss_contribution_mean=("candidate_loss_contribution", "mean"),
                )
                .reindex(range(len(edges) - 1))
                .reset_index()
            )
            grouped["candidate_month"] = month
            grouped["head"] = head
            grouped["share"] = grouped["rows"].fillna(0.0) / len(work)
            cells.extend(grouped.to_dict(orient="records"))
            month_frames[month] = grouped
        a = month_frames[REFERENCE_MONTH]
        b = month_frames[EVALUATION_MONTH]
        if a["net_mean"].isna().any() or b["net_mean"].isna().any():
            attribution.append(
                {
                    "head": head,
                    "from_month": REFERENCE_MONTH,
                    "to_month": EVALUATION_MONTH,
                    "status": "UNRESOLVED_EMPTY_FIXED_BIN",
                }
            )
            continue
        composition, conversion = _two_state_shapley(
            a["share"].to_numpy(float),
            a["net_mean"].to_numpy(float) * 1e4,
            b["share"].to_numpy(float),
            b["net_mean"].to_numpy(float) * 1e4,
        )
        actual = float(
            b["share"].dot(b["net_mean"]) * 1e4
            - a["share"].dot(a["net_mean"]) * 1e4
        )
        attribution.append(
            {
                "head": head,
                "from_month": REFERENCE_MONTH,
                "to_month": EVALUATION_MONTH,
                "status": "COMPLETE",
                "actual_net_change_bps": actual,
                "predicted_state_composition_effect_bps": composition,
                "within_predicted_state_conversion_effect_bps": conversion,
                "reconciliation_error_bps": actual - composition - conversion,
            }
        )
    return pd.DataFrame.from_records(cells), pd.DataFrame.from_records(attribution)


def _high_low_daily_bootstrap(
    joined: pd.DataFrame, *, draws: int, random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    records: list[dict[str, Any]] = []
    for head, head_rows in joined.groupby("head", observed=True, sort=True):
        reference = _stable_top(
            head_rows.loc[head_rows["candidate_month"].eq(REFERENCE_MONTH)],
            0.10,
        )
        low, high = reference["delta_prediction"].quantile([0.2, 0.8])
        for month, month_rows in head_rows.groupby(
            "candidate_month", observed=True, sort=True
        ):
            selected = _stable_top(month_rows, 0.10).copy()
            selected["day"] = selected["cohort_anchor_utc"].dt.floor("D")
            selected["bucket"] = np.where(
                selected["delta_prediction"].ge(high),
                "high",
                np.where(selected["delta_prediction"].le(low), "low", "middle"),
            )
            daily = (
                selected.loc[selected["bucket"].isin(("high", "low"))]
                .groupby(["day", "bucket"], observed=True)
                .agg(net_sum=("execution_net_ev_12h", "sum"), rows=("candidate_id", "size"))
                .reset_index()
            )
            days = np.asarray(sorted(daily["day"].unique()))
            point = selected.groupby("bucket", observed=True)[
                "execution_net_ev_12h"
            ].mean()
            point_diff = float((point.get("high", np.nan) - point.get("low", np.nan)) * 1e4)
            bootstrap: list[float] = []
            if len(days) >= 2:
                for _ in range(int(draws)):
                    sampled = rng.choice(days, size=len(days), replace=True)
                    pieces = [daily.loc[daily["day"].eq(day)] for day in sampled]
                    draw = pd.concat(pieces, ignore_index=True).groupby(
                        "bucket", observed=True
                    ).agg(net_sum=("net_sum", "sum"), rows=("rows", "sum"))
                    if {"high", "low"}.issubset(draw.index):
                        bootstrap.append(
                            float(
                                (
                                    draw.loc["high", "net_sum"]
                                    / draw.loc["high", "rows"]
                                    - draw.loc["low", "net_sum"]
                                    / draw.loc["low", "rows"]
                                )
                                * 1e4
                            )
                        )
            records.append(
                {
                    "head": head,
                    "candidate_month": month,
                    "reference_low_threshold": float(low),
                    "reference_high_threshold": float(high),
                    "high_minus_low_net_bps": point_diff,
                    "day_blocks": int(len(days)),
                    "bootstrap_draws": int(len(bootstrap)),
                    "ci95_low_bps": float(np.quantile(bootstrap, 0.025))
                    if bootstrap
                    else np.nan,
                    "ci95_high_bps": float(np.quantile(bootstrap, 0.975))
                    if bootstrap
                    else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def plan(args: argparse.Namespace) -> dict[str, Any]:
    _, panel_hashes = _verify_artifact(
        Path(args.panel_source),
        "canonical_opportunity_payoff_trust_panel_v2",
        ("panel.parquet",),
    )
    _, feature_hashes = _verify_artifact(
        Path(args.feature_source),
        "canonical_economic_conversion_transition_feature_group_ablation_v1",
        ("oof_feature_group_predictions.parquet",),
    )
    _, target_hashes = _verify_artifact(
        Path(args.target_source),
        "canonical_economic_conversion_transition_target_ablation_v1",
        ("oof_target_predictions.parquet",),
    )
    return {
        "action": "PLAN_ONLY_NO_ATTRIBUTION",
        "schema": SCHEMA,
        "source_sha256": {**panel_hashes, **feature_hashes, **target_hashes},
        "heads": HEADS,
        "selection": "canonical one pooled-global monthly base top-1/5/10/20%; no reranking",
        "fixed_bins": f"{REFERENCE_MONTH} top10 prediction quintiles applied to {EVALUATION_MONTH}",
        "scope": "explanatory attribution only; no admission interaction, policy, or portfolio replay",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.plan_only:
        return plan(args)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    panel_manifest, panel_hashes = _verify_artifact(
        Path(args.panel_source),
        "canonical_opportunity_payoff_trust_panel_v2",
        ("panel.parquet",),
    )
    _, feature_hashes = _verify_artifact(
        Path(args.feature_source),
        "canonical_economic_conversion_transition_feature_group_ablation_v1",
        ("oof_feature_group_predictions.parquet",),
    )
    _, target_hashes = _verify_artifact(
        Path(args.target_source),
        "canonical_economic_conversion_transition_target_ablation_v1",
        ("oof_target_predictions.parquet",),
    )
    panel = _load_panel(Path(args.panel_source))
    predictions = _load_predictions(
        Path(args.feature_source), Path(args.target_source)
    )
    joined_parts: list[pd.DataFrame] = []
    for head, head_predictions in predictions.groupby(
        "head", observed=True, sort=True
    ):
        if head_predictions.duplicated(list(COHORT_KEY)).any():
            raise ValueError(f"prediction identity is not one-to-one for {head}")
        joined_parts.append(
            head_predictions.merge(
                panel,
                on=list(COHORT_KEY),
                how="left",
                validate="one_to_many",
            )
        )
    joined = pd.concat(joined_parts, ignore_index=True)
    if joined["candidate_id"].isna().any():
        raise ValueError("one or more OOF conversion cohorts lack candidate rows")
    joined = joined.loc[
        joined["candidate_month"].isin((REFERENCE_MONTH, EVALUATION_MONTH))
    ].copy()
    expected_heads = set(HEADS)
    if set(joined["head"].unique()) != expected_heads:
        raise ValueError("attribution head coverage changed")
    coverage = (
        joined.groupby(["head", "candidate_month"], observed=True)
        .agg(
            rows=("candidate_id", "size"),
            hours=("cohort_anchor_utc", "nunique"),
            candidates=("candidate_id", "nunique"),
        )
        .reset_index()
    )
    tails = _monthly_tail_metrics(joined)
    bins, attribution = _fixed_bin_attribution(joined)
    high_low = _high_low_daily_bootstrap(
        joined, draws=int(args.bootstrap_draws), random_state=int(args.random_state)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    frames = {
        "joined_head_coverage.parquet": coverage,
        "monthly_base_tail_attribution.parquet": tails,
        "fixed_reference_prediction_bins.parquet": bins,
        "march_april_fixed_bin_attribution.parquet": attribution,
        "high_low_daily_block_bootstrap.parquet": high_low,
    }
    for name, frame in frames.items():
        frame.to_parquet(temporary / name, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_EXPLANATORY_ATTRIBUTION_NOT_PROMOTION_ELIGIBLE",
        "promotion_eligible": False,
        "source_artifacts_sha256": {
            **panel_hashes,
            **feature_hashes,
            **target_hashes,
        },
        "source_panel_identity_sha256": panel_manifest.get("identity_sha256"),
        "heads": HEADS,
        "contracts": {
            "selection": "one pooled-global monthly base top-1/5/10/20% with candidate-ID tie-break; no timestamp/side quota",
            "join": "OOF conversion prediction joined by exact UTC hour, side and frozen causal base-score decile",
            "fixed_bins": "March base-top10 prediction quintiles are frozen before application to April",
            "uncertainty": "high-minus-low economics use deterministic UTC-day block bootstrap",
            "scope": "explanatory attribution only; no candidate reranking, admission interaction, policy, or portfolio replay",
        },
        "rows": {
            "joined_candidate_head_rows": int(len(joined)),
            "unique_candidates": int(joined["candidate_id"].nunique()),
        },
        "outputs_sha256": {
            name: sha256(temporary / name) for name in sorted(frames)
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "joined_candidate_head_rows": int(len(joined)),
        "heads": len(HEADS),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel-source", type=Path, default=PANEL_SOURCE)
    result.add_argument("--feature-source", type=Path, default=FEATURE_SOURCE)
    result.add_argument("--target-source", type=Path, default=TARGET_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--bootstrap-draws", type=int, default=2_000)
    result.add_argument("--random-state", type=int, default=20260729)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
