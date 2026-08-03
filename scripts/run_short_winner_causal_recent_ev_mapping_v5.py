#!/usr/bin/env python3
"""Seal the frozen short winner and its precommitted causal recent-EV map."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_bounded_robust_auxiliary_contribution_ablation as base
from scripts import run_bounded_short_conditional_payoff_ablation as short
from scripts.correct_bounded_side_local_support_composition_ties import bound
from scripts.run_bounded_side_local_support_composition import strict_mae

WINNER = ROOT / "data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2"
READINESS = ROOT / "data_perp/artifacts/short_recent_ev_mapping_readiness_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/short_winner_causal_recent_ev_mapping_20260730_v5"
MAE = short.MAE
POOL_MIN = 2_000
SIDE_MIN = 1_000
SIDE_LAMBDA = 500.0
WINDOW = pd.Timedelta(days=21)
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_DRAWS = 2_000
MAP_ARMS = (
    ("raw", "raw_score", None),
    ("frozen_march_isotonic", "frozen_march_isotonic", None),
    ("causal_pooled_21d", "causal_pooled_21d", "causal_pooled_21d_eligible"),
    ("causal_pooled_side_21d", "causal_pooled_side_21d", "causal_pooled_side_21d_eligible"),
)


class MappingContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def verify_sealed_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise MappingContractError(f"missing sealed manifest under {root}")
    declared = seal_path.read_text().split()[0]
    actual = sha256(manifest_path)
    if declared != actual:
        raise MappingContractError(f"manifest seal mismatch under {root}")
    return json.loads(manifest_path.read_text())


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def load_inputs(args: argparse.Namespace) -> pd.DataFrame:
    frame = base.load(args)
    mae, status = strict_mae(args.mae)
    if mae is None:
        raise MappingContractError(str(status))
    mae["__ts__"] = pd.to_datetime(mae["__ts__"], utc=True)
    joined = frame.merge(mae, on=list(base.ID), validate="one_to_one")
    if joined.duplicated(list(base.ID)).any():
        raise MappingContractError("input candidate keys are not unique")
    return joined


def _fold_assignment(frame: pd.DataFrame, cuts: Sequence[pd.Timestamp]) -> pd.Series:
    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for cut in cuts:
        mask = frame[base.TIME].ge(cut) & frame[base.TIME].lt(cut + pd.Timedelta(days=6))
        result.loc[mask] = cut
    return result


def build_march_score_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    development_all, _, base_cuts = base.reconstruct(frame)
    development = development_all.loc[
        np.isfinite(development_all.robust_decomposed)
    ].copy().reset_index(drop=True)
    development["peak_contribution"] = (
        development["pred_peak_mfe_12h_atr__p_hit"]
        * development["pred_peak_mfe_12h_atr__conditional_mean"]
    )
    features = list(short.F) + [
        "peak_contribution",
        "pred_future_slope_atr_per_hour__diagnostic",
    ]
    short_score = np.full(len(development), np.nan)
    short_fold = pd.Series(pd.NaT, index=development.index, dtype="datetime64[ns, UTC]")
    days = np.array(sorted(development[base.TIME].dt.floor("D").unique()))
    short_cuts = [days[int(len(days) * quantile)] for quantile in (0.4, 0.6, 0.8)]
    training_label_max: dict[tuple[str, pd.Timestamp, str], pd.Timestamp] = {}
    for cut in short_cuts:
        validation = development[base.TIME].ge(cut) & development[base.TIME].lt(
            cut + pd.Timedelta(days=6)
        )
        training = development[base.TIME].lt(cut) & development[base.END].lt(cut)
        train = development.loc[training & development.side_name.eq("short")]
        valid = development.loc[validation & development.side_name.eq("short")]
        training_label_max[("short", cut, "short")] = train[base.END].max()
        if len(valid):
            *_, score = short.fit_decomp(train, valid, features, 2.0)
            short_score[valid.index.to_numpy()] = score
            short_fold.loc[valid.index] = cut
    long_fold = _fold_assignment(development, base_cuts)
    for cut in base_cuts:
        training = development_all[base.TIME].lt(cut) & development_all[base.END].lt(cut)
        train = development_all.loc[training & development_all.side_name.eq("long")]
        training_label_max[("long", cut, "long")] = train[base.END].max()
    development["raw_score"] = np.where(
        development.side_name.eq("short"), short_score, development.robust_decomposed
    )
    development["validation_start_utc"] = np.where(
        development.side_name.eq("short"), short_fold, long_fold
    )
    development["validation_start_utc"] = pd.to_datetime(
        development["validation_start_utc"], utc=True
    )
    development = development.loc[
        np.isfinite(development.raw_score) & development.validation_start_utc.notna()
    ].copy()
    development["validation_end_utc"] = development.validation_start_utc + pd.Timedelta(days=6)
    development["fold_train_cutoff_utc"] = development.validation_start_utc
    development["training_label_resolved_max_utc"] = [
        training_label_max[(side, cut, side)]
        for cut, side in zip(development.validation_start_utc, development.side_name)
    ]
    development["score_available_utc"] = development[base.TIME]
    development["candidate_score_is_oof"] = True
    development["upstream_scores_are_outer_oof"] = True
    development["candidate_score_is_forward_oos"] = False
    development["candidate_score_head"] = np.where(
        development.side_name.eq("short"),
        "short_B_peak_slope_tail2_inner_chronological_oof",
        "long_frozen_robust_decomposed_chronological_oof",
    )
    development["candidate_score_config"] = "B_peak_slope__tail_2"
    development["ledger_stage"] = "march_inner_chronological_oof"
    validate_score_ledger(development, expect_oof=True)
    if len(development) != 33_408:
        raise MappingContractError(
            f"frozen March OOF row parity failed: {len(development)} != 33408"
        )
    return development


def build_april_score_ledger(
    frame: pd.DataFrame, winner_root: Path, march: pd.DataFrame
) -> pd.DataFrame:
    frozen = pd.read_parquet(winner_root / "april_confirmation_predictions.parquet")
    for column in ("__ts__", base.TIME, base.END):
        frozen[column] = pd.to_datetime(frozen[column], utc=True)
    if frozen.duplicated(list(base.ID)).any():
        raise MappingContractError("frozen April score keys are not unique")
    economics = frame.loc[
        frame.candidate_month.eq("2025-04"),
        list(base.ID)
        + [
            base.Y,
            "execution_gross_ev_12h",
            "execution_cost_return",
            "score_base_alpha",
            "score_residual_expected_ev",
            "direct_q25_return",
        ],
    ].copy()
    economics["__ts__"] = pd.to_datetime(economics["__ts__"], utc=True)
    keep = list(base.ID)
    missing = [column for column in economics.columns if column not in frozen.columns or column in keep]
    frozen = frozen.merge(economics[missing], on=keep, validate="one_to_one")
    if len(frozen) != len(economics):
        raise MappingContractError("April frozen/economic candidate population mismatch")
    april_start = frozen[base.TIME].min()
    training_label_max = frame.loc[
        frame.candidate_month.eq("2025-03") & frame[base.END].lt(april_start),
        base.END,
    ].max()
    if not training_label_max < april_start:
        raise MappingContractError("April model training labels are not resolved before evaluation")
    frozen["score_available_utc"] = frozen[base.TIME]
    frozen["validation_start_utc"] = april_start
    frozen["validation_end_utc"] = frozen[base.TIME].max() + pd.Timedelta(hours=1)
    frozen["fold_train_cutoff_utc"] = april_start
    frozen["training_label_resolved_max_utc"] = training_label_max
    frozen["candidate_score_is_oof"] = False
    frozen["upstream_scores_are_outer_oof"] = True
    frozen["candidate_score_is_forward_oos"] = True
    frozen["candidate_score_head"] = np.where(
        frozen.side_name.eq("short"),
        "short_B_peak_slope_tail2_frozen_forward",
        "long_frozen_robust_decomposed_forward",
    )
    frozen["candidate_score_config"] = "B_peak_slope__tail_2"
    frozen["ledger_stage"] = "april_frozen_forward"
    validate_score_ledger(frozen, expect_oof=False)
    return frozen


def validate_score_ledger(frame: pd.DataFrame, *, expect_oof: bool) -> None:
    required = {
        "candidate_id",
        "side_name",
        "__symbol__",
        "__ts__",
        base.TIME,
        base.END,
        "raw_score",
        "score_available_utc",
        "fold_train_cutoff_utc",
        "training_label_resolved_max_utc",
        "validation_start_utc",
        "validation_end_utc",
        "candidate_score_is_oof",
        "candidate_score_is_forward_oos",
        "upstream_scores_are_outer_oof",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise MappingContractError(f"score ledger missing columns: {sorted(missing)}")
    if frame.duplicated(list(base.ID)).any() or not np.isfinite(frame.raw_score).all():
        raise MappingContractError("score ledger keys/scores fail")
    for column in (
        "__ts__",
        base.TIME,
        base.END,
        "score_available_utc",
        "fold_train_cutoff_utc",
        "training_label_resolved_max_utc",
        "validation_start_utc",
        "validation_end_utc",
    ):
        values = pd.to_datetime(frame[column], utc=True, errors="raise")
        if values.isna().any():
            raise MappingContractError(f"score ledger has missing {column}")
    if not frame.score_available_utc.le(frame[base.TIME]).all():
        raise MappingContractError("candidate score is not available at decision")
    if not frame.training_label_resolved_max_utc.lt(frame.validation_start_utc).all():
        raise MappingContractError("training label cutoff overlaps validation")
    if expect_oof:
        if not frame.candidate_score_is_oof.astype(bool).all():
            raise MappingContractError("March candidate scores are not all OOF")
    elif not frame.candidate_score_is_forward_oos.astype(bool).all():
        raise MappingContractError("April candidate scores are not all forward OOS")
    if not frame.upstream_scores_are_outer_oof.astype(bool).all():
        raise MappingContractError("upstream score provenance is not outer OOF")


def causal_map(
    history: pd.DataFrame, evaluate: pd.DataFrame, *, add_side_residual: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = evaluate.copy().reset_index(drop=True)
    prefix = "causal_pooled_side_21d" if add_side_residual else "causal_pooled_21d"
    result[prefix] = np.nan
    result[f"{prefix}_eligible"] = False
    result[f"{prefix}_status"] = "unmapped_weak_pooled"
    result[f"{prefix}_pooled_rows"] = 0
    result[f"{prefix}_side_rows"] = 0
    result[f"{prefix}_side_weight"] = 0.0
    result[f"{prefix}_snapshot_utc"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    audits: list[dict[str, Any]] = []
    for snapshot, indices in result.groupby(result[base.TIME].dt.floor("D"), sort=True).groups.items():
        snapshot = pd.Timestamp(snapshot)
        positions = np.asarray(list(indices), dtype=int)
        batch = result.loc[positions]
        reference = history.loc[
            history[base.END].ge(snapshot - WINDOW)
            & history[base.END].lt(snapshot)
            & history.score_available_utc.lt(snapshot)
            & np.isfinite(history.raw_score)
            & np.isfinite(history[base.Y])
        ].copy()
        overlap = len(set(batch.candidate_id.astype(str)).intersection(reference.candidate_id.astype(str)))
        legal = (
            overlap == 0
            and (reference[base.END].lt(snapshot).all() if len(reference) else True)
            and (
                reference[base.END].ge(snapshot - WINDOW).all()
                if len(reference)
                else True
            )
            and (
                reference.score_available_utc.lt(snapshot).all()
                if len(reference)
                else True
            )
        )
        if not legal:
            raise MappingContractError(f"illegal mapping reference at {snapshot}")
        pooled_ready = len(reference) >= POOL_MIN and reference.raw_score.nunique() >= 2
        status = "unmapped_weak_pooled"
        long_rows = int(reference.side_name.eq("long").sum())
        short_rows = int(reference.side_name.eq("short").sum())
        long_weight = 0.0
        short_weight = 0.0
        result.loc[positions, f"{prefix}_snapshot_utc"] = snapshot
        result.loc[positions, f"{prefix}_pooled_rows"] = len(reference)
        if pooled_ready:
            pooled_model = IsotonicRegression(out_of_bounds="clip").fit(
                reference.raw_score, reference[base.Y]
            )
            raw = batch.raw_score.to_numpy(float)
            mapped = pooled_model.predict(raw)
            status_values = np.full(len(batch), "pooled_anchor", dtype=object)
            if add_side_residual:
                for side_name in ("long", "short"):
                    batch_mask = batch.side_name.eq(side_name).to_numpy()
                    side_reference = reference.loc[reference.side_name.eq(side_name)]
                    side_rows = len(side_reference)
                    result.loc[
                        positions[batch_mask], f"{prefix}_side_rows"
                    ] = side_rows
                    if (
                        side_rows >= SIDE_MIN
                        and side_reference.raw_score.nunique() >= 2
                        and batch_mask.any()
                    ):
                        side_model = IsotonicRegression(out_of_bounds="clip").fit(
                            side_reference.raw_score, side_reference[base.Y]
                        )
                        side_raw = raw[batch_mask]
                        weight = float(side_rows / (side_rows + SIDE_LAMBDA))
                        mapped[batch_mask] += weight * (
                            side_model.predict(side_raw)
                            - pooled_model.predict(side_raw)
                        )
                        status_values[batch_mask] = "pooled_plus_shrunk_side_residual"
                        result.loc[
                            positions[batch_mask], f"{prefix}_side_weight"
                        ] = weight
                        if side_name == "long":
                            long_weight = weight
                        else:
                            short_weight = weight
                    else:
                        status_values[batch_mask] = "pooled_zero_side_residual"
            result.loc[positions, prefix] = mapped
            result.loc[positions, f"{prefix}_eligible"] = True
            result.loc[positions, f"{prefix}_status"] = status_values
            status = "mapped"
        audits.append(
            {
                "map_arm": prefix,
                "snapshot_utc": snapshot,
                "evaluation_rows": len(batch),
                "reference_rows": len(reference),
                "long_reference_rows": long_rows,
                "short_reference_rows": short_rows,
                "reference_label_end_min_utc": reference[base.END].min()
                if len(reference)
                else pd.NaT,
                "reference_label_end_max_utc": reference[base.END].max()
                if len(reference)
                else pd.NaT,
                "reference_score_available_max_utc": reference.score_available_utc.max()
                if len(reference)
                else pd.NaT,
                "evaluation_reference_identity_overlap": overlap,
                "strict_causal_window_pass": legal,
                "pooled_support_pass": pooled_ready,
                "long_side_support_pass": long_rows >= SIDE_MIN,
                "short_side_support_pass": short_rows >= SIDE_MIN,
                "long_side_weight": long_weight,
                "short_side_weight": short_weight,
                "status": status,
            }
        )
    if result.loc[result[f"{prefix}_eligible"], prefix].isna().any():
        raise MappingContractError("eligible mapping rows contain NaN")
    if result.loc[~result[f"{prefix}_eligible"], prefix].notna().any():
        raise MappingContractError("weak pooled rows received a tradable mapped score")
    return result, pd.DataFrame(audits)


def stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return base.order(frame, score, fraction)


def calibration_ece(prediction: np.ndarray, outcome: np.ndarray) -> float:
    return float(base.ece(prediction, outcome) * 1e4)


def evaluate_arm(
    frame: pd.DataFrame, *, arm: str, score: str, eligible: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    local = frame.loc[frame[eligible].astype(bool)].copy() if eligible else frame.copy()
    if local.empty:
        return [], [], [], []
    latest_start = local[base.TIME].max().floor("D") - pd.Timedelta(days=6)
    metrics: list[dict[str, Any]] = []
    sides: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for fraction in FRACTIONS:
        selected = stable_top(local, score, fraction)
        tie = bound(local, score, fraction)
        prediction = selected[score].to_numpy(float)
        outcome = selected[base.Y].to_numpy(float)
        latest = stable_top(local.loc[local[base.TIME].ge(latest_start)], score, fraction)
        metrics.append(
            {
                "map_arm": arm,
                "top_fraction": fraction,
                "candidate_rows": len(local),
                "coverage_fraction": len(local) / len(frame),
                "selected_rows": len(selected),
                "net_bps": float(outcome.mean() * 1e4),
                "gross_bps": float(selected.execution_gross_ev_12h.mean() * 1e4),
                "cost_bps": float(selected.execution_cost_return.mean() * 1e4),
                "positive_rate": float((outcome > 0).mean()),
                "full_rank_ic": float(local[score].corr(local[base.Y], method="spearman")),
                "prediction_bias_bps": float((prediction - outcome).mean() * 1e4),
                "prediction_mae_bps": float(np.abs(prediction - outcome).mean() * 1e4),
                "calibration_ece_bps": calibration_ece(prediction, outcome),
                "latest_week_start_utc": latest_start,
                "latest_week_rows": len(latest),
                "latest_week_net_bps": float(latest[base.Y].mean() * 1e4),
                "cutoff": tie["cutoff"],
                "cutoff_tie_rows": tie["cutoff_tie_rows"],
                "cutoff_tie_fraction_of_book": tie["cutoff_tie_fraction_of_book"],
                "random_tie_expected_net_bps": tie["random_tie_expected_net_bps"],
                "random_tie_expected_precision": tie["random_tie_expected_precision"],
                "best_tie_precision": tie["best_tie_precision"],
                "worst_tie_precision": tie["worst_tie_precision"],
            }
        )
        if math.isclose(fraction, 0.10):
            for side_name, part in selected.groupby("side_name", sort=True):
                sides.append(
                    {
                        "map_arm": arm,
                        "side_name": side_name,
                        "rows": len(part),
                        "share": len(part) / len(selected),
                        "net_bps": float(part[base.Y].mean() * 1e4),
                        "positive_rate": float(part[base.Y].gt(0).mean()),
                    }
                )
            for symbol, part in selected.groupby("__symbol__", sort=True):
                assets.append(
                    {
                        "map_arm": arm,
                        "__symbol__": symbol,
                        "rows": len(part),
                        "share": len(part) / len(selected),
                        "net_bps": float(part[base.Y].mean() * 1e4),
                    }
                )
        daily = selected.assign(day=selected[base.TIME].dt.floor("D")).groupby("day")[
            base.Y
        ].mean()
        if len(daily):
            draws = rng.choice(daily.to_numpy(float), size=(BOOTSTRAP_DRAWS, len(daily)), replace=True)
            means = draws.mean(axis=1) * 1e4
            intervals.append(
                {
                    "map_arm": arm,
                    "top_fraction": fraction,
                    "utc_day_blocks": len(daily),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "equal_day_mean_net_bps": float(daily.mean() * 1e4),
                    "ci_lower_bps": float(np.quantile(means, 0.025)),
                    "ci_upper_bps": float(np.quantile(means, 0.975)),
                }
            )
    return metrics, sides, assets, intervals


def control_metrics(april: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in ("score_base_alpha", "score_residual_expected_ev", "direct_q25_return"):
        for fraction in FRACTIONS:
            selected = stable_top(april, score, fraction)
            rows.append(
                {
                    "control": score,
                    "top_fraction": fraction,
                    "selected_rows": len(selected),
                    "net_bps": float(selected[base.Y].mean() * 1e4),
                }
            )
    return pd.DataFrame(rows)


def promotion_gates(
    metrics: pd.DataFrame,
    sides: pd.DataFrame,
    assets: pd.DataFrame,
    controls: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    arm = "causal_pooled_side_21d"
    top10 = metrics.loc[
        metrics.map_arm.eq(arm) & metrics.top_fraction.eq(0.10)
    ].iloc[0]
    side = sides.loc[sides.map_arm.eq(arm)]
    asset = assets.loc[assets.map_arm.eq(arm)]
    control_top10 = controls.loc[controls.top_fraction.eq(0.10)]
    rows = [
        ("all snapshots causally legal", bool(audit.strict_causal_window_pass.all()), True),
        ("all snapshots mapped", bool(audit.pooled_support_pass.all()), True),
        ("mapped coverage 100%", float(top10.coverage_fraction), 1.0),
        ("top10 random-tie expected net positive", float(top10.random_tie_expected_net_bps), ">0"),
        ("latest-week top10 net positive", float(top10.latest_week_net_bps), ">0"),
        ("cutoff tie fraction <=5%", float(top10.cutoff_tie_fraction_of_book), "<=0.05"),
        ("largest side share <=75%", float(side.share.max()), "<=0.75"),
        ("both sides positive", bool((side.net_bps > 0).all() and len(side) == 2), True),
        ("largest asset share <=10%", float(asset.share.max()), "<=0.10"),
        ("absolute top10 bias <=25bps", abs(float(top10.prediction_bias_bps)), "<=25"),
        ("top10 ECE <=25bps", float(top10.calibration_ece_bps), "<=25"),
        (
            "beats every identical-ID control at top10",
            float(top10.random_tie_expected_net_bps),
            f">{float(control_top10.net_bps.max())}",
        ),
    ]
    gate_rows = []
    for gate, value, threshold in rows:
        if threshold is True:
            passed = bool(value)
        elif threshold == ">0":
            passed = float(value) > 0
        elif threshold == "<=0.05":
            passed = float(value) <= 0.05
        elif threshold == "<=0.75":
            passed = float(value) <= 0.75
        elif threshold == "<=0.10":
            passed = float(value) <= 0.10
        elif threshold == "<=25":
            passed = float(value) <= 25
        elif threshold == 1.0:
            passed = math.isclose(float(value), 1.0)
        elif isinstance(threshold, str) and threshold.startswith(">"):
            passed = float(value) > float(threshold[1:])
        else:
            raise AssertionError(threshold)
        gate_rows.append(
            {"gate": gate, "pass": passed, "value": value, "threshold": threshold}
        )
    table = pd.DataFrame(gate_rows)
    return table, bool(table["pass"].all())


def invalidate_incomplete(root: Path, replacement: Path) -> None:
    if not root.is_dir():
        return
    invalidation = root / "INVALIDATION.json"
    if invalidation.exists():
        return
    write_json(
        invalidation,
        {
            "status": "INVALIDATED_INCOMPLETE_NONAUTHORITATIVE",
            "reason": (
                "Missing final daily causal audit, promotion gates, complete score-lineage "
                "proof, source-seal verification and focused regression coverage."
            ),
            "replacement": str(replacement),
            "all_metrics_non_authoritative": True,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    winner_manifest = verify_sealed_manifest(args.winner)
    readiness_manifest = verify_sealed_manifest(args.readiness)
    if winner_manifest.get("frozen_winner", {}).get("arm") != "B_peak_slope":
        raise MappingContractError("wrong frozen winner arm")
    if float(winner_manifest["frozen_winner"]["short_tail_weight"]) != 2.0:
        raise MappingContractError("wrong frozen winner tail weight")
    prediction_path = args.winner / "april_confirmation_predictions.parquet"
    if winner_manifest["outputs_sha256"]["april_confirmation_predictions.parquet"] != sha256(
        prediction_path
    ):
        raise MappingContractError("frozen April prediction hash mismatch")
    if readiness_manifest.get("admissible_grid", [])[1] != {
        "minimum_reference_rows": POOL_MIN,
        "minimum_short_rows": SIDE_MIN,
        "name": "standard",
        "short_shrinkage": SIDE_LAMBDA,
    }:
        raise MappingContractError("standard readiness contract changed")
    frame = load_inputs(args)
    march = build_march_score_ledger(frame)
    april = build_april_score_ledger(frame, args.winner, march)
    frozen_mapper = IsotonicRegression(out_of_bounds="clip").fit(
        march.raw_score, march[base.Y]
    )
    april["frozen_march_isotonic"] = frozen_mapper.predict(april.raw_score)
    history = pd.concat([march, april], ignore_index=True, sort=False)
    pooled, pooled_audit = causal_map(history, april, add_side_residual=False)
    mapped, side_audit = causal_map(history, april, add_side_residual=True)
    map_columns = [
        column
        for column in mapped.columns
        if column.startswith("causal_pooled_side_21d")
    ]
    april = pooled.merge(
        mapped[list(base.ID) + map_columns], on=list(base.ID), validate="one_to_one"
    )
    audit = pd.concat([pooled_audit, side_audit], ignore_index=True)
    if not audit.strict_causal_window_pass.all():
        raise MappingContractError("daily causal audit failed")
    all_metrics: list[dict[str, Any]] = []
    all_sides: list[dict[str, Any]] = []
    all_assets: list[dict[str, Any]] = []
    all_intervals: list[dict[str, Any]] = []
    for arm, score, eligible in MAP_ARMS:
        metrics, sides, assets, intervals = evaluate_arm(
            april, arm=arm, score=score, eligible=eligible
        )
        all_metrics.extend(metrics)
        all_sides.extend(sides)
        all_assets.extend(assets)
        all_intervals.extend(intervals)
    metrics_table = pd.DataFrame(all_metrics)
    sides_table = pd.DataFrame(all_sides)
    assets_table = pd.DataFrame(all_assets)
    intervals_table = pd.DataFrame(all_intervals)
    controls_table = control_metrics(april)
    gates_table, promotion_eligible = promotion_gates(
        metrics_table, sides_table, assets_table, controls_table, audit
    )
    if promotion_eligible:
        raise MappingContractError(
            "promotion gates unexpectedly pass; portfolio replay requires a separate frozen authorization"
        )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        outputs: dict[str, pd.DataFrame] = {
            "march_inner_chronological_oof_score_ledger.parquet": march,
            "april_frozen_forward_score_ledger_and_maps.parquet": april,
            "daily_mapping_audit.parquet": audit,
            "global_metrics.csv": metrics_table,
            "side_top10.csv": sides_table,
            "asset_top10.csv": assets_table,
            "identical_id_controls.csv": controls_table,
            "utc_day_block_intervals.csv": intervals_table,
            "promotion_gates.csv": gates_table,
        }
        for name, table in outputs.items():
            path = stage / name
            if name.endswith(".parquet"):
                table.to_parquet(path, index=False, compression="zstd")
            else:
                table.to_csv(path, index=False)
        output_hashes = {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}
        input_paths = {
            "winner_manifest": args.winner / "manifest.json",
            "winner_predictions": prediction_path,
            "readiness_manifest": args.readiness / "manifest.json",
            "source": args.source,
            "peak": args.peak,
            "slope": args.slope,
            "mae": args.mae / "oof_predictions.parquet",
        }
        manifest = {
            "schema": "short_winner_causal_recent_ev_mapping_v5",
            "run_id": args.output_dir.name,
            "status": "SEALED_RESEARCH_FAILURE_NO_PORTFOLIO_REPLAY",
            "promotion_eligible": False,
            "portfolio_replay": "NOT_RUN",
            "experiment_status": (
                "March candidate-head chronological OOF plus April frozen-forward causal "
                "policy-map diagnostic; April was previously inspected and is not an untouched final test."
            ),
            "contract": {
                "ranker": "frozen B_peak_slope__tail_2; no April refit",
                "selection": "one pooled-global top-k after mapping; no timestamp, side or asset quota",
                "mapping": (
                    "daily UTC pooled isotonic on prior resolved 21d labels plus optional "
                    "side isotonic-minus-pooled residual shrunk by n_side/(n_side+500)"
                ),
                "support": {"pooled_min": POOL_MIN, "side_min": SIDE_MIN},
                "weak_pooled": "NaN/unmapped warm-up, excluded from mapped selection",
                "weak_side": "pooled anchor retained with exactly zero side residual",
                "label": "exact side-relative deployed-exit 12h net = gross - one explicit cost",
                "cost_units": "decimal return in ledgers; report tables convert to bps",
                "actions": "timing, MAE, target-price and wait actions excluded",
                "portfolio": "not replayed because promotion gates fail",
            },
            "dates_utc": {
                "candidate_head_development": [
                    str(march[base.TIME].min()),
                    str(march[base.TIME].max()),
                ],
                "forward_evaluation": [
                    str(april[base.TIME].min()),
                    str(april[base.TIME].max()),
                ],
                "mapping_reference_window_days": 21,
            },
            "validation": {
                "candidate_head": "three inner chronological March OOF blocks",
                "upstream_scores": "strict outer OOF/frozen scores",
                "purge": "training rows require label_end < validation_start",
                "embargo": "12h outcome resolution enforced by label_end cutoff; no additional gap",
                "calibration_priors": "daily causal 21d, frozen at each UTC-day snapshot",
                "thresholds": "predeclared global top 1/5/10/20%; no April threshold fitting",
                "hpo": "none in mapping; standard readiness grid member precommitted without economics",
            },
            "model": {
                "features": list(short.F)
                + ["peak_contribution", "pred_future_slope_atr_per_hour__diagnostic"],
                "random_seeds": {"classifier": 19, "gain_regressor": 23, "loss_regressor": 29},
                "short_tail_weight": 2.0,
                "mapping_model": "sklearn IsotonicRegression(out_of_bounds=clip)",
            },
            "data": {
                "universe": "canonical exact-ID March-April 2025 candidate population from source artifact",
                "bar_frequency": "hourly candidate decisions; exact 1m execution paths",
                "march_rows": len(march),
                "april_rows": len(april),
                "april_columns": len(april.columns),
            },
            "gates": {
                row.gate: bool(row["pass"]) for _, row in gates_table.iterrows()
            },
            "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
            "outputs_sha256": output_hashes,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
                "git_revision": git_revision(),
            },
            "bootstrap": {
                "method": "fixed selected book, equal UTC-day block resampling",
                "seed": BOOTSTRAP_SEED,
                "draws": BOOTSTRAP_DRAWS,
            },
            "limitations": [
                "No bankroll/portfolio PnL because policy replay is blocked.",
                "No signed residual autocorrelation or hit-rate-surprise policy feature is introduced.",
                "April is a frozen diagnostic reused after prior inspection, not a new final test.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            sha256(stage / "manifest.json") + "  manifest.json\n"
        )
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    for incomplete in args.invalidate:
        invalidate_incomplete(incomplete, args.output_dir)
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source", type=Path, default=base.SRC)
    command.add_argument("--peak", type=Path, default=base.PEAK)
    command.add_argument("--slope", type=Path, default=base.SLOPE)
    command.add_argument("--mae", type=Path, default=MAE)
    command.add_argument("--winner", type=Path, default=WINNER)
    command.add_argument("--readiness", type=Path, default=READINESS)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument(
        "--invalidate",
        type=Path,
        nargs="*",
        default=[
            ROOT / "data_perp/artifacts/short_winner_causal_recent_ev_mapping_20260730_v2",
            ROOT / "data_perp/artifacts/short_winner_causal_recent_ev_mapping_20260730_v3",
            ROOT / "data_perp/artifacts/short_winner_causal_recent_ev_mapping_20260730_v4",
        ],
    )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
