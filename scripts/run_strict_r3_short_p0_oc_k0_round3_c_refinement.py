#!/usr/bin/env python3
"""Round 3B: C3-specific feature and conditional-weight refinement.

The upstream opportunity component is frozen to the Round-2 O250/H6 winner.
This script therefore changes only the conditional conversion learner C3:

    P0 -> O250/H6 [frozen] -> C3 [features/weights] -> K0 [same analytic form]

The work is deliberately sequential.  First, target-specific chronological MDA
is fitted only on true O-positive training rows to choose C30/C45/C60/C90
feature prefixes against the C41 control.  Only the winning feature contract
then proceeds to the four predeclared conditional-C weighting schemes.  O's
seed is fixed across all arms, so a C result cannot be attributed to a hidden
change in the opportunity learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_round3_c_refinement_v1"
ROUND3A = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_targets_20260821_v1"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_refinement_20260822_v1"
MDA_START = pd.Timestamp("2024-05-01T00:00:00Z")
MDA_END = pd.Timestamp("2025-01-01T00:00:00Z")
TARGET = next(item for item in r3.TARGETS if item.name == "C3_normalized_regret")
O_SEED = r3.SEED + 3 * 10_000  # Exact Round-3A C3 upstream O stream.
# Features and C weights are the only permitted differences in this round.
# Reuse the original C3 target stream seed so C41 is an exact paired control.
C_SEED = O_SEED
SEED = 1729
FEATURE_CAPS = (30, 45, 60, 90)
WEIGHTS = ("uniform", "equal_month", "equal_mfe", "equal_month_mfe")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _c_objective(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    """Target-specific C utility: harvestability rank plus conditional net tail.

    The rank term preserves C3's normalized-regret semantics; the bounded tail
    term rejects a feature that simply predicts a clean path while losing its
    economic conversion relationship.  Both are conditional on O=true rows.
    """
    score = np.asarray(prediction, dtype=float)
    y = r3._target(frame, TARGET).astype(float)
    rank_ic = pd.Series(score).corr(pd.Series(y), method="spearman")
    if not np.isfinite(rank_ic):
        return float("nan")
    rank = pd.Series(score).rank(method="first", pct=True).to_numpy(float)
    net = r1._finite(frame["policy_net_bps"]).to_numpy(float)
    top = net[rank >= .80]
    uplift = (float(np.nanmean(top)) - float(np.nanmean(net))) / 500.0 if len(top) else 0.0
    return float(rank_ic + .5 * np.clip(uplift, -2.0, 2.0))


def _strict_c_mda(frame: pd.DataFrame, fields: tuple[str, ...], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = frame.loc[
        r1._valid_label(frame)
        & r1._event(frame, r3.SPEC).astype(bool)
        & frame["__decision_ts__"].ge(MDA_START)
        & frame["__decision_ts__"].lt(MDA_END)
    ].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(local) < r3.MIN_C_ROWS:
        raise RuntimeError("insufficient conditional C3 development support for MDA")
    boundaries = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    deltas: dict[str, list[float]] = {field: [] for field in fields}
    records: list[dict[str, Any]] = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(boundaries[fold + 1]):int(boundaries[fold + 2])].copy()
        if valid.empty:
            continue
        decision_start = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(decision_start)].copy()
        y_fit = r3._target(fit, TARGET)
        if len(fit) < r3.MIN_C_ROWS or np.unique(y_fit).size < 2:
            continue
        x_fit, medians = r1._matrix(fit, fields)
        x_valid, _ = r1._matrix(valid, fields, medians)
        model = r3._model(TARGET, seed + fold)
        model.fit(x_fit, y_fit, sample_weight=r3._c_weights(fit, "uniform"))
        base = _c_objective(valid, r3._predict(model, TARGET, x_valid))
        rng = np.random.default_rng(seed + 1_000 + fold)
        for field in fields:
            permuted = x_valid.copy()
            permuted[field] = rng.permutation(permuted[field].to_numpy())
            delta = base - _c_objective(valid, r3._predict(model, TARGET, permuted))
            deltas[field].append(float(delta))
            records.append({"feature": field, "fold": fold, "validation_start": decision_start, "mda_delta": float(delta), "base_objective": float(base)})
    if not records:
        raise RuntimeError("no valid chronological C3 MDA folds")
    ranking = pd.DataFrame({
        "feature": list(fields),
        "mda_mean": [float(np.nanmean(deltas[field])) if deltas[field] else float("nan") for field in fields],
        "mda_min": [float(np.nanmin(deltas[field])) if deltas[field] else float("nan") for field in fields],
        "mda_positive_folds": [int(np.sum(np.asarray(deltas[field]) > 0.0)) for field in fields],
        "mda_folds": [len(deltas[field]) for field in fields],
    })
    ranking = ranking.sort_values(["mda_mean", "mda_min", "feature"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking, pd.DataFrame(records)


def _stable_prefixes(ranking: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    folds = int(ranking["mda_folds"].max())
    required = int(np.ceil(.60 * folds))
    stable = ranking.loc[ranking["mda_positive_folds"].ge(required) & ranking["mda_mean"].gt(0.0)].copy()
    remainder = ranking.loc[~ranking["feature"].isin(stable["feature"])].copy()
    ordered = pd.concat((stable, remainder), ignore_index=True)["feature"].astype(str).tolist()
    return {cap: tuple(ordered[:cap]) for cap in FEATURE_CAPS}


def _metrics(prediction: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    monthly = pd.DataFrame([
        r1._k0_metrics(part, r3.SPEC, pd.Timestamp(f"{month}-01", tz="UTC"))
        for month, part in prediction.groupby("held_month", sort=True)
    ])
    monthly["arm"] = arm
    era = r1._aggregate_k0(monthly)
    era["arm"] = arm
    use = era.loc[era["era"].isin(("2025", "2026"))].set_index("era")
    selected = float(use["outcome_known_candidates"].sum())
    target_months = monthly.loc[monthly["held_month"].str[:4].isin(("2025", "2026"))]
    return monthly, era, {
        "arm": arm,
        "net_2025": float(use.loc["2025", "net_bps_per_trade"]),
        "net_2026": float(use.loc["2026", "net_bps_per_trade"]),
        "mean_net_bps_per_trade": float(np.average(use["net_bps_per_trade"], weights=use["outcome_known_candidates"])),
        "total_net_bps": float(use["total_net_bps"].sum()),
        "selected": selected,
        "worst_month": float(target_months["net_bps_per_trade"].min()),
        "mean_cvar10": float(use["cvar10_bps"].mean()),
    }


def _rank(summary: pd.DataFrame, reference: float) -> pd.DataFrame:
    out = summary.copy()
    out["participation_vs_c41"] = out["selected"] / max(reference, 1.0)
    out["passes_gate"] = out["net_2025"].ge(90.0) & out["net_2026"].ge(90.0) & out["participation_vs_c41"].ge(.70)
    return out.sort_values(["passes_gate", "mean_net_bps_per_trade", "worst_month", "total_net_bps"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _assert_fixed_o(prediction: pd.DataFrame, control: pd.DataFrame) -> None:
    left = prediction.loc[:, ["candidate_id", "opportunity_raw_score", "opportunity_probability"]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    right = control.loc[:, ["candidate_id", "opportunity_raw_score", "opportunity_probability"]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not left["candidate_id"].equals(right["candidate_id"]):
        raise AssertionError("C arm changed target-free O candidate identities")
    for field in ("opportunity_raw_score", "opportunity_probability"):
        if not np.isclose(left[field].to_numpy(float), right[field].to_numpy(float), rtol=0.0, atol=2e-6, equal_nan=True).all():
            raise AssertionError(f"C arm unexpectedly changed frozen O output: {field}")


def _assert_c3_control_parity(prediction: pd.DataFrame) -> None:
    """C41/uniform must reproduce the completed Round-3A C3 control exactly."""
    path = ROUND3A / "C3_normalized_regret_outer_oof_predictions.parquet"
    reference = pd.read_parquet(path, columns=["candidate_id", "conversion_score", "K0_expected_policy_net_bps"])
    left = prediction.loc[:, ["candidate_id", "conversion_score", "K0_expected_policy_net_bps"]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    right = reference.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not left["candidate_id"].equals(right["candidate_id"]):
        raise AssertionError("C41 does not reproduce Round-3A target-free candidate identities")
    for field in ("conversion_score", "K0_expected_policy_net_bps"):
        if not np.isclose(left[field].to_numpy(float), right[field].to_numpy(float), rtol=0.0, atol=2e-6, equal_nan=True).all():
            raise AssertionError(f"C41 fails Round-3A exact paired-control parity: {field}")


def _table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def _report(out: Path, *, feature_rank: pd.DataFrame, weight_rank: pd.DataFrame, mda: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# Short P0 → O250/H6 → C3 → K0 Round 3B: C-specific refinement", "",
        "Research-only. O250/H6 is identical across every arm; this round changes only C3 features and C3 training weights.", "",
        "## Feature stage", "", _table(feature_rank), "",
        "## Weight stage", "", _table(weight_rank), "",
        "## C3 chronological MDA", "", _table(mda), "",
        "## Contract", "", "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_ROUND3B_C_REFINEMENT_REPORT.md").write_text("\n".join(lines))


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    frame, o_fields, c41, source_hashes = r3._load_frame()
    f115 = r1._load_f115_selection(r1.DEFAULT_FEATURE_SELECTION)
    missing = sorted(set(f115).difference(frame.columns))
    if missing:
        raise AssertionError(f"C3 MDA pool fields unavailable: {missing}")
    mda, mda_folds = _strict_c_mda(frame, f115, SEED)
    prefixes = _stable_prefixes(mda)
    feature_contracts: dict[str, tuple[str, ...]] = {"C41_frozen": tuple(c41)}
    feature_contracts.update({f"C{cap}_mda": fields for cap, fields in prefixes.items()})
    feature_rows: list[dict[str, Any]] = []
    feature_monthly: list[pd.DataFrame] = []
    feature_era: list[pd.DataFrame] = []
    feature_predictions: dict[str, pd.DataFrame] = {}
    control: pd.DataFrame | None = None
    for arm, fields in feature_contracts.items():
        prediction, _ = r3._run_target(frame, o_fields, fields, TARGET, C_SEED, "uniform", o_seed=O_SEED)
        if control is None:
            control = prediction
            _assert_c3_control_parity(prediction)
        else:
            _assert_fixed_o(prediction, control)
        monthly, era, summary = _metrics(prediction, arm)
        summary["feature_count"] = len(fields)
        feature_rows.append(summary); feature_monthly.append(monthly); feature_era.append(era); feature_predictions[arm] = prediction
    feature_summary = pd.DataFrame(feature_rows)
    reference = float(feature_summary.loc[feature_summary["arm"].eq("C41_frozen"), "selected"].iloc[0])
    feature_rank = _rank(feature_summary, reference)
    winner = str(feature_rank.loc[feature_rank["passes_gate"], "arm"].iloc[0])
    weight_rows: list[dict[str, Any]] = []
    weight_monthly: list[pd.DataFrame] = []
    weight_era: list[pd.DataFrame] = []
    weight_predictions: dict[str, pd.DataFrame] = {}
    for index, weight in enumerate(WEIGHTS):
        arm = f"{winner}__{weight}"
        if weight == "uniform":
            prediction = feature_predictions[winner]
        else:
            prediction, _ = r3._run_target(frame, o_fields, feature_contracts[winner], TARGET, C_SEED, weight, o_seed=O_SEED)
            _assert_fixed_o(prediction, control)
        monthly, era, summary = _metrics(prediction, arm)
        summary.update({"feature_arm": winner, "weight": weight, "feature_count": len(feature_contracts[winner])})
        weight_rows.append(summary); weight_monthly.append(monthly); weight_era.append(era); weight_predictions[arm] = prediction
    weight_summary = pd.DataFrame(weight_rows)
    uniform_ref = float(weight_summary.loc[weight_summary["weight"].eq("uniform"), "selected"].iloc[0])
    weight_rank = _rank(weight_summary, uniform_ref)
    final_arm = str(weight_rank.loc[weight_rank["passes_gate"], "arm"].iloc[0])
    out.mkdir(parents=True)
    mda.to_parquet(out / "round3b_c3_stability_mda.parquet", index=False, compression="zstd")
    mda_folds.to_parquet(out / "round3b_c3_stability_mda_folds.parquet", index=False, compression="zstd")
    feature_rank.to_parquet(out / "round3b_feature_ranking.parquet", index=False, compression="zstd")
    weight_rank.to_parquet(out / "round3b_weight_ranking.parquet", index=False, compression="zstd")
    pd.concat(feature_monthly, ignore_index=True).to_parquet(out / "round3b_feature_monthly.parquet", index=False, compression="zstd")
    pd.concat(feature_era, ignore_index=True).to_parquet(out / "round3b_feature_era.parquet", index=False, compression="zstd")
    pd.concat(weight_monthly, ignore_index=True).to_parquet(out / "round3b_weight_monthly.parquet", index=False, compression="zstd")
    pd.concat(weight_era, ignore_index=True).to_parquet(out / "round3b_weight_era.parquet", index=False, compression="zstd")
    for arm, prediction in feature_predictions.items():
        prediction.to_parquet(out / f"{arm}_outer_oof_predictions.parquet", index=False, compression="zstd")
    for arm, prediction in weight_predictions.items():
        prediction.to_parquet(out / f"{arm}_outer_oof_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short",
        "scope": "Round 3B sequential C3 feature then C3 conditional-weight refinement; research-only, no canonical/live mutation",
        "architecture": "frozen P0 → O250_H6 → C3_normalized_regret → K0",
        "opportunity": {"definition": "mfe_6h_bps > 250", "features": list(o_fields), "weights": "uniform", "calibration": "Platt", "o_seed": O_SEED, "c_seed": C_SEED, "cross_arm_invariant": "all C arms have identical target-free O candidate identities/raw scores/probabilities", "c41_control": "C41/uniform exactly reproduces Round-3A C3 conversion and K0 scores"},
        "conversion": {"target": TARGET.description, "mda_pool": list(f115), "mda_window": [MDA_START.isoformat(), MDA_END.isoformat()], "mda_population": "valid true O-positive rows", "mda_fit": "label_available_at < chronological validation start", "mda_objective": "Spearman(C3 prediction, C3 target) + 0.5 * bounded conditional top-20% policy-net uplift", "stability": "positive MDA in >=60% of chronological folds then ranked prefix", "feature_contracts": {name: list(fields) for name, fields in feature_contracts.items()}, "weights": list(WEIGHTS)},
        "selection": {"gate": "net EV/trade >=90 bps in both 2025 and 2026; participation >=70% of stage control", "feature_winner": winner, "weight_winner": final_arm, "tie_break": "mean net EV/trade, then worst month, then total net bps"},
        "causality": {"outer": "label_available_at < held month start", "inner": "label_available_at < inner validation start", "targetfree_scoring": "all held candidates scored before invalid labels are excluded", "forbidden": ["held outcome features", "held percentile admission", "MC1/trust/consensus/risk layer"]},
        "sources": {"round3a_manifest_sha256": _sha256(ROUND3A / "run_manifest.json"), **source_hashes},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, feature_rank=feature_rank, weight_rank=weight_rank, mda=mda, manifest=manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
