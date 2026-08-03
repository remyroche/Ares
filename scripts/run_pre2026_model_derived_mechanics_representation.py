#!/usr/bin/env python3
"""Cross-fitted, model-derived trade-mechanics representations.

This is deliberately a *representation* experiment.  It learns three
probabilities from causal, candidate-time inputs and emits their OOF values as
sidecars.  It does not add an opaque feature or a hard gate to the residual
stack.  The two data vintages currently share only 15 candidate-level
mechanics primitives, so results cover Apr--Dec 2023 and Mar--Apr 2025 only.

The outer split is leave-calendar-month-out OOF, not walk-forward OOS: future
months may be in an outer-training partition.  The artifact is therefore
non-promotion research evidence and must not be replayed or applied to 2026.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp" / "artifacts"
OUT = ART / "pre2026_model_derived_mechanics_representation_20260730_v1"
ANCHOR = ART / "frozen_contextual_score_arms_2023apr_2025jun_20260730_v1" / "blocked_oof_training_panel.parquet"
EARLY = ART / "failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1" / "context.parquet"
LATE = ART / "cross_era_tail_payoff_dataset_20260730_v3" / "cross_era_tail_payoff_dataset.parquet"
LATE_SCORES = ART / "canonical_execution_reliability_input_20260730_v4" / "panel.parquet"

MECHANICS = [
    "broad_washout_recovery", "btc_decoupling_dispersion", "compressed_index_fragmented_assets",
    "correlation_breakdown_dispersion", "correlation_heterogeneity_dispersion",
    "deleveraged_range_climax_reversal", "deleveraging_without_followthrough",
    "flush_recovery_state", "fragile_leverage_rebuild", "fragmented_flush_recovery",
    "fragmented_new_low_breadth", "negative_breadth_pct", "peer_volatility_decoupling",
    "post_flush_leverage_rebuild", "thin_compression",
]
CORE = ["score_base_alpha", "score_residual_expected_ev", "residual_minus_base"]
HEADS = {
    "p_mechanics_opportunity_25bps": "target_opportunity_25bps",
    "p_conversion_net_positive": "target_conversion_net_positive",
    "p_downside_severe_loss_100bps": "target_severe_loss_100bps",
}
ARMS = {"core_only": CORE, "core_plus_model_discovered_mechanics": CORE + MECHANICS}
MAX_TRAIN_ROWS_PER_SIDE = 80_000
RANDOM_SEED = 20260730


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    temporary = path.with_name("." + path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _coerce_time(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("__ts__", "execution_label_end_utc", "execution_label_available_at"):
        if column in out:
            out[column] = pd.to_datetime(out[column], utc=True)
    return out


def _anchor_frame(anchor: Path) -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "__symbol__", "side_name", "execution_label_end_utc",
        "execution_label_available_at", "execution_net_ev_12h", "execution_gross_ev_12h",
        "execution_cost_return", "score_base_alpha", "score_residual_expected_ev",
    ]
    frame = _coerce_time(pd.read_parquet(anchor, columns=columns))
    if frame.candidate_id.duplicated().any():
        raise ValueError("anchor candidate_id must be unique")
    if frame.__ts__.dt.minute.ne(0).any() or frame.__ts__.dt.second.ne(0).any():
        raise ValueError("anchor is not a 1h decision panel")
    frame["execution_label_end_utc"] = frame.execution_label_end_utc.combine_first(frame.execution_label_available_at)
    required = ["execution_label_end_utc", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return"]
    if frame[required].isna().any().any():
        raise ValueError("anchor has missing economics or score inputs")
    frame["residual_minus_base"] = frame.score_residual_expected_ev - frame.score_base_alpha
    return frame


def _join_vintage(anchor: pd.DataFrame, source: Path, vintage: str) -> pd.DataFrame:
    columns = ["candidate_id", "__ts__", "__symbol__", "side_name", *MECHANICS]
    source_frame = _coerce_time(pd.read_parquet(source, columns=columns))
    if source_frame.candidate_id.duplicated().any():
        raise ValueError(f"{vintage} candidate_id must be unique")
    joined = anchor.merge(source_frame, on="candidate_id", how="inner", suffixes=("", "__source"), validate="one_to_one")
    for field in ("__ts__", "__symbol__", "side_name"):
        if not joined[field].eq(joined[f"{field}__source"]).all():
            raise ValueError(f"{vintage} candidate identity mismatch in {field}")
    joined = joined.drop(columns=["__ts____source", "__symbol____source", "side_name__source"])
    if joined[MECHANICS].isna().any().any() or not np.isfinite(joined[MECHANICS].to_numpy(dtype=float)).all():
        raise ValueError(f"{vintage} common mechanics schema is not fully finite")
    joined["vintage"] = vintage
    return joined


def _overlay_late_scores(frame: pd.DataFrame, score_source: Path) -> pd.DataFrame:
    """Use the canonical 2025 OOF score lineage where the broad anchor has no scores."""
    columns = [
        "candidate_id", "__ts__", "__symbol__", "side_name", "execution_net_ev_12h",
        "score_base_alpha", "score_residual_expected_ev",
    ]
    score = _coerce_time(pd.read_parquet(score_source, columns=columns))
    score = score.dropna(subset=["score_base_alpha", "score_residual_expected_ev"])
    if score.candidate_id.duplicated().any():
        raise ValueError("canonical late score overlay candidate_id must be unique")
    out = frame.merge(score, on="candidate_id", how="inner", suffixes=("", "__overlay"), validate="one_to_one")
    for field in ("__ts__", "__symbol__", "side_name"):
        if not out[field].eq(out[f"{field}__overlay"]).all():
            raise ValueError(f"canonical late score overlay identity mismatch in {field}")
    if not np.allclose(out.execution_net_ev_12h, out.execution_net_ev_12h__overlay, atol=1e-12, rtol=0.0, equal_nan=False):
        raise ValueError("canonical late score overlay execution-net mismatch")
    out["score_base_alpha"] = out.score_base_alpha__overlay
    out["score_residual_expected_ev"] = out.score_residual_expected_ev__overlay
    out["residual_minus_base"] = out.score_residual_expected_ev - out.score_base_alpha
    return out.drop(columns=["__ts____overlay", "__symbol____overlay", "side_name__overlay", "execution_net_ev_12h__overlay", "score_base_alpha__overlay", "score_residual_expected_ev__overlay"])


def materialize_union(anchor: Path = ANCHOR, early: Path = EARLY, late: Path = LATE, late_scores: Path = LATE_SCORES) -> pd.DataFrame:
    """Exact schema-intersection union used by this first mechanics study."""
    base = _anchor_frame(Path(anchor))
    early_frame = _join_vintage(base, Path(early), "early_2023").dropna(subset=CORE)
    late_frame = _overlay_late_scores(_join_vintage(base, Path(late), "late_2025"), Path(late_scores))
    frame = pd.concat([early_frame, late_frame], ignore_index=True)
    if frame.candidate_id.duplicated().any():
        raise ValueError("vintage joins overlap candidate identities")
    if frame.execution_label_end_utc.ge(pd.Timestamp("2026-01-01", tz="UTC")).any():
        raise ValueError("2026 outcomes are forbidden")
    frame["outer_month"] = frame.__ts__.dt.strftime("%Y-%m")
    valid = {"2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2025-03", "2025-04"}
    if set(frame.outer_month.unique()) != valid:
        raise ValueError(f"unexpected outer-month coverage: {sorted(frame.outer_month.unique())}")
    frame["target_opportunity_25bps"] = (
        frame.execution_gross_ev_12h.gt(frame.execution_cost_return + 0.0025)
    ).astype(np.int8)
    frame["target_conversion_net_positive"] = frame.execution_net_ev_12h.gt(0.0).astype(np.int8)
    frame["target_severe_loss_100bps"] = frame.execution_net_ev_12h.le(-0.01).astype(np.int8)
    return frame


def _subsample_train(frame: pd.DataFrame, maximum: int = MAX_TRAIN_ROWS_PER_SIDE) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    # Candidate IDs make this stable under input ordering and avoid sampling on outcomes.
    hashes = pd.util.hash_pandas_object(frame.candidate_id, index=False).to_numpy(np.uint64)
    return frame.iloc[np.argsort(hashes, kind="stable")[:maximum]].copy()


def _select_mechanics(train: pd.DataFrame, target: str, side: str) -> list[str]:
    """Fold-local, target-local discovery; CORE is always forced into the refit."""
    use = _subsample_train(train)
    model = CatBoostClassifier(
        loss_function="Logloss", iterations=120, depth=4, learning_rate=0.05,
        l2_leaf_reg=12.0, random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False,
        thread_count=4,
    ).fit(use[CORE + MECHANICS], use[target])
    values = model.get_feature_importance(type="PredictionValuesChange")
    ranked = sorted(zip(MECHANICS, values[len(CORE):]), key=lambda item: (-item[1], item[0]))
    # A compact, data-selected representation; no manually authored product is emitted.
    chosen = [name for name, importance in ranked if np.isfinite(importance) and importance > 0][:8]
    if not chosen:
        raise ValueError(f"no positive fold-local mechanics importance for {target}/{side}")
    return chosen


def _fit_probability(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, seed_offset: int) -> np.ndarray:
    use = _subsample_train(train)
    if use[target].nunique() != 2:
        raise ValueError(f"single-class training data for {target}")
    model = CatBoostClassifier(
        loss_function="Logloss", iterations=160, depth=4, learning_rate=0.05,
        l2_leaf_reg=12.0, random_seed=RANDOM_SEED + seed_offset, verbose=False,
        allow_writing_files=False, thread_count=4,
    ).fit(use[features], use[target])
    return model.predict_proba(test[features])[:, 1]


def expected_calibration_error(actual: pd.Series, prediction: pd.Series, bins: int = 10) -> float:
    y = actual.to_numpy(dtype=float); p = np.clip(prediction.to_numpy(dtype=float), 0.0, 1.0)
    if not len(y):
        return float("nan")
    index = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.mean(index == bucket) * abs(y[index == bucket].mean() - p[index == bucket].mean()) for bucket in range(bins) if np.any(index == bucket)))


def head_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (month, side, arm, head), group in predictions.groupby(["outer_month", "side_name", "arm", "head"], sort=True):
        y, p = group.actual_target, group.prediction
        rows.append({
            "outer_month": month, "side_name": side, "arm": arm, "head": head, "rows": len(group),
            "prevalence": y.mean(),
            "roc_auc": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
            "average_precision": average_precision_score(y, p) if y.nunique() == 2 else np.nan,
            "brier": brier_score_loss(y, p), "ece10": expected_calibration_error(y, p), "bias": float(p.mean() - y.mean()),
        })
    return pd.DataFrame(rows)


def _rank01(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def score_and_economics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = frame.copy()
    wide = joined.pivot(index="candidate_id", columns=["arm", "head"], values="prediction")
    wide.columns = [f"{arm}__{head}" for arm, head in wide.columns]
    meta = joined.drop_duplicates("candidate_id").set_index("candidate_id")[["outer_month", "side_name", "__ts__", "__symbol__", "execution_net_ev_12h", "score_residual_expected_ev"]]
    x = meta.join(wide, how="inner")
    rows: list[pd.DataFrame] = []
    for month, group in x.groupby("outer_month", sort=True):
        out = group.copy()
        residual = _rank01(out.score_residual_expected_ev)
        out["residual_control"] = residual
        for arm in ARMS:
            # Predeclared equal rank corrections.  This is not a fitted policy or a hard admission gate.
            out[f"{arm}__representation_score"] = residual + 0.25 * _rank01(out[f"{arm}__p_mechanics_opportunity_25bps"]) + 0.25 * _rank01(out[f"{arm}__p_conversion_net_positive"]) - 0.25 * _rank01(out[f"{arm}__p_downside_severe_loss_100bps"])
        rows.append(out.assign(outer_month=month))
    score = pd.concat(rows).reset_index()
    economics: list[dict[str, object]] = []
    for month, group in score.groupby("outer_month", sort=True):
        n = math.ceil(len(group) * 0.10)
        for arm, column in [("residual_control", "residual_control"), *[(arm, f"{arm}__representation_score") for arm in ARMS]]:
            chosen = group.sort_values([column, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            economics.append({
                "outer_month": month, "arm": arm, "candidate_rows": len(group), "selected_rows": len(chosen),
                "global_top10_net_ev": chosen.execution_net_ev_12h.mean(),
                "global_top10_positive_net_rate": chosen.execution_net_ev_12h.gt(0).mean(),
                "global_top10_severe_loss_rate": chosen.execution_net_ev_12h.le(-0.01).mean(),
                "long_net_ev": chosen.loc[chosen.side_name.eq("long"), "execution_net_ev_12h"].mean(),
                "short_net_ev": chosen.loc[chosen.side_name.eq("short"), "execution_net_ev_12h"].mean(),
            })
    return score, pd.DataFrame(economics)


def run(output: Path = OUT, *, anchor: Path = ANCHOR, early: Path = EARLY, late: Path = LATE, late_scores: Path = LATE_SCORES) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    frame = materialize_union(anchor, early, late, late_scores)
    predictions: list[pd.DataFrame] = []
    discovery: list[dict[str, object]] = []
    fold_audit: list[dict[str, object]] = []
    for month, held in frame.groupby("outer_month", sort=True):
        train_all = frame[frame.outer_month.ne(month) & frame.execution_label_end_utc.lt(held.__ts__.min()) == False].copy()
        # Leave-month-out is intentionally symmetric OOF.  Label end is still recorded and all rows resolve before 2026.
        train_all = frame[frame.outer_month.ne(month)].copy()
        for side, test in held.groupby("side_name", sort=True):
            train = train_all[train_all.side_name.eq(side)].copy()
            if len(train) < 1_000 or len(test) < 100:
                raise ValueError(f"insufficient side support: {month}/{side}")
            for head_no, (head, target) in enumerate(HEADS.items()):
                if train[target].nunique() != 2 or test[target].nunique() != 2:
                    raise ValueError(f"event support failure: {month}/{side}/{head}")
                selected = _select_mechanics(train, target, side)
                for rank, feature in enumerate(selected, start=1):
                    discovery.append({"outer_month": month, "side_name": side, "head": head, "rank": rank, "feature": feature})
                for arm, fields in ARMS.items():
                    features = CORE if arm == "core_only" else CORE + selected
                    prediction = _fit_probability(train, test, features, target, head_no + (0 if arm == "core_only" else 100))
                    predictions.append(test[["candidate_id", "outer_month", "__ts__", "__symbol__", "side_name", "execution_net_ev_12h", "score_residual_expected_ev", target]].rename(columns={target: "actual_target"}).assign(arm=arm, head=head, prediction=prediction))
                    fold_audit.append({"outer_month": month, "side_name": side, "head": head, "arm": arm, "train_rows": len(train), "test_rows": len(test), "train_label_end_max": train.execution_label_end_utc.max(), "test_start": test.__ts__.min(), "feature_count": len(features), "features": "|".join(features), "outer_protocol": "leave_month_out_oof_not_forward_oos"})
    oof = pd.concat(predictions, ignore_index=True)
    metrics = head_metrics(oof)
    scored, economics = score_and_economics(oof)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        oof.to_parquet(stage / "oof_mechanics_head_predictions.parquet", index=False)
        scored.to_parquet(stage / "oof_representation_scores.parquet", index=False)
        metrics.to_csv(stage / "head_metrics_by_month_side.csv", index=False)
        pd.DataFrame(discovery).to_csv(stage / "fold_local_mechanics_discovery.csv", index=False)
        pd.DataFrame(fold_audit).to_csv(stage / "fold_audit.csv", index=False)
        economics.to_csv(stage / "pooled_global_top10_economics_by_month.csv", index=False)
        contract = {
            "schema": "pre2026_model_derived_mechanics_representation_v1",
            "status": "CROSS_FITTED_OOF_DEVELOPMENT_ONLY_NON_PROMOTION",
            "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
            "outer_protocol": "side-local leave-calendar-month-out OOF; explicitly not walk-forward OOS",
            "coverage": "2023-04..2023-12 and 2025-03-12..2025-04 only; no compatible candidate mechanics panel for 2024 or 2025-05..06",
            "inputs": {"core": CORE, "common_causal_mechanics_schema": MECHANICS},
            "heads": HEADS,
            "feature_discovery": "per held month/side/head CatBoost PredictionValuesChange on outer training rows; top eight positive mechanics fields; CORE forced; fixed learner, no HPO",
            "prohibited": ["targets", "outcomes", "exit/action fields", "GMM/DAE IDs/posteriors", "era/source/provenance", "2026 rows"],
            "score": "residual rank + .25 opportunity rank + .25 conversion rank - .25 severe-downside rank; fixed diagnostic only, no fitted mapping/policy",
            "selection": "one pooled global top10 per month across timestamps and sides; monthly slices decompose that fixed book",
            "source_sha256": {str(Path(anchor)): sha(Path(anchor)), str(Path(early)): sha(Path(early)), str(Path(late)): sha(Path(late)), str(Path(late_scores)): sha(Path(late_scores))},
            "implementation_sha256": sha(Path(__file__)),
        }
        dump(stage / "contract.json", contract)
        manifest = {"schema": contract["schema"], "contract": contract, "counts": {"candidate_rows": len(frame), "oof_prediction_rows": len(oof), "outer_months": sorted(frame.outer_month.unique().tolist()), "all_hour_aligned": bool(frame.__ts__.dt.minute.eq(0).all()), "all_label_end_before_2026": bool(frame.execution_label_end_utc.lt(pd.Timestamp("2026-01-01", tz="UTC")).all())}}
        dump(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha(stage / "manifest.json") + "  manifest.json\n")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


if __name__ == "__main__":
    print(run())
