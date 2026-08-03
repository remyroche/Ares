#!/usr/bin/env python3
"""Run raw-feature, long-split exact-H12 base/residual target ablations.

This is deliberately a *research* refit.  The historical panel contains only
the candidate population produced by an older policy and current-spread
counterfactual labels; it is not the full base universe and cannot support a
promotion.  Within that limitation it is a genuine raw-feature base refit:

* 12 months base training (2023-04..2024-03), then 8 months frozen base OOS;
* first four base-OOS months train the residual; final four are untouched; and
* all selection uses one pooled global book across sides and timestamps after
  a pooled, causal 21-day score-to-net map.

Every target is stated in *net of row cost* terms and has a positive economic
hurdle.  Base targets vary the opportunity handoff; residual targets vary the
amount of emphasis placed on the globally top-ranked policy tail.  No target
is constructed per timestamp or per side rank.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_long_base_residual_target_ablation import (
    BASE_OOS_END,
    BASE_TRAIN_END,
    BASE_TRAIN_START,
    META_TRAIN_END,
    SIDES,
    _book_metrics,
    _causal_recent_map,
    _fold_ids,
    calendar_masks,
    global_top_mask,
)


PANEL_DIR = ROOT / "data_perp/artifacts/long_exact_h12_raw_base_panel_20260730_v2"
PANEL = PANEL_DIR / "raw_base_panel.parquet"
FEATURE_CONTRACT = PANEL_DIR / "raw_feature_contract.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/long_raw_base_residual_h12_ablation_20260730_v1"
TOP_FEATURES = 64
HURDLE = 0.0025  # Extra positive buffer after the row's already-subtracted cost.


@dataclass(frozen=True)
class Arm:
    name: str
    role: str


BASE_ARMS = (
    Arm("net_hurdle_soft", "base"),
    Arm("risk_penalised_net", "base"),
    Arm("timely_clean_net", "base"),
)
META_ARMS = (
    Arm("net_residual", "meta"),
    Arm("global_tail_weighted_residual", "meta"),
    Arm("policy_soft_clear", "meta"),
    Arm("clean_tail_weighted_residual", "meta"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=float), -40.0, 40.0)))


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    return np.nan_to_num(pd.to_numeric(frame[name], errors="coerce").to_numpy(float), nan=default, posinf=default, neginf=default)


def base_target(frame: pd.DataFrame, arm: str) -> np.ndarray:
    """Cost-clearing soft opportunity targets for the raw base layer."""

    net = _numeric(frame, "execution_net_ev_12h")
    adverse = _numeric(frame, "__adverse_competing_risk_12h__")
    mae = np.clip(_numeric(frame, "__mae_before_meaningful_mfe_atr_12h__"), 0.0, 10.0)
    opportunity = _numeric(frame, "__opportunity_occurred_12h__")
    time_hours = np.clip(_numeric(frame, "__time_to_first_meaningful_mfe_hours_12h__", 12.0), 0.0, 12.0)
    net_soft = _sigmoid((net - HURDLE) / 0.010)
    if arm == "net_hurdle_soft":
        return net_soft
    if arm == "risk_penalised_net":
        # Economic net remains the core.  Competing adverse movement and large
        # pre-MFE drawdown only make the post-cost hurdle stricter.
        adjusted = net - HURDLE - 0.0040 * adverse - 0.0010 * np.tanh(mae / 3.0)
        return _sigmoid(adjusted / 0.010)
    if arm == "timely_clean_net":
        # This cannot promote a net loser: the clean/fast path only modulates
        # the already cost-clearing economic label.
        clean_fast = opportunity * (1.0 - adverse) * (1.0 - time_hours / 12.0)
        return net_soft * (0.50 + 0.50 * clean_fast)
    raise ValueError(f"unknown base arm: {arm}")


def meta_target(frame: pd.DataFrame, arm: str) -> np.ndarray:
    """Residual targets.  All labels retain a net-cost hurdle."""

    net = _numeric(frame, "execution_net_ev_12h")
    base = _numeric(frame, "base_expected_net")
    if arm in {"net_residual", "global_tail_weighted_residual", "clean_tail_weighted_residual"}:
        return net - base
    if arm == "policy_soft_clear":
        return _sigmoid((net - HURDLE) / 0.0075)
    raise ValueError(f"unknown meta arm: {arm}")


def _matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"missing raw features: {missing}")
    return frame.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _fit(matrix: pd.DataFrame, target: np.ndarray, weights: np.ndarray, *, seed: int, small: bool = False) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=140 if small else 260,
        learning_rate=0.045 if small else 0.035,
        num_leaves=23,
        max_depth=5,
        min_child_samples=180 if small else 220,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        reg_lambda=15.0,
        reg_alpha=0.15,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(matrix, target, sample_weight=weights)
    return model


def _select_features(frame: pd.DataFrame, features: list[str], target: np.ndarray, *, seed: int) -> list[str]:
    """Per-side, base-train-only gain screen; no meta/evaluation labels enter."""

    matrix = _matrix(frame, features)
    finite = [name for name in features if matrix[name].notna().mean() >= 0.50 and matrix[name].nunique(dropna=True) > 1]
    probe = _fit(_matrix(frame, finite), target, np.ones(len(frame)), seed=seed, small=True)
    gains = pd.Series(probe.booster_.feature_importance(importance_type="gain"), index=finite)
    chosen = gains.sort_values(ascending=False, kind="stable").head(TOP_FEATURES)
    selected = chosen.index[chosen.gt(0.0)].tolist()
    if len(selected) < 16:
        raise ValueError("base feature screen retained too few non-zero-gain fields")
    return selected


def _base_oof_calibrator(frame: pd.DataFrame, features: list[str], target: np.ndarray, *, seed: int) -> IsotonicRegression:
    """OOF target-score -> actual exact net conversion on base training only."""

    matrix = _matrix(frame, features)
    folds = _fold_ids(frame)
    oof = np.full(len(frame), np.nan)
    for fold in np.unique(folds):
        train, valid = folds != fold, folds == fold
        model = _fit(matrix.loc[train], target[train], np.ones(int(train.sum())), seed=seed + int(fold))
        oof[valid] = model.predict(matrix.loc[valid])
        del model
        gc.collect()
    mapper = IsotonicRegression(out_of_bounds="clip")
    mapper.fit(oof, _numeric(frame, "execution_net_ev_12h"))
    return mapper


def _global_tail_weights(frame: pd.DataFrame, *, clean: bool) -> np.ndarray:
    """Policy emphasis is pooled global rank, never a timestamp-local rank."""

    selected = global_top_mask(_numeric(frame, "base_expected_net"), 0.10)
    weights = np.where(selected, 4.0, 1.0)
    if clean:
        opportunity = _numeric(frame, "__opportunity_occurred_12h__")
        adverse = _numeric(frame, "__adverse_competing_risk_12h__")
        weights *= 1.0 + selected * 1.5 * opportunity * (1.0 - adverse)
    return weights.astype(float)


def _meta_oof_scores(
    frame: pd.DataFrame,
    features: list[str],
    arm: str,
    *,
    sample_weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fold-local OOF scores used only as causal map history."""

    matrix = _matrix(frame, features)
    folds = _fold_ids(frame)
    oof = np.full(len(frame), np.nan)
    target = meta_target(frame, arm)
    weights = np.asarray(sample_weights, dtype=float)
    if len(weights) != len(frame) or not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("precomputed residual sample weights are invalid")
    for fold in np.unique(folds):
        train, valid = folds != fold, folds == fold
        model = _fit(matrix.loc[train], target[train], weights[train], seed=seed + int(fold))
        raw = model.predict(matrix.loc[valid])
        # Residual arms return in net-return units; probability arm is mapped
        # directly from its soft cost-clearing score below.
        oof[valid] = raw if arm == "policy_soft_clear" else _numeric(frame.iloc[np.flatnonzero(valid)], "base_expected_net") + raw
        del model
        gc.collect()
    return oof


def _metrics_with_side_decomposition(scored: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    records = _book_metrics(scored, "mapped_expected_net", arm=arm)
    valid = scored.loc[np.isfinite(scored["mapped_expected_net"])].copy().reset_index(drop=True)
    for fraction in (0.01, 0.05, 0.10, 0.20):
        selected = global_top_mask(valid["mapped_expected_net"], fraction)
        book = valid.loc[selected]
        for side, part in book.groupby("side_name", sort=True):
            records.append({
                "arm": arm, "scope": "global_book_membership_by_side", "side_name": side,
                "fraction": fraction, "eligible_rows": int(len(valid)), "selected_rows": int(len(part)),
                "mean_net_bps": float(part.execution_net_ev_12h.mean() * 10_000.0),
                "positive_net_rate": float(part.execution_net_ev_12h.gt(0).mean()),
                "long_share": float(part.side_name.eq("long").mean()),
                "rank_ic": float(valid.mapped_expected_net.corr(valid.execution_net_ev_12h, method="spearman")),
            })
    return records


def _attach_pooled_tail_weights(working: pd.DataFrame, meta_train_mask: np.ndarray) -> pd.DataFrame:
    """Attach one immutable cross-side policy tail before any side split.

    This is deliberately done once over the *complete* residual-training
    cohort.  A side-specific residual then receives its rows' precomputed
    weight; it must never derive a new within-side tail membership.
    """

    if len(working) != len(meta_train_mask):
        raise ValueError("meta training mask length differs from working frame")
    result = working.copy()
    result["pooled_tail_member"] = False
    result["pooled_tail_weight"] = 1.0
    train_positions = np.flatnonzero(meta_train_mask)
    train = result.iloc[train_positions]
    member = global_top_mask(_numeric(train, "base_expected_net"), 0.10)
    # ``global_top_mask`` is deterministic and selects ceil(10%) exactly.
    if int(member.sum()) != max(1, int(np.ceil(len(train) * 0.10))):
        raise AssertionError("pooled global tail selected an unexpected count")
    result.loc[result.index[train_positions], "pooled_tail_member"] = member
    result.loc[result.index[train_positions], "pooled_tail_weight"] = np.where(member, 4.0, 1.0)
    opportunity = _numeric(train, "__opportunity_occurred_12h__")
    adverse = _numeric(train, "__adverse_competing_risk_12h__")
    result.loc[result.index[train_positions], "pooled_clean_tail_weight"] = np.where(
        member,
        4.0 * (1.0 + 1.5 * opportunity * (1.0 - adverse)),
        1.0,
    )
    result["pooled_clean_tail_weight"] = result["pooled_clean_tail_weight"].fillna(1.0)
    # Recombining side slices must reproduce the original immutable membership.
    joined = pd.concat([result.loc[result.side_name.eq(side), ["candidate_id", "pooled_tail_member"]] for side in SIDES], ignore_index=True)
    expected = result.loc[:, ["candidate_id", "pooled_tail_member"]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    observed = joined.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not expected.equals(observed):
        raise AssertionError("pooled tail membership changed after side split/rejoin")
    return result


def _read_panel(path: Path, contract: Path) -> tuple[pd.DataFrame, list[str]]:
    features = json.loads(contract.read_text(encoding="utf-8"))["raw_feature_columns"]
    required = ["candidate_id", "__ts__", "__symbol__", "side_name", "frozen_base_score", "execution_label_end_utc", "execution_label_available_at", "execution_net_ev_12h", "__opportunity_occurred_12h__", "__adverse_competing_risk_12h__", "__mae_before_meaningful_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__", *features]
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(required)))
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    for name in ("execution_label_end_utc", "execution_label_available_at"):
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
    frame["execution_label_end_utc"] = frame["execution_label_end_utc"].fillna(frame["execution_label_available_at"])
    if frame["execution_label_end_utc"].isna().any() or frame.duplicated(["candidate_id"]).any():
        raise ValueError("unusable exact-H12 panel identity/availability")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame = frame.loc[frame.side_name.isin(SIDES) & frame.__ts__.ge(BASE_TRAIN_START) & frame.__ts__.lt(BASE_OOS_END)].copy()
    masks = calendar_masks(frame["__ts__"])
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("requested 12/8/4/4 calendar is incomplete")
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), features


def run(*, panel: Path, contract: Path, output: Path, seed: int = 20260730) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    frame, raw_features = _read_panel(panel, contract)
    masks = calendar_masks(frame["__ts__"])
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    all_predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    tail_memberships: list[pd.DataFrame] = []
    try:
        # Frozen-score context control, evaluated with precisely the same map
        # and the same pooled final book as every raw-model arm.
        history = frame.loc[masks["meta_train"], ["__ts__", "execution_label_end_utc", "execution_net_ev_12h", "frozen_base_score"]].rename(columns={"frozen_base_score": "raw_score"})
        evaluate = frame.loc[masks["meta_oos"], ["candidate_id", "side_name", "__symbol__", "__ts__", "execution_label_end_utc", "execution_net_ev_12h", "frozen_base_score"]].rename(columns={"frozen_base_score": "raw_score"})
        control = _causal_recent_map(history, evaluate)
        control["base_target_arm"] = "frozen_base_score_control"
        control["meta_target_arm"] = "not_applicable"
        all_predictions.append(control)
        metrics.extend(_metrics_with_side_decomposition(control, "frozen_base_score_control"))

        base_train = frame.loc[masks["base_train"]].copy().reset_index(drop=True)
        for base_index, base_arm in enumerate(BASE_ARMS):
            base_oos_score = np.full(len(frame), np.nan)
            selected_by_side: dict[str, list[str]] = {}
            for side_index, side in enumerate(SIDES):
                train = base_train.loc[base_train.side_name.eq(side)].reset_index(drop=True)
                target = base_target(train, base_arm.name)
                selected = _select_features(train, raw_features, target, seed=seed + base_index * 100 + side_index)
                selected_by_side[side] = selected
                feature_rows.extend({"base_target_arm": base_arm.name, "side_name": side, "rank": rank + 1, "feature": name} for rank, name in enumerate(selected))
                calibrator = _base_oof_calibrator(train, selected, target, seed=seed + base_index * 100 + side_index * 10 + 10)
                final = _fit(_matrix(train, selected), target, np.ones(len(train)), seed=seed + base_index * 100 + side_index * 10 + 19)
                position = np.flatnonzero(masks["base_oos"] & frame.side_name.eq(side).to_numpy())
                raw = final.predict(_matrix(frame.iloc[position], selected))
                base_oos_score[position] = calibrator.predict(raw)
                del final, calibrator
                gc.collect()
            if not np.isfinite(base_oos_score[masks["base_oos"]]).all():
                raise AssertionError("incomplete frozen base OOS predictions")
            working = frame.loc[masks["base_oos"]].copy().reset_index(drop=True)
            working["base_expected_net"] = base_oos_score[masks["base_oos"]]
            local_masks = calendar_masks(working["__ts__"])
            # Compute membership exactly once across both sides.  It is then
            # attached to candidate IDs before all side-local residual fits.
            working = _attach_pooled_tail_weights(working, local_masks["meta_train"])
            tail_memberships.append(
                working.loc[
                    local_masks["meta_train"],
                    ["candidate_id", "side_name", "__ts__", "pooled_tail_member", "pooled_tail_weight", "pooled_clean_tail_weight"],
                ].assign(base_target_arm=base_arm.name)
            )
            for meta_index, meta_arm in enumerate(META_ARMS):
                final_score = np.full(len(working), np.nan)
                history_parts: list[pd.DataFrame] = []
                for side_index, side in enumerate(SIDES):
                    train = working.loc[local_masks["meta_train"] & working.side_name.eq(side)].reset_index(drop=True)
                    test_position = np.flatnonzero(local_masks["meta_oos"] & working.side_name.eq(side).to_numpy())
                    features = ["base_expected_net", *selected_by_side[side]]
                    target = meta_target(train, meta_arm.name)
                    if meta_arm.name == "global_tail_weighted_residual":
                        weights = train["pooled_tail_weight"].to_numpy(float)
                    elif meta_arm.name == "clean_tail_weighted_residual":
                        weights = train["pooled_clean_tail_weight"].to_numpy(float)
                    else:
                        weights = np.ones(len(train))
                    model = _fit(_matrix(train, features), target, weights, seed=seed + base_index * 1000 + meta_index * 30 + side_index)
                    predicted = model.predict(_matrix(working.iloc[test_position], features))
                    final_score[test_position] = predicted if meta_arm.name == "policy_soft_clear" else working.iloc[test_position].base_expected_net.to_numpy(float) + predicted
                    del model
                    gc.collect()
                    oof = _meta_oof_scores(
                        train,
                        features,
                        meta_arm.name,
                        sample_weights=weights,
                        seed=seed + 9000 + base_index * 100 + meta_index * 10 + side_index,
                    )
                    part = train.loc[:, ["__ts__", "execution_label_end_utc", "execution_net_ev_12h"]].copy()
                    part["raw_score"] = oof
                    history_parts.append(part)
                if not np.isfinite(final_score[local_masks["meta_oos"]]).all():
                    raise AssertionError("incomplete residual OOS predictions")
                history = pd.concat(history_parts, ignore_index=True)
                evaluate = working.loc[local_masks["meta_oos"], ["candidate_id", "side_name", "__symbol__", "__ts__", "execution_label_end_utc", "execution_net_ev_12h"]].copy()
                evaluate["raw_score"] = final_score[local_masks["meta_oos"]]
                mapped = _causal_recent_map(history, evaluate)
                arm_name = f"{base_arm.name}__{meta_arm.name}"
                mapped["base_target_arm"] = base_arm.name
                mapped["meta_target_arm"] = meta_arm.name
                all_predictions.append(mapped)
                metrics.extend(_metrics_with_side_decomposition(mapped, arm_name))
                gc.collect()
        predictions = pd.concat(all_predictions, ignore_index=True)
        pd.DataFrame(metrics).to_csv(stage / "pooled_global_book_metrics.csv", index=False)
        pd.DataFrame(feature_rows).to_csv(stage / "base_selected_features.csv", index=False)
        pd.concat(tail_memberships, ignore_index=True).to_parquet(stage / "pooled_tail_membership.parquet", index=False)
        predictions.to_parquet(stage / "meta_oos_predictions.parquet", index=False)
        manifest = {
            "schema": "long_raw_base_residual_h12_ablation_v1",
            "status": "COMPLETED_CANDIDATE_CONDITIONED_COUNTERFACTUAL_RESEARCH_NO_PROMOTION",
            "input": {"panel": str(panel), "panel_sha256": _sha256(panel), "feature_contract": str(contract), "feature_contract_sha256": _sha256(contract)},
            "calendar": {"base_train": "2023-04..2024-03", "base_oos": "2024-04..2024-11", "meta_train": "2024-04..2024-07", "meta_oos": "2024-08..2024-11", "walk_forward_required": False},
            "targets": {
                "base": "three soft labels, each net of row cost with a +25bp post-cost hurdle; risk/timing terms only tighten or modulate that net label",
                "meta": "net residual, globally top-decile weighted residual, soft post-cost-clear probability, and clean globally top-decile weighted residual",
            },
            "feature_selection": f"per base target and side, gain screen using only 2023-04..2024-03; top {TOP_FEATURES}; residual sees base expected net plus that frozen side-local base feature set",
            "selection": "one pooled global top-k across both sides and all timestamps after one pooled causal 21-day isotonic map; no per-timestamp/side/asset rerank or quota",
            "tail_weight_contract": "weighted residual arms use one immutable pooled-global top-10 membership over the complete Apr..Jul residual-training cohort before side splitting; every residual OOF fold slices the same candidate membership",
            "oof": "base target-to-net calibrator and residual mapping history are chronological blocked OOF; final residual evaluation is untouched Aug..Nov 2024",
            "limitations": ["candidate-conditioned old selected/monitor population", "current-spread counterfactual rather than factual historical execution", "no historical L2 or bit-exact pre-2025 geometry parity", "research evidence only; no promotion"],
            "arms": {"base": [arm.name for arm in BASE_ARMS], "meta": [arm.name for arm in META_ARMS]},
            "outputs": {name: _sha256(stage / name) for name in ("pooled_global_book_metrics.csv", "base_selected_features.csv", "pooled_tail_membership.parquet", "meta_oos_predictions.parquet")},
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--feature-contract", type=Path, default=FEATURE_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    print(json.dumps(run(panel=args.panel, contract=args.feature_contract, output=args.output, seed=args.seed), indent=2, default=str))


if __name__ == "__main__":
    main()
