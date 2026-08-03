#!/usr/bin/env python3
"""Long-horizon exact-H12 base/reidual target ablation.

This is a diagnostic re-targeting study over a frozen candidate panel, not a
production refit.  The raw base feature store is not available across the
required historical span, so the base layer here is explicitly an adapter over
the frozen OOF ``score_base_alpha`` plus decision-time regime/transition
context.  It still tests the relevant question: whether cost-clearing H12
opportunity targets make a better handoff to a policy-focused residual layer.

Calendar (UTC, non-walk-forward by request):

* base fit: 2023-04-01 through 2024-03-31 (12 months);
* frozen base predictions: 2024-04-01 through 2024-11-30 (8 months);
* residual/meta fit: 2024-04-01 through 2024-07-31 (first 4 OOS months); and
* untouched residual/meta predictions: 2024-08-01 through 2024-11-30.

All primary selection is a single pooled global top-k book across both sides
and all timestamps.  Month rows only decompose that fixed membership; they do
not rerank within a month or timestamp.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/long_base_residual_h12_target_ablation_20260730_v3"
SIDES = ("long", "short")
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
CALIBRATION_DAYS = 21

BASE_TRAIN_START = pd.Timestamp("2023-04-01T00:00:00Z")
BASE_TRAIN_END = pd.Timestamp("2024-04-01T00:00:00Z")
BASE_OOS_END = pd.Timestamp("2024-12-01T00:00:00Z")
META_TRAIN_END = pd.Timestamp("2024-08-01T00:00:00Z")

REGIME_FEATURES = (
    "regime_state_entropy", "regime_state_margin", "regime_state_uncertainty",
    "regime_state_ood_score",
)
TRANSITION_FEATURES = (
    "transition_active_probability", "transition_state_entropy",
    "transition_state_margin", "transition_state_uncertainty",
    "transition_state_ood_score", "transition_state_p__stable",
    "transition_state_p__approach", "transition_state_p__immediate_lead",
    "transition_state_p__transition", "transition_state_p__acceleration",
    "transition_state_p__early_destination", "transition_state_p__settled_destination",
)
BASE_FEATURES = ("score_base_alpha", *REGIME_FEATURES, *TRANSITION_FEATURES)
META_FEATURES = ("base_expected_net", *REGIME_FEATURES, *TRANSITION_FEATURES)


@dataclass(frozen=True)
class Arm:
    name: str
    role: str


BASE_ARMS = (
    Arm("cost_clear_0bps", "base"),
    Arm("cost_clear_25bps", "base"),
    Arm("cost_clear_upside", "base"),
)
META_ARMS = (
    Arm("direct_net", "meta"),
    Arm("net_residual", "meta"),
    Arm("tail_weighted_net_residual", "meta"),
    Arm("cost_clear_25bps", "meta"),
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


def global_top_mask(score: Iterable[float], fraction: float) -> np.ndarray:
    """Stable global book selection; never group by timestamp or side."""

    values = np.nan_to_num(np.asarray(list(score), dtype=float), nan=-np.inf)
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(np.ceil(len(values) * float(fraction))))
    selected = np.zeros(len(values), dtype=bool)
    selected[np.argsort(-values, kind="mergesort")[:count]] = True
    return selected


def calendar_masks(timestamps: Iterable[object]) -> dict[str, np.ndarray]:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    return {
        "base_train": (ts.ge(BASE_TRAIN_START) & ts.lt(BASE_TRAIN_END)).to_numpy(),
        "base_oos": (ts.ge(BASE_TRAIN_END) & ts.lt(BASE_OOS_END)).to_numpy(),
        "meta_train": (ts.ge(BASE_TRAIN_END) & ts.lt(META_TRAIN_END)).to_numpy(),
        "meta_oos": (ts.ge(META_TRAIN_END) & ts.lt(BASE_OOS_END)).to_numpy(),
    }


def base_target(frame: pd.DataFrame, arm: str) -> np.ndarray:
    net = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float)
    gross = pd.to_numeric(frame["execution_gross_ev_12h"], errors="raise").to_numpy(float)
    cost = pd.to_numeric(frame["execution_cost_return"], errors="raise").to_numpy(float)
    if arm == "cost_clear_0bps":
        return _sigmoid(net / 0.010)
    if arm == "cost_clear_25bps":
        return _sigmoid((net - 0.0025) / 0.010)
    if arm == "cost_clear_upside":
        # A cost-clearing hurdle plus the excess gross path above row cost.
        return _sigmoid(net / 0.010) * _sigmoid((gross - cost - 0.0025) / 0.0125)
    raise ValueError(f"unknown base target arm: {arm}")


def _matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"panel is missing causal input fields: {missing}")
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _fit_regressor(matrix: pd.DataFrame, target: np.ndarray, weights: np.ndarray, *, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=220, learning_rate=0.035,
        num_leaves=23, max_depth=5, min_child_samples=160,
        colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
        reg_lambda=12.0, reg_alpha=0.10, random_state=int(seed),
        n_jobs=2, verbosity=-1,
    )
    model.fit(matrix, target, sample_weight=weights)
    return model


def _fit_classifier(matrix: pd.DataFrame, target: np.ndarray, weights: np.ndarray, *, seed: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=220, learning_rate=0.035,
        num_leaves=23, max_depth=5, min_child_samples=160,
        colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
        reg_lambda=12.0, reg_alpha=0.10, random_state=int(seed),
        n_jobs=2, verbosity=-1,
    )
    model.fit(matrix, target.astype(int), sample_weight=weights)
    return model


def _fold_ids(frame: pd.DataFrame, n_folds: int = 4) -> np.ndarray:
    """Static chronological blocks: OOF calibration, deliberately not WF."""

    ordered = frame.loc[:, ["__ts__"]].copy().sort_values("__ts__", kind="stable")
    blocks = np.array_split(np.arange(len(ordered)), n_folds)
    result = np.empty(len(frame), dtype=int)
    for fold, positions in enumerate(blocks):
        result[ordered.index.to_numpy()[positions]] = fold
    return result


def _crossfit_base_calibrator(frame: pd.DataFrame, matrix: pd.DataFrame, target: np.ndarray, *, seed: int) -> tuple[IsotonicRegression, np.ndarray]:
    oof = np.full(len(frame), np.nan, dtype=float)
    folds = _fold_ids(frame)
    for fold in np.unique(folds):
        train = folds != fold
        valid = ~train
        model = _fit_regressor(matrix.loc[train], target[train], np.ones(int(train.sum())), seed=seed + int(fold))
        oof[valid] = model.predict(matrix.loc[valid])
        del model
        gc.collect()
    net = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof, net)
    return calibrator, oof


def _causal_recent_map(history: pd.DataFrame, evaluate: pd.DataFrame) -> pd.DataFrame:
    """One pooled, 21-day causal score-to-net map; no side or timestamp maps."""

    required = {"__ts__", "execution_label_end_utc", "raw_score", "execution_net_ev_12h"}
    missing = sorted(required.difference(history.columns) | required.difference(evaluate.columns))
    if missing:
        raise ValueError(f"recent map fields missing: {missing}")
    full = pd.concat([history, evaluate], ignore_index=True, sort=False)
    full["__ts__"] = pd.to_datetime(full["__ts__"], utc=True, errors="raise")
    full["execution_label_end_utc"] = pd.to_datetime(full["execution_label_end_utc"], utc=True, errors="raise")
    result = evaluate.copy().reset_index(drop=True)
    result["mapped_expected_net"] = np.nan
    result["map_reference_rows"] = 0
    day = pd.to_datetime(result["__ts__"], utc=True).dt.floor("D")
    for value in day.drop_duplicates().sort_values():
        mask = day.eq(value).to_numpy()
        start = value - pd.Timedelta(days=CALIBRATION_DAYS)
        reference = full.loc[(full["__ts__"].ge(start)) & (full["execution_label_end_utc"].lt(value))].copy()
        reference = reference.loc[np.isfinite(reference["raw_score"]) & np.isfinite(reference["execution_net_ev_12h"])]
        result.loc[mask, "map_reference_rows"] = int(len(reference))
        if len(reference) < 500 or reference["raw_score"].nunique() < 2:
            continue
        mapper = IsotonicRegression(out_of_bounds="clip")
        mapper.fit(reference["raw_score"], reference["execution_net_ev_12h"])
        result.loc[mask, "mapped_expected_net"] = mapper.predict(result.loc[mask, "raw_score"])
    return result


def _book_metrics(scored: pd.DataFrame, score_column: str, *, arm: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    valid = scored.loc[np.isfinite(scored[score_column])].copy().reset_index(drop=True)
    for fraction in TOP_FRACTIONS:
        selected = global_top_mask(valid[score_column], fraction)
        book = valid.loc[selected].copy()
        record = {
            "arm": arm, "scope": "one_pooled_global_book", "fraction": float(fraction),
            "eligible_rows": int(len(valid)), "selected_rows": int(len(book)),
            "mean_net_bps": float(book.execution_net_ev_12h.mean() * 10_000.0),
            "positive_net_rate": float(book.execution_net_ev_12h.gt(0.0).mean()),
            "long_share": float(book.side_name.eq("long").mean()),
            "rank_ic": float(valid[score_column].corr(valid.execution_net_ev_12h, method="spearman")),
        }
        output.append(record)
        # Decomposition uses the above global selection exactly; no month rerank.
        for month, part in book.assign(month=pd.to_datetime(book["__ts__"], utc=True).dt.strftime("%Y-%m")).groupby("month", sort=True):
            output.append({**record, "scope": "global_book_membership_by_month", "month": month,
                           "selected_rows": int(len(part)), "mean_net_bps": float(part.execution_net_ev_12h.mean() * 10_000.0),
                           "positive_net_rate": float(part.execution_net_ev_12h.gt(0.0).mean()), "long_share": float(part.side_name.eq("long").mean())})
    return output


def _read_panel(path: Path) -> pd.DataFrame:
    columns = ["candidate_id", "side_name", "__symbol__", "__ts__", "execution_label_end_utc", "execution_label_available_at", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "score_residual_expected_ev", *BASE_FEATURES]
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["execution_label_end_utc"] = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="raise")
    available = pd.to_datetime(frame["execution_label_available_at"], utc=True, errors="raise")
    # The historical reconstructed rows omit the endpoint but retain their
    # exact resolved-at timestamp.  Requiring availability is strictly later
    # than requiring the H12 endpoint and therefore remains safe for mapping.
    frame["execution_label_end_utc"] = frame["execution_label_end_utc"].fillna(available)
    if frame["execution_label_end_utc"].isna().any():
        raise ValueError("exact-H12 label availability is missing")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame = frame.loc[frame.side_name.isin(SIDES) & frame["__ts__"].ge(BASE_TRAIN_START) & frame["__ts__"].lt(BASE_OOS_END)].copy()
    if frame.duplicated(["candidate_id", "side_name", "__symbol__", "__ts__"]).any():
        raise ValueError("candidate identity is not unique")
    masks = calendar_masks(frame["__ts__"])
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("requested calendar is not fully represented")
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def run(*, panel: Path, output: Path, seed: int = 20260730) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    frame = _read_panel(panel)
    masks = calendar_masks(frame["__ts__"])
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    try:
        # Immutable controls: exact same historical/evaluation split, mapper,
        # and global-book metric as the target arms below.
        for control_name, score_column in (
            ("control_frozen_base_alpha", "score_base_alpha"),
            ("control_frozen_residual_ev", "score_residual_expected_ev"),
        ):
            history = frame.loc[masks["meta_train"], ["__ts__", "execution_label_end_utc", "execution_net_ev_12h", score_column]].copy()
            history = history.rename(columns={score_column: "raw_score"})
            evaluate = frame.loc[masks["meta_oos"], ["candidate_id", "side_name", "__symbol__", "__ts__", "execution_label_end_utc", "execution_net_ev_12h", score_column]].copy()
            evaluate = evaluate.rename(columns={score_column: "raw_score"})
            mapped = _causal_recent_map(history, evaluate)
            mapped["base_target_arm"] = control_name
            mapped["meta_target_arm"] = "not_applicable"
            all_predictions.append(mapped)
            metrics.extend(_book_metrics(mapped, "mapped_expected_net", arm=control_name))
        for base_index, base_arm in enumerate(BASE_ARMS):
            base_prediction = np.full(len(frame), np.nan, dtype=float)
            base_train = frame.loc[masks["base_train"]].copy().reset_index(drop=True)
            for side_index, side in enumerate(SIDES):
                train_local = base_train.loc[base_train.side_name.eq(side)].reset_index(drop=True)
                train_x = _matrix(train_local, BASE_FEATURES)
                target = base_target(train_local, base_arm.name)
                calibrator, _ = _crossfit_base_calibrator(train_local, train_x, target, seed=seed + base_index * 100 + side_index * 10)
                full_train_x = _matrix(train_local, BASE_FEATURES)
                final = _fit_regressor(full_train_x, target, np.ones(len(train_local)), seed=seed + base_index * 100 + side_index * 10 + 9)
                oos_position = np.flatnonzero(masks["base_oos"] & frame.side_name.eq(side).to_numpy())
                raw = final.predict(_matrix(frame.iloc[oos_position], BASE_FEATURES))
                base_prediction[oos_position] = calibrator.predict(raw)
                del final, calibrator
                gc.collect()
            if not np.isfinite(base_prediction[masks["base_oos"]]).all():
                raise AssertionError("base OOS predictions incomplete")
            working = frame.loc[masks["base_oos"]].copy().reset_index(drop=True)
            working["base_expected_net"] = base_prediction[masks["base_oos"]]
            meta_mask = calendar_masks(working["__ts__"])
            for meta_index, meta_arm in enumerate(META_ARMS):
                raw_meta = np.full(len(working), np.nan, dtype=float)
                threshold = float(working.loc[meta_mask["meta_train"], "base_expected_net"].quantile(0.90))
                for side_index, side in enumerate(SIDES):
                    train_pos = np.flatnonzero(meta_mask["meta_train"] & working.side_name.eq(side).to_numpy())
                    test_pos = np.flatnonzero(meta_mask["meta_oos"] & working.side_name.eq(side).to_numpy())
                    train = working.iloc[train_pos]
                    x_train = _matrix(train, META_FEATURES)
                    x_test = _matrix(working.iloc[test_pos], META_FEATURES)
                    net = train.execution_net_ev_12h.to_numpy(float)
                    weights = np.ones(len(train), dtype=float)
                    if meta_arm.name == "direct_net":
                        model = _fit_regressor(x_train, net, weights, seed=seed + base_index * 1000 + meta_index * 30 + side_index)
                        raw_meta[test_pos] = model.predict(x_test)
                    elif meta_arm.name == "net_residual":
                        model = _fit_regressor(x_train, net - train.base_expected_net.to_numpy(float), weights, seed=seed + base_index * 1000 + meta_index * 30 + side_index)
                        raw_meta[test_pos] = working.iloc[test_pos].base_expected_net.to_numpy(float) + model.predict(x_test)
                    elif meta_arm.name == "tail_weighted_net_residual":
                        weights = np.where(train.base_expected_net.to_numpy(float) >= threshold, 4.0, 1.0)
                        model = _fit_regressor(x_train, net - train.base_expected_net.to_numpy(float), weights, seed=seed + base_index * 1000 + meta_index * 30 + side_index)
                        raw_meta[test_pos] = working.iloc[test_pos].base_expected_net.to_numpy(float) + model.predict(x_test)
                    else:
                        target = (net > 0.0025).astype(float)
                        weights = np.where(train.base_expected_net.to_numpy(float) >= threshold, 3.0, 1.0)
                        model = _fit_classifier(x_train, target, weights, seed=seed + base_index * 1000 + meta_index * 30 + side_index)
                        raw_meta[test_pos] = model.predict_proba(x_test)[:, 1]
                    del model
                    gc.collect()
                history = working.loc[meta_mask["meta_train"], ["__ts__", "execution_label_end_utc", "execution_net_ev_12h"]].copy()
                history["raw_score"] = np.nan
                # Fit a score-only crossfit on meta training for map history, avoiding in-sample score calibration.
                for side in SIDES:
                    pos = np.flatnonzero(meta_mask["meta_train"] & working.side_name.eq(side).to_numpy())
                    local = working.iloc[pos].reset_index(drop=True)
                    x = _matrix(local, META_FEATURES)
                    net = local.execution_net_ev_12h.to_numpy(float)
                    folds = _fold_ids(local)
                    local_score = np.full(len(local), np.nan)
                    for fold in np.unique(folds):
                        tr, va = folds != fold, folds == fold
                        fold_seed = seed + 9000 + base_index * 100 + meta_index * 10 + int(fold)
                        if meta_arm.name == "direct_net":
                            model = _fit_regressor(x.loc[tr], net[tr], np.ones(int(tr.sum())), seed=fold_seed)
                            local_score[va] = model.predict(x.loc[va])
                        elif meta_arm.name == "net_residual":
                            model = _fit_regressor(
                                x.loc[tr], net[tr] - local.base_expected_net.to_numpy(float)[tr],
                                np.ones(int(tr.sum())), seed=fold_seed,
                            )
                            local_score[va] = local.base_expected_net.to_numpy(float)[va] + model.predict(x.loc[va])
                        elif meta_arm.name == "tail_weighted_net_residual":
                            fold_weights = np.where(local.base_expected_net.to_numpy(float)[tr] >= threshold, 4.0, 1.0)
                            model = _fit_regressor(
                                x.loc[tr], net[tr] - local.base_expected_net.to_numpy(float)[tr],
                                fold_weights, seed=fold_seed,
                            )
                            local_score[va] = local.base_expected_net.to_numpy(float)[va] + model.predict(x.loc[va])
                        else:
                            fold_weights = np.where(local.base_expected_net.to_numpy(float)[tr] >= threshold, 3.0, 1.0)
                            model = _fit_classifier(x.loc[tr], (net[tr] > 0.0025), fold_weights, seed=fold_seed)
                            local_score[va] = model.predict_proba(x.loc[va])[:, 1]
                        del model
                        gc.collect()
                    history.loc[history.index.isin(working.index[pos]), "raw_score"] = local_score
                evaluate = working.loc[meta_mask["meta_oos"], ["candidate_id", "side_name", "__symbol__", "__ts__", "execution_label_end_utc", "execution_net_ev_12h"]].copy()
                evaluate["raw_score"] = raw_meta[meta_mask["meta_oos"]]
                mapped = _causal_recent_map(history.dropna(subset=["raw_score"]), evaluate)
                arm_name = f"{base_arm.name}__{meta_arm.name}"
                mapped["base_target_arm"] = base_arm.name
                mapped["meta_target_arm"] = meta_arm.name
                all_predictions.append(mapped)
                metrics.extend(_book_metrics(mapped, "mapped_expected_net", arm=arm_name))
                gc.collect()
        predictions = pd.concat(all_predictions, ignore_index=True)
        pd.DataFrame(metrics).to_csv(stage / "pooled_global_book_metrics.csv", index=False)
        predictions.to_parquet(stage / "meta_oos_predictions.parquet", index=False)
        manifest = {
            "schema": "long_base_residual_h12_target_ablation_v3",
            "status": "COMPLETED_DIAGNOSTIC_NO_PROMOTION",
            "input": {"path": str(panel), "sha256": _sha256(panel)},
            "calendar": {"base_train": "2023-04..2024-03", "base_oos": "2024-04..2024-11", "meta_train": "2024-04..2024-07", "meta_oos": "2024-08..2024-11", "walk_forward_required": False},
            "target_contract": {
                "base": "cost-clearing exact-H12 opportunity soft targets; all targets use row net or gross minus row cost",
                "meta": "exact-H12 net or net-conversion residual; tail arm weights the pooled global base top decile only",
                "cost": "execution_net_ev_12h is already gross minus row cost; no cost is subtracted a second time",
            },
            "selection_contract": "one pooled global top-k across both sides and every evaluation timestamp; month rows are membership decomposition only",
            "base_role": "historically-supported re-targeting adapter over frozen base-alpha plus decision-time context, not a raw-feature production base refit",
            "controls": "frozen base-alpha and frozen residual-EV scores replayed with the identical pooled 21-day map and global-book evaluation",
            "base_arms": [arm.name for arm in BASE_ARMS], "meta_arms": [arm.name for arm in META_ARMS],
            "mapping": "one pooled 21-day causal isotonic score-to-net map; references require exact H12 label availability before map day (strictly safer than endpoint-only); no side/timestamp map",
            "outputs": {name: _sha256(stage / name) for name in ("pooled_global_book_metrics.csv", "meta_oos_predictions.parquet")},
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    print(json.dumps(run(panel=args.panel, output=args.output, seed=args.seed), indent=2, default=str))


if __name__ == "__main__":
    main()
