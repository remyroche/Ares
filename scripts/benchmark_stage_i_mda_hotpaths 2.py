#!/usr/bin/env python3
"""Small reproducible benchmark for Stage-I MDA preparation hot paths.

This never fits a model or writes artifacts.  It is deliberately isolated from
the live selector so it can be run while a Stage-I experiment is in progress:

    PYTHONPATH=. python3 scripts/benchmark_stage_i_mda_hotpaths.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import lgbm_pipeline as lp


def _median_seconds(callback, repeats: int) -> float:
    elapsed = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        callback()
        elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed))


def _legacy_availability(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[np.ndarray, ...]:
    train_numeric = train.apply(pd.to_numeric, errors="coerce")
    evaluation_numeric = evaluation.apply(pd.to_numeric, errors="coerce")
    train_finite = np.isfinite(train_numeric.to_numpy(dtype=np.float32, copy=False))
    evaluation_finite = np.isfinite(
        evaluation_numeric.to_numpy(dtype=np.float32, copy=False)
    )
    return (
        train_finite.sum(axis=0).astype(np.int32),
        evaluation_finite.sum(axis=0).astype(np.int32),
        np.asarray(
            [
                train_numeric.iloc[:, column].dropna().nunique() > 1
                for column in range(train.shape[1])
            ],
            dtype=bool,
        ),
    )


def _legacy_missingness_permutation(
    values: np.ndarray,
    affected: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """The former per-feature preparation path, retained only for benchmarking."""
    subset = values[affected].copy()
    column = values[:, 0]
    finite = np.flatnonzero(np.isfinite(column)).astype(np.int32)
    mapped = column.copy()
    if len(finite) > 1:
        mapped[finite] = column[
            finite[np.argsort(permutation[finite], kind="stable")]
        ]
    subset[:, 0] = mapped[affected]
    return subset


class _BenchmarkBooster:
    def predict(self, values):
        values = np.asarray(values, dtype=np.float32)
        self.last = values.copy()
        # Stand-in only: actual LightGBM scoring normally dominates this work.
        return np.nan_to_num(values[:, 0], nan=0.0) + 0.01 * values[:, 1]


class _BenchmarkModel:
    booster_ = _BenchmarkBooster()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--features", type=int, default=340)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--lightgbm-batch", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_values = rng.normal(size=(args.rows, args.features)).astype(np.float32)
    evaluation_values = rng.normal(size=(args.rows // 3, args.features)).astype(np.float32)
    # Mimic a mix of causal readiness prefixes and genuinely sparse fields.
    warmup_columns = np.arange(0, args.features, max(1, args.features // 20))
    train_values[: args.rows // 8, warmup_columns] = np.nan
    train_values[::19, warmup_columns[::2]] = np.nan
    train = pd.DataFrame(train_values)
    evaluation = pd.DataFrame(evaluation_values)

    legacy = _legacy_availability(train, evaluation)
    vectorised = lp._mda_feature_availability_from_numeric_values(
        train_values,
        evaluation_values,
        min_train_finite_rows=64,
        min_evaluation_finite_rows=32,
    )
    equivalent = (
        np.array_equal(legacy[0], vectorised[0])
        and np.array_equal(legacy[1], vectorised[1])
        and np.array_equal(legacy[2], vectorised[2])
    )
    if not equivalent:
        raise SystemExit("availability benchmark detected an output mismatch")
    legacy_seconds = _median_seconds(
        lambda: _legacy_availability(train, evaluation), args.repeats
    )
    vectorised_seconds = _median_seconds(
        lambda: lp._mda_feature_availability_from_numeric_values(
            train_values,
            evaluation_values,
            min_train_finite_rows=64,
            min_evaluation_finite_rows=32,
        ),
        args.repeats,
    )
    perm_values = train_values[:, : min(8, args.features)].copy()
    perm_values[: args.rows // 8, 0] = np.nan
    perm_frame = pd.DataFrame(perm_values)
    affected = np.arange(len(perm_frame), dtype=np.int32)
    permutations = [rng.permutation(len(perm_frame)).astype(np.int32) for _ in range(12)]
    legacy_permuted = [
        _legacy_missingness_permutation(perm_values, affected, order)
        for order in permutations
    ]
    cached_values = perm_frame.to_numpy(dtype=np.float32, copy=False)
    cache = lp._mda_missing_finite_positions(cached_values)
    new_permuted = []
    model = _BenchmarkModel()
    for order in permutations:
        lp._predict_permuted_rows(
            model,
            perm_frame,
            np.zeros(len(perm_frame), dtype=np.float32),
            [0],
            order,
            classifier=False,
            feature_names=list(perm_frame.columns),
            affected_rows_by_feature=None,
            mode="full",
            preserve_feature_missingness=True,
            X_valid_values=cached_values,
            affected_idx=affected,
            affected_method="full_permutation",
            finite_positions_by_feature=cache,
        )
        new_permuted.append(model.booster_.last.copy())
    permutation_equivalent = all(
        np.array_equal(old, new, equal_nan=True)
        for old, new in zip(legacy_permuted, new_permuted)
    )
    if not permutation_equivalent:
        raise SystemExit("missingness permutation benchmark detected an output mismatch")
    legacy_permutation_seconds = _median_seconds(
        lambda: [
            _BenchmarkBooster().predict(
                _legacy_missingness_permutation(perm_values, affected, order)
            )
            for order in permutations
        ],
        args.repeats,
    )
    fast_permutation_seconds = _median_seconds(
        lambda: [
            lp._predict_permuted_rows(
                model,
                perm_frame,
                np.zeros(len(perm_frame), dtype=np.float32),
                [0],
                order,
                classifier=False,
                feature_names=list(perm_frame.columns),
                affected_rows_by_feature=None,
                mode="full",
                preserve_feature_missingness=True,
                X_valid_values=cached_values,
                affected_idx=affected,
                affected_method="full_permutation",
                finite_positions_by_feature=cache,
            )
            for order in permutations
        ],
        args.repeats,
    )
    report = {
                "rows": int(args.rows),
                "features": int(args.features),
                "availability_equivalent": equivalent,
                "legacy_median_seconds": legacy_seconds,
                "vectorised_median_seconds": vectorised_seconds,
                "speedup": legacy_seconds / max(vectorised_seconds, 1e-12),
                "permutation_mask_equivalent": permutation_equivalent,
                "legacy_12_repeat_preparation_seconds": legacy_permutation_seconds,
                "cached_12_repeat_preparation_seconds": fast_permutation_seconds,
                "cached_permutation_speedup": legacy_permutation_seconds
                / max(fast_permutation_seconds, 1e-12),
                "note": (
                    "This measures preparation only; LightGBM prediction remains the "
                    "dominant cost for full MDA permutations."
                ),
    }
    if args.lightgbm_batch:
        import lightgbm as lgb

        batch_features = min(64, args.features)
        batch_rows = min(8_000, args.rows)
        batch_values = train_values[:batch_rows, :batch_features].copy()
        batch_values[:, 0] = np.nan_to_num(batch_values[:, 0], nan=0.0)
        batch_frame = pd.DataFrame(
            batch_values, columns=[f"f{i}" for i in range(batch_features)]
        )
        target = (
            np.nan_to_num(batch_values[:, 0], nan=0.0)
            - 0.3 * np.nan_to_num(batch_values[:, 1], nan=0.0)
        )
        fitted = lgb.LGBMRegressor(
            n_estimators=80, max_depth=5, num_leaves=31,
            learning_rate=0.05, verbosity=-1, n_jobs=1, random_state=args.seed,
        ).fit(batch_frame, target)
        baseline = fitted.predict(batch_frame).astype(np.float32)
        batch_orders = [
            rng.permutation(batch_rows).astype(np.int32) for _ in range(4)
        ]
        common = dict(
            model=fitted, X_valid=batch_frame, baseline_pred=baseline,
            feature_indices=[0], classifier=False,
            feature_names=list(batch_frame.columns), affected_rows_by_feature=None,
            mode="full", preserve_feature_missingness=False,
            X_valid_values=batch_values,
        )
        sequential_values = [
            lp._predict_permuted_rows(perm_order=order, **common)[0]
            for order in batch_orders
        ]
        batched_values = [
            item[0]
            for item in lp._predict_permuted_rows_batch(
                perm_orders=batch_orders, **common
            )
        ]
        if not all(
            np.array_equal(left, right)
            for left, right in zip(sequential_values, batched_values)
        ):
            raise SystemExit("batched LightGBM prediction changed MDA predictions")
        sequential_seconds = _median_seconds(
            lambda: [
                lp._predict_permuted_rows(perm_order=order, **common)
                for order in batch_orders
            ],
            args.repeats,
        )
        batch_seconds = _median_seconds(
            lambda: lp._predict_permuted_rows_batch(
                perm_orders=batch_orders, **common
            ),
            args.repeats,
        )
        report.update(
            {
                "lightgbm_batch_rows": batch_rows,
                "lightgbm_batch_features": batch_features,
                "lightgbm_batch_predictions": len(batch_orders),
                "lightgbm_batch_exact_parity": True,
                "lightgbm_sequential_seconds": sequential_seconds,
                "lightgbm_batched_seconds": batch_seconds,
                "lightgbm_prediction_batch_speedup": sequential_seconds
                / max(batch_seconds, 1e-12),
            }
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
