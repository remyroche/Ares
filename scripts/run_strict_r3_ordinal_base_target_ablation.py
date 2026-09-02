#!/usr/bin/env python3
"""Strict 3m/3m ordinal base-target ablations for either trading side.

This is deliberately a base-layer experiment.  It compares the current R3
three-class target with a small, predeclared family of ordinal H12-net targets
on identical point-in-time candidates, features, train/OOS windows and frozen
base parameters.  For the short side it additionally compares training-only
sample weighting and evaluates the resulting score tails with a side-correct,
frozen exact-one-minute parent SimplePolicy outcome.  The policy outcome is
joined only *after* each arm has ranked all OOS executable candidates.

The runner is not a policy HPO and does not promote an artifact.  In
particular, the symmetric short policy is a fixed economic diagnostic, not a
borrowed long-policy winner.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_policy_contract import (  # noqa: E402
    Exact1mExecutionContract,
    Exact1mPolicyParams,
    simulate_exact_1m_parent_policy,
)


TRAIN_START = pd.Timestamp("2024-01-01T00:00:00Z")
OOS_START = pd.Timestamp("2024-04-01T00:00:00Z")
OOS_END = pd.Timestamp("2024-07-01T00:00:00Z")
TAILS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
POLICY_TAILS = (0.01, 0.02, 0.05, 0.10)
SEED = 17
FEATURE_CONTRACT = ROOT / "config/strict_r3_canonical_v2_feature_contract.json"
MINUTE_ROOT = ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv"

# Canonical current R3 base parameters.  We intentionally do not HPO the
# ordinal arms on April--June: only their target and, for the short side, the
# predeclared training-loss weighting differ.
FROZEN_BASE_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 140,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 350,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_lambda": 8.0,
    "random_state": SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    family: Literal["r3", "ordinal"]
    description: str
    edges: tuple[float, float, float] | None = None
    weight_mode: Literal["uniform", "boundary_certainty", "mild_class", "hybrid"] = "uniform"


COMMON_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        "R3_current_control",
        "r3",
        "Current base target: adverse-first / weak / robust clear; score=P(clear)-0.5*P(adverse).",
    ),
    TargetSpec("O_n150_z0_p25_uniform", "ordinal", "<=-150, -150..0, 0..+25, >+25 bps.", (-150.0, 0.0, 25.0)),
    TargetSpec("O_n200_z0_p50_uniform", "ordinal", "<=-200, -200..0, 0..+50, >+50 bps.", (-200.0, 0.0, 50.0)),
    TargetSpec("O_n250_z0_p50_uniform", "ordinal", "<=-250, -250..0, 0..+50, >+50 bps.", (-250.0, 0.0, 50.0)),
    TargetSpec("O_n250_z0_p75_uniform", "ordinal", "<=-250, -250..0, 0..+75, >+75 bps.", (-250.0, 0.0, 75.0)),
    TargetSpec("O_n300_z0_p100_uniform", "ordinal", "<=-300, -300..0, 0..+100, >+100 bps.", (-300.0, 0.0, 100.0)),
)
SHORT_WEIGHT_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("O_n200_z0_p50_boundary", "ordinal", "n200/p50 with boundary-certainty weight.", (-200.0, 0.0, 50.0), "boundary_certainty"),
    TargetSpec("O_n200_z0_p50_mildclass", "ordinal", "n200/p50 with mild square-root class balance.", (-200.0, 0.0, 50.0), "mild_class"),
    TargetSpec("O_n200_z0_p50_hybrid", "ordinal", "n200/p50 with certainty × mild class balance.", (-200.0, 0.0, 50.0), "hybrid"),
    TargetSpec("O_n250_z0_p75_hybrid", "ordinal", "n250/p75 with certainty × mild class balance.", (-250.0, 0.0, 75.0), "hybrid"),
)


def _packb_to_kraken_symbol(symbol: str) -> str:
    value = str(symbol).strip()
    if not value:
        raise ValueError("blank short symbol")
    return value.replace("/", "_")


def _minute_path_pruned(root: Path, symbol: str, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    """Read only immutable overlapping minute fragments, preserving gaps."""
    location = root / f"symbol={symbol}"
    selected: list[Path] = []
    if location.exists():
        for year in range(start.year, (end_exclusive - pd.Timedelta(minutes=1)).year + 1):
            for path in sorted((location / f"year={year}").glob("*.parquet")):
                tokens = path.stem.rsplit("-", 2)
                try:
                    first, last = int(tokens[-2]), int(tokens[-1])
                except (ValueError, IndexError):
                    selected.append(path)
                    continue
                if last >= int(start.timestamp()) and first < int(end_exclusive.timestamp()):
                    selected.append(path)
    tables = [pq.ParquetFile(path).read(columns=["ts", "open", "high", "low", "close"]) for path in selected]
    if tables:
        raw = pa.concat_tables(tables, promote_options="permissive").to_pandas()
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="raise")
        raw = raw.loc[raw.ts.ge(start) & raw.ts.lt(end_exclusive)]
        raw = raw.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    else:
        raw = pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC"))
    grid = pd.date_range(start.floor("min"), (end_exclusive - pd.Timedelta(minutes=1)).floor("min"), freq="min", tz="UTC")
    return raw.reindex(grid)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: pd.Series | pd.Index) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _side_paths(side: str) -> dict[str, Path]:
    if side == "long":
        return {
            "features": ROOT / "data_perp/artifacts/strict_r3_schema_v2_features_long_2024_20260809_v1/canonical120_features.parquet",
            "candidates": ROOT / "data_perp/artifacts/strict_r3_schema_v2_target_free_long_2024_20260809_v1/target_free_candidate_population.parquet",
            "labels": ROOT / "data_perp/artifacts/strict_r3_schema_v2_exact_tp6_r3_long_2024_20260809_v1",
        }
    return {
        "features": ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v5/features/canonical120_features.parquet",
        "candidates": ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v1/short_target_free_candidate_population.parquet",
        "labels": ROOT / "data_perp/artifacts/strict_r3_short_target_labels_2024_20260820_v1",
    }


def _feature_fields(side: str) -> list[str]:
    payload = json.loads(FEATURE_CONTRACT.read_text())
    fields = [str(value) for value in payload["base_fields_by_side"][side]]
    if len(fields) != 120 or len(set(fields)) != len(fields):
        raise ValueError(f"{side} feature contract must have 120 unique fields")
    return fields


def _selected_feature_fields(
    path: Path | None, frozen_fields: list[str],
) -> tuple[list[str], dict[str, Any] | None]:
    """Read an immutable training-only subset without widening the base pool."""
    if path is None:
        return frozen_fields, None
    payload = json.loads(path.read_text())
    selected = payload.get("selected_features", payload) if isinstance(payload, dict) else payload
    if not isinstance(selected, list):
        raise ValueError("selected-feature contract must be a list or contain selected_features")
    fields = [str(value) for value in selected]
    if len(fields) < 4 or len(fields) != len(set(fields)):
        raise ValueError("selected-feature contract must contain at least four unique fields")
    extras = [field for field in fields if field not in frozen_fields]
    if extras:
        raise ValueError(f"selected-feature contract widens frozen base pool: {extras[:8]}")
    return fields, {
        "path": str(path.resolve()), "sha256": _sha256(path),
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
    }


def _load_candidates(path: Path, side: str) -> pd.DataFrame:
    fields = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "entry_executable", "eligibility_reason",
    ]
    frame = pd.read_parquet(path, columns=fields)
    for column in ("__ts__", "__decision_ts__"):
        frame[column] = _utc(frame[column])
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).str.lower().eq(side).all():
        raise ValueError(f"{side} target-free candidate identities are invalid")
    if frame.entry_executable.isna().any():
        raise ValueError("target-free candidate execution flag is null")
    return frame


def _load_features(path: Path, fields: list[str], candidates: pd.DataFrame, side: str) -> pd.DataFrame:
    schema = set(pd.read_parquet(path, columns=None).columns)
    missing = set(fields).difference(schema)
    if missing:
        raise ValueError(f"feature panel misses frozen fields: {sorted(missing)[:10]}")
    if "candidate_id" in schema:
        columns = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", *fields]
        # Restrict the physical read to the point-in-time population requested
        # by the caller.  This avoids an inner training-only selector silently
        # reading later feature rows merely because they share one parquet.
        lower = candidates["__ts__"].min()
        upper = candidates["__ts__"].max() + pd.Timedelta(hours=1)
        frame = pd.read_parquet(
            path, columns=columns,
            filters=[("__ts__", ">=", lower), ("__ts__", "<", upper)],
        )
        frame = frame.merge(
            candidates.loc[:, [
                "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
                "entry_executable", "eligibility_reason",
            ]],
            on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"],
            how="left",
            validate="one_to_one",
        )
        if frame.entry_executable.isna().any():
            raise AssertionError("feature panel is not identical to target-free candidates")
    else:
        # The 2024 long panel predates target-free identity attachment.  Bind
        # it by the candidate's decision-time keys; labels/outcomes are not
        # present in either input.
        frame = pd.read_parquet(
            path,
            columns=["__ts__", "__symbol__", *fields],
            filters=[("__ts__", ">=", TRAIN_START), ("__ts__", "<", OOS_END)],
        )
        frame["__ts__"] = _utc(frame["__ts__"])
        if frame.duplicated(["__ts__", "__symbol__"]).any():
            raise ValueError("legacy long feature keys are not unique")
        frame = candidates.merge(frame, on=["__ts__", "__symbol__"], how="left", validate="one_to_one")
    for column in ("__ts__", "__decision_ts__"):
        frame[column] = _utc(frame[column])
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).str.lower().eq(side).all():
        raise ValueError(f"{side} feature identities are invalid")
    return frame


def _load_labels(root: Path, side: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in pd.date_range(TRAIN_START, OOS_END, freq="MS", inclusive="left"):
        path = root / "parts" / f"month={month:%Y-%m}" / f"side={side}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path))
    frame = pd.concat(parts, ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[column] = _utc(frame[column])
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).str.lower().eq(side).all():
        raise ValueError(f"{side} H12 labels are invalid")
    return frame


def _valid_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["label_valid"].astype("boolean").fillna(False).astype(bool)
        & ~frame["target_invalid"].astype("boolean").fillna(True).astype(bool)
    )


def _r3_target(frame: pd.DataFrame) -> pd.Series:
    valid = _valid_label(frame)
    event = pd.to_numeric(frame["t2_tp6_sl4_event"], errors="coerce")
    clear = pd.to_numeric(frame["robust_clear_event_b25"], errors="coerce").eq(1.0)
    target = pd.Series(np.nan, index=frame.index, dtype="float64")
    target.loc[valid] = 1.0
    target.loc[valid & event.eq(1.0)] = 0.0
    target.loc[valid & clear] = 2.0
    return target


def _ordinal_target(frame: pd.DataFrame, edges: tuple[float, float, float]) -> pd.Series:
    lower, middle, upper = edges
    valid = _valid_label(frame)
    net = pd.to_numeric(frame["t4_tp6_sl4_net_bps"], errors="coerce")
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    result.loc[valid & net.le(lower)] = 0.0
    result.loc[valid & net.gt(lower) & net.le(middle)] = 1.0
    result.loc[valid & net.gt(middle) & net.le(upper)] = 2.0
    result.loc[valid & net.gt(upper)] = 3.0
    if result.loc[valid].isna().any():
        raise AssertionError("valid economic row did not receive an ordinal class")
    return result


def _matrix(frame: pd.DataFrame, fields: list[str], medians: pd.Series) -> pd.DataFrame:
    values = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(medians)
    if values.isna().any().any():
        raise AssertionError("training-only median imputation left non-finite features")
    return values.astype("float32")


def _coverage_fields(frame: pd.DataFrame, fields: list[str]) -> tuple[list[str], pd.Series]:
    population = frame.loc[frame.entry_executable.astype(bool), fields]
    coverage = population.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean()
    kept = [field for field in fields if float(coverage[field]) >= 0.90]
    return kept, coverage


def _training_weights(
    train: pd.DataFrame,
    target: pd.Series,
    spec: TargetSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.ones(len(train), dtype=np.float64)
    if spec.family != "ordinal" or spec.weight_mode == "uniform":
        return values, {"mode": spec.weight_mode, "min": 1.0, "mean": 1.0, "max": 1.0}
    if spec.edges is None:
        raise ValueError("ordinal weight requires edges")
    net = pd.to_numeric(train["t4_tp6_sl4_net_bps"], errors="coerce").to_numpy(float)
    edges = np.asarray(spec.edges, dtype=float)
    # The distance is entirely a training-label property.  It never becomes an
    # OOS/inference feature or alters an OOS rank distribution.
    distance = np.min(np.abs(net[:, None] - edges[None, :]), axis=1)
    certainty = 0.25 + 0.75 * (1.0 / (1.0 + np.exp(-np.clip(distance / 50.0, -35.0, 35.0))))
    if spec.weight_mode == "boundary_certainty":
        values = certainty
    else:
        classes = target.astype(int).to_numpy()
        counts = np.bincount(classes, minlength=4).astype(float)
        class_weight = np.sqrt(len(classes) / np.maximum(4.0 * counts, 1.0))
        class_weight /= np.mean(class_weight[classes])
        if spec.weight_mode == "mild_class":
            values = class_weight[classes]
        elif spec.weight_mode == "hybrid":
            values = certainty * class_weight[classes]
        else:
            raise ValueError(spec.weight_mode)
    values = np.clip(values, 0.25, 4.0)
    values /= float(np.mean(values))
    return values.astype(np.float32), {
        "mode": spec.weight_mode,
        "min": float(values.min()), "mean": float(values.mean()), "max": float(values.max()),
        "effective_rows": float((values.sum() ** 2) / np.square(values).sum()),
    }


def _ordinal_centres(train: pd.DataFrame, target: pd.Series) -> np.ndarray:
    """Training-only robust class values for a comparable economic score."""
    net = pd.to_numeric(train["t4_tp6_sl4_net_bps"], errors="coerce")
    values: list[float] = []
    for label in range(4):
        local = net.loc[target.eq(label)]
        # The ordinal target should not turn one enormous path return into a
        # base-score scale.  Medians are robust and strictly train-only.
        values.append(float(local.median()))
    centres = np.asarray(values, dtype=np.float64)
    if not np.isfinite(centres).all():
        raise ValueError("ordinal target has an unsupported training class")
    return np.maximum.accumulate(centres)


def _fit_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fields: list[str],
    medians: pd.Series,
    spec: TargetSpec,
) -> tuple[np.ndarray, np.ndarray, pd.Series, dict[str, Any]]:
    target = _r3_target(train) if spec.family == "r3" else _ordinal_target(train, spec.edges or (0.0, 0.0, 0.0))
    if target.isna().any():
        raise AssertionError("fit population includes invalid target")
    classes = 3 if spec.family == "r3" else 4
    params = dict(FROZEN_BASE_PARAMS)
    params.update(objective="multiclass", num_class=classes)
    weights, weight_audit = _training_weights(train, target, spec)
    model = lgb.LGBMClassifier(**params)
    model.fit(_matrix(train, fields, medians), target.astype(int).to_numpy(), sample_weight=weights)
    probabilities = np.asarray(model.predict_proba(_matrix(test, fields, medians)), dtype=np.float32)
    if probabilities.shape[1] != classes:
        raise AssertionError("classifier did not retain all declared target classes")
    if spec.family == "r3":
        score = probabilities[:, 2] - 0.5 * probabilities[:, 0]
        centres = None
    else:
        centres = _ordinal_centres(train, target)
        score = probabilities @ centres.astype(np.float32)
    return score.astype(np.float32), probabilities, target, {
        "model": model,
        "weight_audit": weight_audit,
        "ordinal_train_class_net_medians_bps": None if centres is None else centres.tolist(),
    }


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    return float(left.loc[valid].corr(right.loc[valid], method="spearman")) if int(valid.sum()) >= 2 else float("nan")


def _metric_rows(frame: pd.DataFrame, spec: TargetSpec, scope: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = _valid_label(frame)
    resolved = frame.loc[valid].copy()
    target = _r3_target(resolved) if spec.family == "r3" else _ordinal_target(resolved, spec.edges or (0.0, 0.0, 0.0))
    pcols = [f"p{index}" for index in range(3 if spec.family == "r3" else 4)]
    probabilities = resolved[pcols].to_numpy(float)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    aggregate = {
        "spec": spec.name, "family": spec.family, "weight_mode": spec.weight_mode,
        "scope": scope, "scored_executable_rows": int(len(frame)), "resolved_rows": int(len(resolved)),
        "resolved_fraction": float(len(resolved) / max(len(frame), 1)),
        "target_log_loss": float(log_loss(target.astype(int), probabilities, labels=list(range(probabilities.shape[1])))),
        "macro_f1": float(f1_score(target.astype(int), probabilities.argmax(axis=1), average="macro")),
        "score_target_spearman": _spearman(resolved.score, target),
        "score_net_bps_spearman": _spearman(resolved.score, pd.to_numeric(resolved.t4_tp6_sl4_net_bps, errors="coerce")),
    }
    rows: list[dict[str, Any]] = []
    ordered = frame.sort_values("score", ascending=False, kind="stable")
    for fraction in TAILS:
        selected = ordered.iloc[:max(1, int(math.ceil(len(ordered) * fraction)))]
        selected_resolved = selected.loc[_valid_label(selected)]
        rows.append({
            "spec": spec.name, "family": spec.family, "weight_mode": spec.weight_mode, "scope": scope,
            "tail_fraction": fraction, "tail_rows_requested": int(len(selected)),
            "tail_rows_resolved": int(len(selected_resolved)),
            "tail_label_coverage": float(len(selected_resolved) / max(len(selected), 1)),
            "mean_score": float(selected.score.mean()),
            "mean_h12_gross_bps": float(pd.to_numeric(selected_resolved.t4_tp6_sl4_gross_bps, errors="coerce").mean()),
            "mean_h12_net_bps": float(pd.to_numeric(selected_resolved.t4_tp6_sl4_net_bps, errors="coerce").mean()),
            "median_h12_net_bps": float(pd.to_numeric(selected_resolved.t4_tp6_sl4_net_bps, errors="coerce").median()),
        })
    return aggregate, rows


def _short_policy_outcomes(
    selected: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate frozen, side-correct exact-1m SimplePolicy tails after ranking.

    This fixed diagnostic uses the legacy symmetric parent geometry (SL 3 ATR,
    trailing activation 0.5 ATR, giveback 0.25 ATR), a H12 timeout and cost
    once.  It deliberately has no short HPO authority.
    """
    if selected.empty:
        return pd.DataFrame(), {}
    policy = Exact1mPolicyParams(sl_mult=3.0, trailing_activation_mult=0.5, fixed_trailing_gap_mult=0.25)
    contract = Exact1mExecutionContract(entry_delay_minutes=0)
    median_atr = float(np.nanmedian(pd.to_numeric(train.atr_1h, errors="coerce") / pd.to_numeric(train.tp6_sl4_entry_price, errors="coerce")))
    if not np.isfinite(median_atr) or median_atr <= 0.0:
        raise ValueError("training-only short ATR reference is invalid")
    pieces: list[pd.DataFrame] = []
    horizon = int(contract.horizon_minutes)
    offset = np.arange(horizon, dtype=np.int64)
    for symbol, group in selected.groupby("__symbol__", sort=True):
        local = group.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        minute = _minute_path_pruned(
            MINUTE_ROOT,
            _packb_to_kraken_symbol(str(symbol)),
            pd.Timestamp(local.__decision_ts__.min()),
            pd.Timestamp(local.__decision_ts__.max()) + pd.Timedelta(minutes=horizon),
        )
        starts = minute.index.get_indexer(pd.DatetimeIndex(local.__decision_ts__)).astype(np.int64)
        source = minute.loc[:, ["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        high, low, close = (source[column].to_numpy(np.float64) for column in ("high", "low", "close"))
        valid = (starts >= 0) & (starts + horizon <= len(source))
        for chunk_start in range(0, len(local), 512):
            take = np.arange(chunk_start, min(chunk_start + 512, len(local)))
            take = take[valid[take]]
            if not len(take):
                continue
            positions = starts[take, None] + offset[None, :]
            replay = simulate_exact_1m_parent_policy(
                entry=pd.to_numeric(local.loc[take, "tp6_sl4_entry_price"], errors="coerce").to_numpy(float),
                atr=pd.to_numeric(local.loc[take, "atr_1h"], errors="coerce").to_numpy(float),
                highs=high[positions], lows=low[positions], closes=close[positions],
                entry_timestamps=pd.DatetimeIndex(local.loc[take, "__decision_ts__"]),
                params=policy, contract=contract, median_atr_fraction=median_atr, side="short",
            )
            output = local.loc[take, ["candidate_id", "__ts__", "__decision_ts__", "__symbol__"]].copy()
            output["policy_path_valid"] = np.asarray(replay["path_valid"], dtype=bool)
            output["policy_gross_bps"] = np.asarray(replay["gross_bps"], dtype=np.float32)
            output["policy_net_bps"] = np.asarray(replay["net_bps"], dtype=np.float32)
            output["policy_exit_minute"] = np.asarray(replay["exit_bar"], dtype=np.int16)
            output["policy_exit_reason"] = np.asarray(replay["exit_reason"], dtype=object)
            pieces.append(output)
    output = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not output.empty and output.candidate_id.duplicated().any():
        raise AssertionError("short policy replay duplicated candidate identities")
    return output, {
        "schema": "strict_r3_short_symmetric_exact1m_policy_diagnostic_v1",
        "side": "short", "entry": "exact decision-time one-minute open", "timeout_minutes": horizon,
        "cost_bps_once": 100.0, "params": policy.to_dict(),
        "median_atr_fraction_training_only": median_atr,
    }


def _policy_metrics(predictions: dict[str, pd.DataFrame], policy: pd.DataFrame, specs: tuple[TargetSpec, ...]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    by_id = policy.set_index("candidate_id") if not policy.empty else pd.DataFrame()
    for spec in specs:
        ordered = predictions[spec.name].sort_values("score", ascending=False, kind="stable")
        for fraction in POLICY_TAILS:
            selected = ordered.iloc[:max(1, int(math.ceil(len(ordered) * fraction)))]
            joined = selected.loc[:, ["candidate_id"]].join(by_id, on="candidate_id", how="left") if not policy.empty else pd.DataFrame()
            valid = joined.loc[joined.policy_path_valid.fillna(False).astype(bool)] if not joined.empty else joined
            records.append({
                "spec": spec.name, "weight_mode": spec.weight_mode, "tail_fraction": fraction,
                "tail_rows_requested": int(len(selected)), "policy_rows_resolved": int(len(valid)),
                "policy_label_coverage": float(len(valid) / max(len(selected), 1)),
                "mean_policy_gross_bps": float(pd.to_numeric(valid.policy_gross_bps, errors="coerce").mean()) if len(valid) else float("nan"),
                "mean_policy_net_bps": float(pd.to_numeric(valid.policy_net_bps, errors="coerce").mean()) if len(valid) else float("nan"),
                "median_policy_net_bps": float(pd.to_numeric(valid.policy_net_bps, errors="coerce").median()) if len(valid) else float("nan"),
            })
    return pd.DataFrame(records)


def run(*, side: str, out: Path, features_path: Path | None = None, labels_root: Path | None = None, candidates_path: Path | None = None, selected_features_path: Path | None = None, only_spec: str | None = None) -> Path:
    side = str(side).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    defaults = _side_paths(side)
    features_path = (features_path or defaults["features"]).resolve()
    labels_root = (labels_root or defaults["labels"]).resolve()
    candidates_path = (candidates_path or defaults["candidates"]).resolve()
    out.mkdir(parents=True)
    frozen_fields = _feature_fields(side)
    fields, selected_feature_contract = _selected_feature_fields(selected_features_path, frozen_fields)
    candidates = _load_candidates(candidates_path, side)
    # This research split never consumes later point-in-time candidates.  The
    # early restriction materially reduces the long legacy feature merge's
    # memory footprint without filtering on labels or outcomes.
    candidates = candidates.loc[
        candidates.__ts__.ge(TRAIN_START) & candidates.__ts__.lt(OOS_END)
    ].copy()
    # Load the complete frozen source contract; a selected subset is a
    # traceable projection, never a separate or outcome-conditioned feature
    # materialization route.
    features = _load_features(features_path, frozen_fields, candidates, side)
    labels = _load_labels(labels_root, side)
    ledger = features.merge(labels, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    if len(ledger) != len(features):
        raise AssertionError("target-free feature/label merge changed candidate identities")
    ledger["entry_executable"] = ledger.entry_executable.astype(bool)
    train_population = ledger.loc[ledger.__ts__.ge(TRAIN_START) & ledger.__ts__.lt(OOS_START)]
    kept, coverage = _coverage_fields(train_population, frozen_fields)
    if set(fields).difference(kept):
        raise ValueError("selected feature contract fails target-free executable coverage: " + str({name: float(coverage[name]) for name in fields if name not in kept}))
    train = ledger.loc[
        ledger.__ts__.ge(TRAIN_START) & ledger.__ts__.lt(OOS_START) & ledger.entry_executable
        & _valid_label(ledger) & ledger.__label_available_at__.lt(OOS_START)
    ].copy()
    test = ledger.loc[ledger.__ts__.ge(OOS_START) & ledger.__ts__.lt(OOS_END) & ledger.entry_executable].copy()
    if train.empty or test.empty:
        raise RuntimeError("strict chronological 3m/3m population is empty")
    medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    if medians.isna().any():
        raise AssertionError("a frozen feature lacks a training-only median")
    specs = COMMON_SPECS + (SHORT_WEIGHT_SPECS if side == "short" else ())
    if only_spec is not None:
        specs = tuple(spec for spec in specs if spec.name == str(only_spec))
        if len(specs) != 1:
            raise ValueError(f"unknown side-{side} spec: {only_spec}")
    summary: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    arm_audits: dict[str, Any] = {}
    for spec in specs:
        print(f"fitting {side} {spec.name}", flush=True)
        score, probabilities, target, fitted = _fit_arm(train, test, fields, medians, spec)
        prediction = test.loc[:, [
            "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "label_valid", "target_invalid", "invalid_reason",
            "tp6_sl4_entry_price", "atr_1h", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "t2_tp6_sl4_event", "robust_clear_event_b25",
        ]].copy()
        prediction["score"] = score
        for index in range(probabilities.shape[1]):
            prediction[f"p{index}"] = probabilities[:, index]
        prediction.to_parquet(out / f"oos_predictions_{spec.name}.parquet", index=False, compression="zstd")
        fitted["model"].booster_.save_model(str(out / f"model_{spec.name}.txt"))
        total, total_tails = _metric_rows(prediction, spec, "2024-04_to_2024-06")
        summary.append(total); tails.extend(total_tails)
        scoped = prediction.assign(month=prediction.__ts__.dt.strftime("%Y-%m"))
        for month, group in scoped.groupby("month", sort=True):
            row, rows = _metric_rows(group.drop(columns="month"), spec, str(month))
            summary.append(row); tails.extend(rows)
        predictions[spec.name] = prediction
        arm_audits[spec.name] = {
            "target_train_class_counts": {str(key): int(value) for key, value in target.value_counts().sort_index().items()},
            "weight": fitted["weight_audit"], "ordinal_train_class_net_medians_bps": fitted["ordinal_train_class_net_medians_bps"],
        }
        del fitted, probabilities, score
        gc.collect()
    pd.DataFrame(summary).to_parquet(out / "metrics_by_scope.parquet", index=False, compression="zstd")
    pd.DataFrame(tails).to_parquet(out / "metrics_by_scope_tail.parquet", index=False, compression="zstd")
    policy_audit: dict[str, Any] | None = None
    if side == "short":
        # Selection is a pure union of model scores.  Only after all score
        # selections are frozen do we read future one-minute bars.
        union_ids = pd.concat(
            [
                frame.nlargest(
                    max(1, int(math.ceil(len(frame) * max(POLICY_TAILS)))),
                    "score",
                )["candidate_id"]
                for frame in predictions.values()
            ],
            ignore_index=True,
        ).drop_duplicates()
        selected = test.loc[test.candidate_id.isin(set(union_ids))].copy()
        policy, policy_audit = _short_policy_outcomes(selected, train)
        policy.to_parquet(out / "short_symmetric_policy_selected_outcomes.parquet", index=False, compression="zstd")
        _policy_metrics(predictions, policy, specs).to_parquet(out / "policy_metrics_by_tail.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_ordinal_base_target_ablation_v1",
        "status": "complete", "side": side,
        "train_decision_window": f"[{TRAIN_START.isoformat()}, {OOS_START.isoformat()})",
        "strict_label_availability_gate": f"label_available_at < {OOS_START.isoformat()}",
        "oos_decision_window": f"[{OOS_START.isoformat()}, {OOS_END.isoformat()})",
        "entry": "signal close + one hour; exact decision-minute open", "label_horizon": "12 hours",
        "h12_geometry": "TP +6 ATR / SL -4 ATR; adverse wins same-minute tie", "cost_bps_once": 100.0,
        "base_parameters": FROZEN_BASE_PARAMS, "feature_contract": f"base_fields_by_side.{side}",
        "frozen_feature_pool_count": len(frozen_fields), "feature_count": len(fields), "selected_features": fields,
        "selected_feature_contract": selected_feature_contract,
        "feature_coverage_gate": ">=90% on target-free entry-executable training candidates only",
        "feature_coverage": {field: float(coverage[field]) for field in fields}, "training_rows": int(len(train)),
        "oos_scored_executable_rows": int(len(test)), "oos_h12_resolved_rows": int(_valid_label(test).sum()),
        "specifications": [asdict(spec) for spec in specs], "arm_audits": arm_audits,
        "short_policy_diagnostic": policy_audit,
        "features_sha256": _sha256(features_path), "candidates_sha256": _sha256(candidates_path),
        "labels_manifest_sha256": _sha256(labels_root / "run_manifest.json"), "feature_contract_sha256": _sha256(FEATURE_CONTRACT),
        "selection_note": "April-June is held target-selection evidence only. No arm is promoted and no policy HPO occurred.",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--selected-features-json", type=Path, default=None)
    parser.add_argument("--only-spec", type=str, default=None, help="debug/recovery: run exactly one predeclared arm")
    args = parser.parse_args()
    print(run(side=args.side, out=args.out, features_path=args.features, labels_root=args.labels, candidates_path=args.candidates, selected_features_path=args.selected_features_json, only_spec=args.only_spec))


if __name__ == "__main__":
    main()
