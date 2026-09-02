#!/usr/bin/env python3
"""Strict-reserve Stage-4 native MC1 funnel for the frozen C0/C2 finalists.

This is offline, long-only research.  It never reads a live bundle nor writes
admission, portfolio, execution, or canonical artifacts.  C0 is retained;
C2 is the only Stage-3 challenger.  Current and BCF mappers are fitted
separately and are never fed each other's score-family fields.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
C2_DATASET = ROOT / "data_perp/artifacts/strict_r3_stage4_c2_conversion_finalized_20260823_v2"
C0_SCORES = ROOT / (
    "data_perp/artifacts/strict_r3_base_recall_residual2_consensus_research_"
    "20260822_v4_control_parity_full_v2_20260822/control_rescored_target_free.parquet"
)
BCF_SCORES = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
)
LABELS = ROOT / (
    "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_"
    "2026oos_20260822_v1/outcome_joined_recall_ledger.parquet"
)
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_stage4_mc1_native_funnel_20260823_v1"

RESERVE_DAYS = 28
WINDOW_DAYS = 21
SEED = 1729
CORE = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)
ARMS = {
    "c0_m0": CORE,
    "c2_m0": CORE,
    "c2_m1_anchor": (*CORE, "base_anchor_bps"),
    "c2_m2_agreement": (*CORE, "base_anchor_bps", "stage3_c2_head_iqr", "stage3_c2_head_mad"),
}
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
MODEL_SPECS = {
    "standard": {
        "max_depth": 2, "max_rows": 50_000, "min_samples_leaf": 100,
        "l2_regularization": 20.0,
    },
    # The two predeclared M4 capacity controls.  Depth three deliberately
    # uses materially stronger support/regularisation than the d2 controls.
    "d2_100k": {
        "max_depth": 2, "max_rows": 100_000, "min_samples_leaf": 100,
        "l2_regularization": 20.0,
    },
    "d3_100k": {
        "max_depth": 3, "max_rows": 100_000, "min_samples_leaf": 250,
        "l2_regularization": 50.0,
    },
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_bands(frame: pd.DataFrame) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "final_score"]].copy()
    work["__position__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable", na_position="last",
    )
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["score_band"] = np.minimum(9, ((rank - .5) / count * 10.0).astype(np.int8))
    return work.sort_values("__position__", kind="stable")["score_band"].to_numpy(np.int8)


def _robust_mean(values: np.ndarray, trim: float = .10) -> float:
    clean = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not len(clean):
        return float("nan")
    amount = int(math.floor(len(clean) * trim))
    if amount and len(clean) > 2 * amount:
        clean = clean[amount:-amount]
    return float(clean.mean())


def _day_balanced(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, day in frame.groupby("day", sort=True):
        ordered = day.sort_values(
            ["__decision_ts__", "final_score", "candidate_id"],
            ascending=[True, False, True], kind="stable",
        ).copy()
        position = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
        top = ordered.loc[position.le(50)]
        rest = ordered.loc[position.gt(50)]
        if len(rest):
            rest = rest.sample(min(250, len(rest)), random_state=SEED)
        pieces.append(pd.concat([top, rest], ignore_index=False))
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()


def _fit_static(
    train: pd.DataFrame, features: Sequence[str], *, model_spec: str,
) -> tuple[dict[str, object], np.ndarray]:
    spec = MODEL_SPECS[model_spec]
    sample = _day_balanced(train)
    sample = sample.loc[np.isfinite(pd.to_numeric(sample["policy_net_bps"], errors="coerce"))].copy()
    medians = sample.loc[:, features].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    matrix = sample.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(medians)
    target = pd.to_numeric(sample["policy_net_bps"], errors="coerce")
    low, high = target.quantile([.02, .98]).to_numpy(float)
    target = target.clip(low, high)
    if len(matrix) > int(spec["max_rows"]):
        chosen = matrix.sample(int(spec["max_rows"]), random_state=SEED).index
        matrix, target = matrix.loc[chosen], target.loc[chosen]
    model = HistGradientBoostingRegressor(
        max_depth=int(spec["max_depth"]), max_iter=80, learning_rate=.04,
        l2_regularization=float(spec["l2_regularization"]),
        min_samples_leaf=int(spec["min_samples_leaf"]), random_state=SEED,
    ).fit(matrix, target)
    global_mean = _robust_mean(sample["policy_net_bps"].to_numpy(float))
    curve = np.full(10, global_mean, dtype=float)
    for band, group in sample.groupby("score_band", sort=True):
        values = pd.to_numeric(group["policy_net_bps"], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        mean, std, count = finite.mean(), max(finite.std(), 1.0), len(finite)
        precision = count / (std * std + 1.0)
        prior = 80.0 / (250.0**2)
        curve[int(band)] = (precision * mean + prior * global_mean) / (precision + prior)
    curve = -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -curve)
    return {
        "features": tuple(features), "medians": medians.to_numpy(float), "model": model,
        "clip_low_bps": float(low), "clip_high_bps": float(high), "training_rows": int(len(matrix)),
        "model_spec": model_spec,
    }, np.asarray(curve, dtype=float)


def _daily_shift(
    history: pd.DataFrame, *, curve: np.ndarray, dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, float]:
    values: dict[pd.Timestamp, float] = {}
    valid = history.loc[
        history["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(history["policy_net_bps"], errors="coerce"))
    ].copy()
    valid["residual"] = pd.to_numeric(valid["policy_net_bps"], errors="coerce").to_numpy(float) - curve[
        valid["score_band"].to_numpy(int)
    ]
    for day in sorted(pd.DatetimeIndex(dates).unique()):
        recent = valid.loc[
            valid["policy_label_available_ts"].lt(day)
            & valid["__decision_ts__"].ge(day - pd.Timedelta(days=WINDOW_DAYS))
        , "residual"].to_numpy(float)
        values[day] = _robust_mean(recent)
    return values


def _load_labels() -> pd.DataFrame:
    labels = pd.read_parquet(LABELS, columns=[
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    ])
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True, errors="coerce")
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy labels are not one-to-one")
    return labels


def _head_summary(frame: pd.DataFrame, heads: list[str]) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame.loc[:, heads].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    iqr = np.nanpercentile(matrix, 75, axis=1) - np.nanpercentile(matrix, 25, axis=1)
    mad = np.nanmedian(np.abs(matrix - np.nanmedian(matrix, axis=1, keepdims=True)), axis=1)
    return iqr, mad


def _load_current_c0(labels: pd.DataFrame, *, include_agreement: bool = True) -> pd.DataFrame:
    heads = []
    if include_agreement:
        heads = sorted(
            field for field in pq.ParquetFile(C0_SCORES).schema_arrow.names
            if field.startswith("conditional_head__") and field.endswith("__rank")
        )
    # The M4 C0-only control does not consume agreement fields.  Do not load
    # the broad head matrix merely to discard it: that has no semantic effect
    # and materially increases the memory footprint of the capacity check.
    columns = [*IDENTITY, "base_anchor_bps", *CORE, *heads]
    frame = pd.read_parquet(C0_SCORES, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if heads:
        frame["stage3_c2_head_iqr"], frame["stage3_c2_head_mad"] = _head_summary(frame, heads)
        frame = frame.drop(columns=heads)
    frame["score_arm"] = "c0"
    return frame.merge(labels, on="candidate_id", how="left", validate="one_to_one")


def _replace_with_c2(c0: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    current = c0.copy().set_index("candidate_id", drop=False)
    fields = [*IDENTITY, "base_anchor_bps", *CORE, "stage3_c2_head_iqr", "stage3_c2_head_mad"]
    for row in inventory.itertuples(index=False):
        raw = pd.read_parquet(row.path, columns=fields)
        timestamp = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        raw = raw.loc[(timestamp >= row.decision_start) & (timestamp < row.decision_end_exclusive)].copy()
        raw = raw.set_index("candidate_id", drop=False)
        missing = raw.index.difference(current.index)
        if len(missing):
            raise AssertionError("C2 shard contains an identity outside C0 parity population")
        for field in fields[1:]:
            current.loc[raw.index, field] = raw[field]
        current.loc[raw.index, "score_arm"] = "c2"
    return current.reset_index(drop=True)


def _load_bcf(labels: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(BCF_SCORES, columns=[*IDENTITY, *CORE])
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return frame.merge(labels, on="candidate_id", how="left", validate="one_to_one")


def _fit_score_blocks(
    *, panel: pd.DataFrame, inventory: pd.DataFrame, features: Sequence[str], family: str, arm: str,
    model_spec: str, out: Path,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    model_root = out / "models" / family / arm
    model_root.mkdir(parents=True, exist_ok=True)
    panel = panel.copy()
    panel["day"] = panel["__decision_ts__"].dt.normalize()
    panel["score_band"] = _score_bands(panel)
    for block in inventory.itertuples(index=False):
        cutoff = pd.Timestamp(block.conversion_cutoff)
        reserve = cutoff - pd.Timedelta(days=RESERVE_DAYS)
        held = panel.loc[
            panel["__decision_ts__"].ge(block.decision_start)
            & panel["__decision_ts__"].lt(block.decision_end_exclusive)
        ].copy()
        if held.empty:
            continue
        train = panel.loc[
            panel["__decision_ts__"].lt(reserve)
            & panel["policy_label_available_ts"].lt(reserve)
            & panel["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(panel["policy_net_bps"], errors="coerce"))
        ].copy()
        train = train.dropna(subset=list(features))
        if len(train) < 1_000:
            continue
        payload, curve = _fit_static(train, features, model_spec=model_spec)
        daily = _daily_shift(panel.loc[panel["__decision_ts__"].lt(block.decision_end_exclusive)], curve=curve, dates=held["day"])
        matrix = held.loc[:, features].apply(pd.to_numeric, errors="coerce")
        complete = np.isfinite(matrix.to_numpy(float)).all(axis=1)
        expected = np.full(len(held), np.nan, dtype=float)
        if complete.any():
            filled = matrix.loc[complete].fillna(pd.Series(payload["medians"], index=features))
            expected[complete] = payload["model"].predict(filled) + held.loc[complete, "day"].map(daily).to_numpy(float)
        joblib.dump({**payload, "structural_curve_bps": curve, "cutoff": cutoff, "reserve_start": reserve}, model_root / f"block={block.block}.joblib")
        output.append(pd.DataFrame({
            "candidate_id": held["candidate_id"].astype(str).to_numpy(),
            "__decision_ts__": held["__decision_ts__"].to_numpy(),
            "family": family, "arm": arm, "model_spec": model_spec, "control_block": block.block,
            "mc1_expected_bps": expected,
            "mc1_recent_global_shift_bps": held["day"].map(daily).to_numpy(float),
            "mc1_available": complete & np.isfinite(expected),
        }))
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--capacity-only", action="store_true",
        help="Run only the predeclared C0/BCF M4 100k capacity controls.",
    )
    parser.add_argument(
        "--capacity-family", choices=("current", "bcf", "both"), default="both",
        help="For capacity-only runs, materialise one family at a time to cap memory.",
    )
    parser.add_argument(
        "--capacity-spec", choices=("d2_100k", "d3_100k"),
        help="For capacity-only runs, materialise one model specification at a time.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    manifest = json.loads((C2_DATASET / "run_manifest.json").read_text())
    inventory = pd.read_parquet(C2_DATASET / "c2_target_free_shards.parquet")
    for field in ("decision_start", "decision_end_exclusive"):
        inventory[field] = pd.to_datetime(inventory[field], utc=True, errors="raise")
    if args.capacity_spec and not args.capacity_only:
        raise ValueError("--capacity-spec requires --capacity-only")
    labels = _load_labels()
    args.out_dir.mkdir(parents=True)
    need_current = not args.capacity_only or args.capacity_family in {"current", "both"}
    need_bcf = not args.capacity_only or args.capacity_family in {"bcf", "both"}
    c0 = _load_current_c0(labels, include_agreement=not args.capacity_only) if need_current else None
    # The predeclared M4 capacity check is intentionally C0-only after C2
    # failed the preceding group-ablation gate; avoid materialising C2 shards.
    c2 = None if args.capacity_only else _replace_with_c2(c0, inventory)
    bcf = _load_bcf(labels) if need_bcf else None
    outputs: list[pd.DataFrame] = []
    if args.capacity_only:
        # M4 follows the group-ablation gate: C2 lost to C0 in all three
        # chronological slices, so only retained C0 and its separately native
        # BCF partner receive additional capacity.
        specs = (args.capacity_spec,) if args.capacity_spec else ("d2_100k", "d3_100k")
        current_plan = [
            (f"c0_m0_{spec}", c0, CORE, spec) for spec in specs
        ] if args.capacity_family in {"current", "both"} else []
        bcf_plan = [
            (f"bcf_m0_{spec}", spec) for spec in specs
        ] if args.capacity_family in {"bcf", "both"} else []
    else:
        current_plan = [
            (arm, c0 if arm == "c0_m0" else c2, features, "standard")
            for arm, features in ARMS.items()
        ]
        bcf_plan = [("bcf_m0", "standard")]
    for arm, panel, features, model_spec in current_plan:
        outputs.append(_fit_score_blocks(
            panel=panel, inventory=inventory, features=features, family="current", arm=arm,
            model_spec=model_spec, out=args.out_dir,
        ))
    # BCF remains separately native and never consumes current/C2 fields.
    for arm, model_spec in bcf_plan:
        outputs.append(_fit_score_blocks(
            panel=bcf, inventory=inventory, features=CORE, family="bcf", arm=arm,
            model_spec=model_spec, out=args.out_dir,
        ))
    predictions = pd.concat(outputs, ignore_index=True)
    predictions.to_parquet(args.out_dir / "native_mc1_predictions.parquet", index=False, compression="zstd")
    run_manifest = {
        "schema": "strict_r3_stage4_native_mc1_funnel_v1",
        "scope": "offline long-only research; no live/canonical/admission/portfolio/exit artifact modified",
        "reserve_days": RESERVE_DAYS,
        "target": "canonical source-aligned frozen rich parent policy_net_bps; invalid paths excluded",
        "model_specs": MODEL_SPECS,
        "capacity_only": bool(args.capacity_only),
        "current_arms": [item[0] for item in current_plan],
        "bcf_arms": [item[0] for item in bcf_plan],
        "bcf_native": "separate BCF score family only; no current/C2 values enter it",
        "recent_shift": "same-family 21d 10%-trimmed score-band residual; labels available strictly before each day",
        "c2_dataset_manifest": str(C2_DATASET / "run_manifest.json"),
        "c2_dataset_rows": manifest["rows_selected"],
        "prediction_rows": int(len(predictions)),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "stage4_native_mc1_complete", "rows": len(predictions)}))


if __name__ == "__main__":
    main()
