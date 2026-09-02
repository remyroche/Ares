#!/usr/bin/env python3
"""Fit strict-prequential source-aligned C0/C1 Base-EV mapper vintages.

This is a *new* mapper contract for the regenerated Router50/F72 Base score
family.  It deliberately does not pretend that Base/Router/Under coordinates
have the names or semantics of the deleted BCF/current stack.  The familiar
``both -> C0-only -> C1-only`` admission-tier selector is reused only after
each family has independently emitted target-free expected-EV coordinates.

C0 uses strictly OOF upstream geometry.  C1 adds only candidate-time C1-LVA
snapshots.  Exact rich-policy outcomes are attached after target-free inputs
are constructed and are eligible for fitting only when resolved before the
held month or the causal prior-21-day shift day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_c0_c1_agreement_tier import (  # noqa: E402
    UNPAIRED_ORDER_C0_THEN_C1,
    select_c0_c1_agreement_tiers,
)
from extreme_price_movements.inference.p8u_c1_mc1_inference_package import (  # noqa: E402
    C1_SNAPSHOT_FEATURES,
)


IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
C0_FEATURES = (
    "base_rank_ts", "router_primary_rank", "base_minus_router_rank",
    "under_rank_ts", "under_available",
)
C1_AVAILABILITY = "sr_snapshot_available"
C1_FEATURES = (*C0_FEATURES, *C1_SNAPSHOT_FEATURES, C1_AVAILABILITY)
SEED = 1729
SHIFT_DAYS = 21
SHIFT_TRIM = 0.10
MODEL_PARAMS = {
    "max_depth": 2, "max_iter": 80, "learning_rate": 0.04,
    "l2_regularization": 20.0, "min_samples_leaf": 100,
    "random_state": SEED,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(_sha(child).encode())
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _months(start: object, end: object) -> tuple[pd.Timestamp, ...]:
    start, end = _utc(start).normalize().replace(day=1), _utc(end).normalize().replace(day=1)
    if end <= start:
        raise ValueError("held end must follow held start")
    return tuple(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS", tz="UTC"))


def _robust_mean(values: Iterable[float]) -> float:
    x = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    if not len(x):
        return float("nan")
    k = int(math.floor(len(x) * SHIFT_TRIM))
    if k and len(x) > 2 * k:
        x = x[k: len(x) - k]
    return float(x.mean())


def _score_bands(frame: pd.DataFrame) -> np.ndarray:
    """Ten deterministic timestamp-local bands of the Base rank coordinate."""
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "base_rank_ts"]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["base_rank_ts"] = pd.to_numeric(work["base_rank_ts"], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "base_rank_ts", "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["__band__"] = np.minimum(9, (10.0 * (rank + .5) / count).astype(np.int8))
    return work.sort_values("__row__", kind="stable")["__band__"].to_numpy(np.int8)


def _matrix(frame: pd.DataFrame, fields: tuple[str, ...], medians: np.ndarray) -> np.ndarray:
    missing = set(fields).difference(frame.columns)
    if missing:
        raise KeyError(f"successor C0/C1 mapper input missing {sorted(missing)}")
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    return values.fillna(pd.Series(medians, index=fields)).to_numpy(np.float64)


def _day_balanced(train: pd.DataFrame) -> pd.DataFrame:
    work = train.copy()
    work["__day__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise").dt.normalize()
    selected: list[pd.DataFrame] = []
    for _day, group in work.groupby("__day__", sort=True):
        group = group.sort_values(["__decision_ts__", "base_rank_ts", "candidate_id"], ascending=[True, False, True], kind="stable")
        head, tail = group.head(50), group.iloc[50:]
        if len(tail):
            tail = tail.sample(min(250, len(tail)), random_state=SEED)
        selected.append(pd.concat((head, tail), ignore_index=False))
    if not selected:
        return work.iloc[0:0].copy()
    output = pd.concat(selected, ignore_index=True)
    if len(output) > 50_000:
        output = output.sample(50_000, random_state=SEED).sort_values(
            ["policy_label_available_ts", "candidate_id"], kind="stable"
        )
    return output.drop(columns="__day__", errors="ignore")


def _curve(train: pd.DataFrame) -> np.ndarray:
    full = train.copy()
    full["__band__"] = _score_bands(full)
    y = pd.to_numeric(full["policy_net_bps"], errors="coerce")
    global_mean = _robust_mean(y)
    curve = np.full(10, global_mean, dtype=float)
    for band, group in full.groupby("__band__", sort=True):
        values = pd.to_numeric(group["policy_net_bps"], errors="coerce").dropna().to_numpy(float)
        if not len(values):
            continue
        mean, deviation = float(values.mean()), max(float(values.std(ddof=0)), 1.0)
        precision = len(values) / (deviation * deviation + 1.0)
        prior_precision = 80.0 / (250.0**2)
        curve[int(band)] = (precision * mean + prior_precision * global_mean) / (precision + prior_precision)
    return -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -curve)


@dataclass
class _Package:
    family: str
    fields: tuple[str, ...]
    medians: np.ndarray
    model: HistGradientBoostingRegressor
    curve: np.ndarray
    target_clip: tuple[float, float]
    train_start: str
    held_start: str
    fit_rows: int
    sampled_rows: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_matrix(frame, self.fields, self.medians)).astype(float)


def _fit(train: pd.DataFrame, *, family: str, fields: tuple[str, ...], start: pd.Timestamp, held: pd.Timestamp) -> _Package:
    clean = train.loc[
        train["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(train["policy_net_bps"], errors="coerce"))
    ].copy()
    if len(clean) < 5_000:
        raise RuntimeError(f"{family}/{held:%Y-%m}: insufficient valid strict-prequential training rows ({len(clean)})")
    low, high = pd.to_numeric(clean["policy_net_bps"], errors="raise").quantile([.02, .98]).to_numpy(float)
    sampled = _day_balanced(clean)
    medians = sampled.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").median().to_numpy(float)
    target = pd.to_numeric(sampled["policy_net_bps"], errors="raise").clip(low, high)
    model = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(_matrix(sampled, fields, medians), target)
    return _Package(
        family=family, fields=fields, medians=medians, model=model, curve=_curve(clean),
        target_clip=(float(low), float(high)), train_start=start.isoformat(), held_start=held.isoformat(),
        fit_rows=len(clean), sampled_rows=len(sampled),
    )


def _shift(package: _Package, history: pd.DataFrame, *, held: pd.Timestamp, held_end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in pd.date_range(held.normalize(), (held_end - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC"):
        eligible = history.loc[
            history["__decision_ts__"].ge(day - pd.Timedelta(days=SHIFT_DAYS))
            & history["__decision_ts__"].lt(day)
            & history["policy_path_valid"].fillna(False).astype(bool)
            & history["policy_label_available_ts"].lt(day)
            & np.isfinite(pd.to_numeric(history["policy_net_bps"], errors="coerce"))
        ].copy()
        bands = _score_bands(eligible) if len(eligible) else np.empty(0, dtype=np.int8)
        residual = pd.to_numeric(eligible["policy_net_bps"], errors="coerce").to_numpy(float) - package.curve[bands]
        rows.append({
            "decision_day": day, "recent_shift_bps": _robust_mean(residual), "resolved_rows": int(len(eligible)),
            "max_policy_label_available_ts": eligible["policy_label_available_ts"].max() if len(eligible) else pd.NaT,
            "window_start": day - pd.Timedelta(days=SHIFT_DAYS), "window_end_exclusive": day,
            "family": package.family,
        })
    state = pd.DataFrame(rows)
    max_label = pd.to_datetime(state["max_policy_label_available_ts"], utc=True, errors="coerce")
    if not ((max_label.lt(state["decision_day"])) | max_label.isna()).all():
        raise AssertionError("prior-21d mapper shift consumed unresolved outcome")
    return state


def _read_inputs(source: Path, snapshots: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_free = pd.read_parquet(source / "target_free_upstream_scores.parquet").copy()
    labels = pd.read_parquet(source / "policy_attached_replay_panel.parquet").copy()
    policy_columns = [column for column in labels.columns if column.startswith("policy_")]
    if set(policy_columns).intersection(target_free.columns):
        raise AssertionError("target-free source carries policy output")
    for frame in (target_free, labels):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    snapshot_parts = [pd.read_parquet(path).copy() for path in snapshots]
    if not snapshot_parts:
        raise ValueError("at least one C1 snapshot panel is required")
    snapshot = pd.concat(snapshot_parts, ignore_index=True, sort=False)
    snapshot["candidate_id"] = snapshot["candidate_id"].astype(str)
    snapshot["snapshot_ts"] = pd.to_datetime(snapshot["snapshot_ts"], utc=True, errors="raise")
    required_snapshot = {"candidate_id", "snapshot_ts", *C1_SNAPSHOT_FEATURES, C1_AVAILABILITY}
    missing = required_snapshot.difference(snapshot.columns)
    if missing:
        raise KeyError(f"C1 snapshot ledger misses {sorted(missing)}")
    if snapshot.duplicated(["candidate_id", "snapshot_ts"]).any():
        raise AssertionError("C1 snapshot ledger duplicates candidate-time identities")
    target_free = target_free.merge(
        snapshot.loc[:, ["candidate_id", "snapshot_ts", *C1_SNAPSHOT_FEATURES, C1_AVAILABILITY]],
        left_on=["candidate_id", "__decision_ts__"], right_on=["candidate_id", "snapshot_ts"], how="left", validate="one_to_one",
    ).drop(columns="snapshot_ts")
    target_free[C1_AVAILABILITY] = target_free[C1_AVAILABILITY].fillna(0).astype(np.int8)
    if len(target_free) != len(labels):
        raise AssertionError("target-free and label panels differ in identity count")
    labels = labels.loc[:, [*IDENTITY, *policy_columns]]
    full = target_free.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(full) != len(target_free):
        raise AssertionError("policy join changed target-free source membership")
    return target_free, full


def _fit_family(full: pd.DataFrame, *, family: str, fields: tuple[str, ...], held_months: tuple[pd.Timestamp, ...], package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    predictions: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    packages: list[dict[str, object]] = []
    for held in held_months:
        held_end, train_start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=6)
        train = full.loc[
            full["__decision_ts__"].ge(train_start) & full["__decision_ts__"].lt(held)
            & full["policy_label_available_ts"].lt(held)
        ].copy()
        test = full.loc[full["__decision_ts__"].ge(held) & full["__decision_ts__"].lt(held_end)].copy()
        if test.empty:
            raise RuntimeError(f"{family}/{held:%Y-%m}: no target-free held rows")
        package = _fit(train, family=family, fields=fields, start=train_start, held=held)
        state = _shift(package, full, held=held, held_end=held_end)
        output = test.loc[:, list(IDENTITY)].copy()
        output["static_expected_bps"] = package.predict(test)
        output["score_band_curve_bps"] = package.curve[_score_bands(test)]
        daily = state.set_index("decision_day")["recent_shift_bps"]
        output["recent_shift_bps"] = output["__decision_ts__"].dt.normalize().map(daily).fillna(0.0).to_numpy(float)
        output["mc1_expected_bps"] = output["static_expected_bps"] + output["recent_shift_bps"]
        output["mapper_family"] = family
        predictions.append(output)
        folder = package_root / family / f"{held:%Y-%m}"
        folder.mkdir(parents=True, exist_ok=False)
        joblib.dump(package, folder / "package.joblib", compress=3)
        _write_json(folder / "package_manifest.json", {
            "schema": "p8u_successor_c0_c1_mapper_v1", "family": family, "feature_order": list(fields),
            "train_start": train_start.isoformat(), "train_end_exclusive": held.isoformat(),
            "held_start": held.isoformat(), "held_end_exclusive": held_end.isoformat(), "train_months": 6,
            "model_parameters": MODEL_PARAMS, "fit_rows": package.fit_rows, "sampled_rows": package.sampled_rows,
            "target": "exact_1m_rich_policy_net_bps clipped p02-p98; 100bps cost embedded once",
            "target_clip": list(package.target_clip), "score_band_coordinate": "base_rank_ts timestamp-local decile",
            "shift": "prior 21d, 10% trimmed residual versus package score-band curve, resolved labels only",
        })
        _write_json(folder / "band_curve.json", {"expected_bps": package.curve.tolist(), "band_order": list(range(10))})
        state.to_parquet(folder / "prior21d_shift_state.parquet", index=False, compression="zstd")
        packages.append({"family": family, "month": f"{held:%Y-%m}", "path": str(folder.relative_to(package_root)), "sha256": _tree_hash(folder)})
        audit.append({
            "family": family, "held_month": f"{held:%Y-%m}", "train_start": train_start.isoformat(),
            "train_rows": int(len(train)), "valid_train_rows": package.fit_rows, "held_rows": int(len(test)),
            "c1_available_train": float(train[C1_AVAILABILITY].mean()), "c1_available_held": float(test[C1_AVAILABILITY].mean()),
        })
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit), packages


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    temporary = output.with_name(f".{output.name}.build-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        source = args.source.resolve()
        snapshots = [path.resolve() for path in args.c1_snapshots]
        target_free, full = _read_inputs(source, snapshots)
        held = _months(args.held_start, args.held_end)
        package_root = temporary / "packages"
        c0_pred, c0_audit, c0_packages = _fit_family(full, family="c0_base_geometry", fields=C0_FEATURES, held_months=held, package_root=package_root)
        c1_pred, c1_audit, c1_packages = _fit_family(full, family="c1_lva_geometry", fields=C1_FEATURES, held_months=held, package_root=package_root)
        for name, frame in (("c0", c0_pred), ("c1", c1_pred)):
            frame.to_parquet(temporary / f"predictions_{name}_target_free.parquet", index=False, compression="zstd")
        c0_coord = c0_pred.loc[:, list(IDENTITY) + ["mc1_expected_bps"]].copy()
        c1_coord = c1_pred.loc[:, list(IDENTITY) + ["mc1_expected_bps"]].copy()
        # The regenerated upstream has one authoritative Base score family.
        # Each mapper therefore emits the same expected-EV coordinate for its
        # BCF/current slots; the only independent family comparison is C0
        # core geometry versus C1 core + LVA geometry.
        def _selector_input(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.rename(columns={"mc1_expected_bps": "bcf_mc1_expected_bps"}).copy()
            out["current_mc1_expected_bps"] = out["bcf_mc1_expected_bps"]
            out["auction_priority_bps"] = out["bcf_mc1_expected_bps"]
            return out
        selected = select_c0_c1_agreement_tiers(
            c0_scores=_selector_input(c0_coord), c1_scores=_selector_input(c1_coord),
            admission_floor_bps=float(args.admission_floor_bps), unpaired_order=UNPAIRED_ORDER_C0_THEN_C1,
        )
        selected.to_parquet(temporary / "agreement_tier_target_free_predictions.parquet", index=False, compression="zstd")
        replay = selected.merge(
            full.loc[:, list(IDENTITY) + [column for column in full.columns if column.startswith("policy_")]],
            on=list(IDENTITY), how="left", validate="one_to_one",
        )
        replay.to_parquet(temporary / "agreement_tier_policy_replay.parquet", index=False, compression="zstd")
        audit = pd.concat((c0_audit, c1_audit), ignore_index=True)
        audit.to_parquet(temporary / "fold_audit.parquet", index=False, compression="zstd")
        _write_json(temporary / "run_manifest.json", {
            "schema": "p8u_successor_c0_c1_prequential_v1", "status": "complete",
            "scope": "offline no-order source-aligned C0/C1 exact-policy mapper and agreement-tier producer",
            "source": {
                "path": str(source), "manifest_sha256": _sha(source / "run_manifest.json"),
                "target_free_panel_sha256": _sha(source / "target_free_upstream_scores.parquet"),
                "policy_replay_panel_sha256": _sha(source / "policy_attached_replay_panel.parquet"),
            },
            # ``--c1-snapshots`` is the immutable Parquet panel itself, while
            # the matching provenance receipt lives beside it.  Do not treat
            # the Parquet file as a directory when sealing this mapper bundle.
            "c1_snapshots": [
                {
                    "path": str(path), "panel_sha256": _sha(path),
                    "manifest_sha256": _sha(path.parent / "run_manifest.json"),
                }
                for path in snapshots
            ],
            "held_months": [f"{x:%Y-%m}" for x in held], "train_months": 6,
            "c0_features": list(C0_FEATURES), "c1_features": list(C1_FEATURES),
            "c0_c1_authority": "C0 core Base/Router/Under OOF geometry versus C1 C0 geometry plus causal LVA snapshots",
            "tier": "both-admitted -> C0-only -> C1-only", "admission_floor_bps": float(args.admission_floor_bps),
            "target_free_rows": int(len(target_free)), "selected_target_free_rows": int(len(selected)),
            "causality": {
                "upstream": "Router50/Base/Under score panel was target-free before mapper construction; Under has zero direct upstream rank authority",
                "c1": "candidate-time prequential LVA snapshots; unavailable snapshot remains explicit feature, never identity filter",
                "labels": "exact rich-policy labels attached only after target-free source and C1 features; each model fit uses six prior calendar months resolved before held boundary",
                "shift": "daily 21d residual shift uses only labels resolved before each decision day",
                "authority": "no portfolio, execution, exchange, or order authority",
            },
            "packages": [*c0_packages, *c1_packages],
            "outputs": {
                name: _sha(temporary / name) for name in (
                    "predictions_c0_target_free.parquet", "predictions_c1_target_free.parquet",
                    "agreement_tier_target_free_predictions.parquet",
                    "agreement_tier_policy_replay.parquet", "fold_audit.parquet",
                )
            },
        })
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--c1-snapshots", type=Path, action="append", required=True,
        help="One or more immutable, non-overlapping C1 snapshot panels.",
    )
    parser.add_argument("--held-start", required=True)
    parser.add_argument("--held-end", required=True)
    parser.add_argument("--admission-floor-bps", type=float, default=50.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if float(args.admission_floor_bps) <= 0.0:
        raise SystemExit("admission floor must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
