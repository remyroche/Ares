#!/usr/bin/env python3
"""Materialise exact-1m short policy-conversion labels without routing on paths.

This produces the training/evaluation substrate for the short base policy-rank
funnel.  Every candidate identity is loaded first from the immutable short
label ledger.  Exact one-minute paths are opened only afterwards; an absent or
incomplete path is recorded as an invalid policy label, never as a zero PnL.

The seven outputs are one frozen local neighbourhood around the current
SimplePolicy diagnostic.  They are *labels*, not a policy HPO and not a live
execution authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_policy_contract import (  # noqa: E402
    Exact1mExecutionContract,
    Exact1mPolicyParams,
    simulate_exact_1m_parent_policy,
)
from scripts.run_strict_r3_ordinal_base_target_ablation import (  # noqa: E402
    MINUTE_ROOT,
    _minute_path_pruned,
    _packb_to_kraken_symbol,
)


SCHEMA = "strict_r3_short_policy_conversion_labels_v1"
SIDE = "short"
HORIZON_MINUTES = 720
CHUNK_ROWS = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _variants() -> dict[str, Exact1mPolicyParams]:
    """Predeclared canonical-local policy neighbourhood from the roadmap."""
    return {
        "p0_canonical": Exact1mPolicyParams(
            sl_mult=3.0, trailing_activation_mult=0.50, fixed_trailing_gap_mult=0.25,
        ),
        "p1_sl25": Exact1mPolicyParams(
            sl_mult=2.5, trailing_activation_mult=0.50, fixed_trailing_gap_mult=0.25,
        ),
        "p2_sl35": Exact1mPolicyParams(
            sl_mult=3.5, trailing_activation_mult=0.50, fixed_trailing_gap_mult=0.25,
        ),
        "p3_activation40": Exact1mPolicyParams(
            sl_mult=3.0, trailing_activation_mult=0.40, fixed_trailing_gap_mult=0.25,
        ),
        "p4_activation60": Exact1mPolicyParams(
            sl_mult=3.0, trailing_activation_mult=0.60, fixed_trailing_gap_mult=0.25,
        ),
        "p5_giveback20": Exact1mPolicyParams(
            sl_mult=3.0, trailing_activation_mult=0.50, fixed_trailing_gap_mult=0.20,
        ),
        "p6_giveback30": Exact1mPolicyParams(
            sl_mult=3.0, trailing_activation_mult=0.50, fixed_trailing_gap_mult=0.30,
        ),
    }


def _month_paths(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    return [root / "parts" / f"month={stamp:%Y-%m}" / "side=short.parquet"
            for stamp in pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left")]


def _load_month(path: Path) -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "__label_available_at__",
        "tp6_sl4_entry_price", "atr_1h", "label_valid", "target_invalid", "invalid_reason",
        "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
    ]
    frame = pd.read_parquet(path, columns=columns)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError(f"invalid short identity/side in {path}")
    return frame.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _policy_columns(keys: list[str]) -> list[str]:
    output = ["policy_input_valid", "policy_path_valid", "policy_label_available_at"]
    for key in keys:
        output += [
            f"{key}_gross_bps", f"{key}_net_bps", f"{key}_exit_minute", f"{key}_exit_reason",
        ]
    return output


def _blank_output(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "__label_available_at__",
        "label_valid", "target_invalid", "invalid_reason", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
    ]
    output = frame.loc[:, columns].copy()
    entry = pd.to_numeric(frame.tp6_sl4_entry_price, errors="coerce")
    atr = pd.to_numeric(frame.atr_1h, errors="coerce")
    output["policy_input_valid"] = (entry.gt(0.0) & atr.gt(0.0)).astype(bool)
    output["policy_path_valid"] = False
    output["policy_label_available_at"] = output["__decision_ts__"] + pd.Timedelta(minutes=HORIZON_MINUTES)
    for key in keys:
        output[f"{key}_gross_bps"] = np.float32(np.nan)
        output[f"{key}_net_bps"] = np.float32(np.nan)
        output[f"{key}_exit_minute"] = np.int16(-1)
        output[f"{key}_exit_reason"] = "invalid_exact_1m_path"
    return output


def _materialize_symbol(
    frame: pd.DataFrame,
    *, variants: dict[str, Exact1mPolicyParams], median_atr_fraction: float,
) -> pd.DataFrame:
    keys = list(variants)
    output = _blank_output(frame, keys)
    eligible = output.policy_input_valid.to_numpy(dtype=bool)
    if not eligible.any():
        return output
    local = frame.reset_index(drop=True)
    first = pd.Timestamp(local.loc[eligible, "__decision_ts__"].min())
    last = pd.Timestamp(local.loc[eligible, "__decision_ts__"].max()) + pd.Timedelta(minutes=HORIZON_MINUTES)
    minute = _minute_path_pruned(MINUTE_ROOT, _packb_to_kraken_symbol(str(local.__symbol__.iloc[0])), first, last)
    starts = minute.index.get_indexer(pd.DatetimeIndex(local.__decision_ts__)).astype(np.int64)
    source = minute.loc[:, ["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    high, low, close = (source[column].to_numpy(np.float64) for column in ("high", "low", "close"))
    contract = Exact1mExecutionContract(entry_delay_minutes=0)
    index = np.flatnonzero(eligible & (starts >= 0) & (starts + HORIZON_MINUTES <= len(source)))
    offset = np.arange(HORIZON_MINUTES, dtype=np.int64)
    for first_row in range(0, len(index), CHUNK_ROWS):
        rows = index[first_row:first_row + CHUNK_ROWS]
        positions = starts[rows, None] + offset[None, :]
        paths = {"high": high[positions], "low": low[positions], "close": close[positions]}
        entry = pd.to_numeric(local.loc[rows, "tp6_sl4_entry_price"], errors="coerce").to_numpy(float)
        atr = pd.to_numeric(local.loc[rows, "atr_1h"], errors="coerce").to_numpy(float)
        timestamps = pd.DatetimeIndex(local.loc[rows, "__decision_ts__"])
        canonical_valid: np.ndarray | None = None
        for key, params in variants.items():
            replay = simulate_exact_1m_parent_policy(
                entry=entry, atr=atr, highs=paths["high"], lows=paths["low"], closes=paths["close"],
                entry_timestamps=timestamps, params=params, contract=contract,
                median_atr_fraction=median_atr_fraction, side=SIDE,
            )
            valid = np.asarray(replay["path_valid"], dtype=bool)
            if canonical_valid is None:
                canonical_valid = valid
            elif not np.array_equal(canonical_valid, valid):
                raise AssertionError("policy geometry changed exact-path validity")
            output.loc[rows, f"{key}_gross_bps"] = np.asarray(replay["gross_bps"], dtype=np.float32)
            output.loc[rows, f"{key}_net_bps"] = np.asarray(replay["net_bps"], dtype=np.float32)
            output.loc[rows, f"{key}_exit_minute"] = np.asarray(replay["exit_bar"], dtype=np.int16)
            output.loc[rows, f"{key}_exit_reason"] = np.asarray(replay["exit_reason"], dtype=object)
        output.loc[rows, "policy_path_valid"] = canonical_valid
    return output


def _reference_median(labels_root: Path, start: pd.Timestamp, train_end: pd.Timestamp) -> float:
    pieces: list[pd.DataFrame] = []
    for path in _month_paths(labels_root, start, train_end):
        frame = pd.read_parquet(path, columns=["tp6_sl4_entry_price", "atr_1h"])
        pieces.append(frame)
    all_rows = pd.concat(pieces, ignore_index=True)
    entry = pd.to_numeric(all_rows.tp6_sl4_entry_price, errors="coerce")
    atr = pd.to_numeric(all_rows.atr_1h, errors="coerce")
    ratio = atr / entry
    ratio = ratio[np.isfinite(ratio) & ratio.gt(0.0)]
    if ratio.empty:
        raise ValueError("training-only ATR reference is empty")
    return float(ratio.median())


def _month_audit(frame: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    valid = frame.policy_path_valid.astype(bool)
    assert all(frame.loc[valid, f"{key}_net_bps"].notna().all() for key in keys)
    assert all(frame.loc[~valid, f"{key}_net_bps"].isna().all() for key in keys)
    return {
        "candidate_rows": int(len(frame)),
        "policy_input_valid_rows": int(frame.policy_input_valid.sum()),
        "policy_path_valid_rows": int(valid.sum()),
        "policy_path_coverage": float(valid.mean()),
        "h12_valid_rows": int((frame.label_valid.astype(bool) & ~frame.target_invalid.astype(bool)).sum()),
        "mean_canonical_net_bps_valid": float(pd.to_numeric(frame.loc[valid, "p0_canonical_net_bps"], errors="coerce").mean()) if valid.any() else float("nan"),
    }


def _resolve_median_atr_fraction(
    *, labels_root: Path, start: pd.Timestamp, train_end: pd.Timestamp,
    frozen_median_atr_fraction: float | None,
) -> tuple[float, str]:
    """Return a training-only ATR scale without reopening evaluation labels.

    A frozen override is deliberately allowed for a later target-free label
    extension: the selected P0 contract was fit with a pre-2025 reference, so
    recomputing its median from evaluation rows would change the target
    semantics.  The caller must persist the value and its provenance.
    """
    if frozen_median_atr_fraction is not None:
        value = float(frozen_median_atr_fraction)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("frozen median ATR fraction must be finite and positive")
        return value, "frozen_pre_evaluation_reference"
    if not (start < train_end < end):
        raise ValueError("require start < train_end < end when median is not frozen")
    return _reference_median(labels_root, start, train_end), "computed_from_labels_root_before_train_end"


def run(
    *, out: Path, labels_root: Path, start: pd.Timestamp, end: pd.Timestamp,
    train_end: pd.Timestamp, resume: bool, frozen_median_atr_fraction: float | None = None,
) -> Path:
    if out.exists() and not resume:
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True, exist_ok=resume)
    variants = _variants()
    keys = list(variants)
    median_atr, median_source = _resolve_median_atr_fraction(
        labels_root=labels_root,
        start=start,
        train_end=train_end,
        frozen_median_atr_fraction=frozen_median_atr_fraction,
    )
    audits: dict[str, Any] = {}
    for path in _month_paths(labels_root, start, end):
        month = path.parent.name.removeprefix("month=")
        destination = out / "parts" / f"month={month}" / "side=short.parquet"
        if resume and destination.exists():
            prior = pd.read_parquet(destination, columns=["candidate_id", "policy_path_valid"])
            audits[month] = {"status": "reused", "candidate_rows": int(len(prior)), "policy_path_valid_rows": int(prior.policy_path_valid.sum())}
            continue
        print(f"materialising short policy labels {month}", flush=True)
        source = _load_month(path)
        pieces = [
            _materialize_symbol(group.reset_index(drop=True), variants=variants, median_atr_fraction=median_atr)
            for _, group in source.groupby("__symbol__", sort=True)
        ]
        result = pd.concat(pieces, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if len(result) != len(source) or result.candidate_id.duplicated().any():
            raise AssertionError("policy label materialisation changed candidate identities")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(destination, index=False, compression="zstd")
        audits[month] = _month_audit(result, keys)
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "side": SIDE,
        "decision_window": f"[{start.isoformat()}, {end.isoformat()})",
        "training_reference_end": train_end.isoformat(),
        "entry": "exact decision-minute open; signal close + one hour",
        "label_available_at": "decision timestamp + 12 hours",
        "policy_cost_bps_once": 100.0,
        "policy_variants": {key: value.to_dict() for key, value in variants.items()},
        "median_atr_fraction_training_only": median_atr,
        "median_atr_fraction_source": median_source,
        "candidate_identity_source": str(labels_root),
        "labels_manifest_sha256": _sha256(labels_root / "run_manifest.json"),
        "minute_root": str(MINUTE_ROOT),
        "invalid_policy_semantics": "path-invalid outputs are null and excluded from supervision; never economic zero failures",
        "month_audit": audits,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--start", default="2023-10-01T00:00:00Z")
    parser.add_argument("--train-end", default="2024-10-01T00:00:00Z")
    parser.add_argument("--end", default="2025-01-01T00:00:00Z")
    parser.add_argument(
        "--frozen-median-atr-fraction",
        type=float,
        help="predeclared training-only ATR/entry median; do not recompute it from evaluation labels",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run(
        out=args.out.resolve(), labels_root=args.labels.resolve(), start=_utc(args.start),
        end=_utc(args.end), train_end=_utc(args.train_end), resume=bool(args.resume),
        frozen_median_atr_fraction=args.frozen_median_atr_fraction,
    ))


if __name__ == "__main__":
    main()
