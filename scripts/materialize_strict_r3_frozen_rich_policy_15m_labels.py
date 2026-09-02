#!/usr/bin/env python3
"""Materialise frozen rich-policy H12 outcomes for offline Strict-R3 research.

This producer is deliberately outcome-only.  It starts with a pre-existing
target-free candidate identity universe, loads only its decision timestamp and
symbol, then joins future 15-minute bars *after* that identity is fixed.  It
never changes routing, score eligibility, MC1, portfolio state, or live code.

The policy is the sealed long frozen rich contract: decision-time hourly
Wilder-14 ATR, H12 / 48 completed 15-minute bars, hard stop, smooth capital
protection, trailing profit, fast adverse exit, timeout, and one 100-bps cost.
The output is a 15-minute historical outcome proxy.  It is not an exact
one-minute execution replay and is labelled as such in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams, simulate_rich_policy  # noqa: E402
from scripts.run_strict_r3_rich_policy_hpo import HORIZON_BARS, _hourly_signal_atr, _symbol_filename  # noqa: E402


SCHEMA = "strict_r3_frozen_rich_policy_15m_labels_v1"
REQUIRED = (
    "candidate_id", "__decision_ts__", "__symbol__",
)


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _assert_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _load_frozen_policy(path: Path) -> tuple[RichPolicyParams, float, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not np.isclose(float(payload.get("cost_bps", np.nan)), 100.0):
        raise AssertionError("frozen rich policy must apply exactly 100 bps of cost")
    params = RichPolicyParams.from_mapping(dict(payload.get("params") or {}))
    if not bool(params.smooth_capital_protection_enabled):
        raise AssertionError("requested frozen policy is missing smooth capital protection")
    if not bool(params.adverse_exit_enabled):
        raise AssertionError("requested frozen policy is missing its fast-adverse rule")
    median = float(payload.get("median_atr_fraction_fitted_on_complete_2024_development", np.nan))
    if not np.isfinite(median) or median <= 0.0:
        raise AssertionError("frozen rich policy has invalid development ATR reference")
    return params, median, payload


def _candidate_population(
    *,
    candidate_root: Path | None,
    source_policy: Path | None,
    candidate_score_root: Path | None,
    candidate_score_file: Path | None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Fix the outcome universe before reading any future 15-minute path."""
    if candidate_score_file is not None:
        path = Path(candidate_score_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        names = set(pd.read_parquet(path, engine="pyarrow").columns)
        prohibited = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts"}
        leaked = sorted(prohibited.intersection(names))
        if leaked:
            raise AssertionError(f"{path}: candidate score file is not target-free: {leaked}")
        missing = sorted(set(REQUIRED).difference(names))
        if missing:
            raise AssertionError(f"{path}: missing target-free identity fields {missing}")
        population = pd.read_parquet(path, columns=list(REQUIRED))
        population["__decision_ts__"] = pd.to_datetime(population["__decision_ts__"], utc=True, errors="raise")
        population["__symbol__"] = population["__symbol__"].astype(str)
        population["candidate_id"] = population["candidate_id"].astype(str)
        if population["candidate_id"].duplicated().any():
            raise AssertionError("target-free candidate score file has duplicate candidate IDs")
        if not population.candidate_id.str.contains("\\|long\\|").all():
            raise AssertionError("frozen rich-policy outcome extension is long-only")
        return population.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable"), [path]
    if candidate_score_root is not None:
        paths = sorted(candidate_score_root.glob("month=*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no target-free score months under {candidate_score_root}")
        parts: list[pd.DataFrame] = []
        prohibited = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts"}
        for path in paths:
            names = set(pd.read_parquet(path, engine="pyarrow").columns)
            leaked = sorted(prohibited.intersection(names))
            if leaked:
                raise AssertionError(f"{path}: candidate score source is not target-free: {leaked}")
            missing = sorted({"candidate_id", "__decision_ts__", "side_name"}.difference(names))
            if missing:
                raise AssertionError(f"{path}: missing target-free identity fields {missing}")
            raw = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "side_name"])
            raw["__symbol__"] = raw["candidate_id"].astype(str).str.split("|", n=1, expand=True)[0]
            parts.append(raw.loc[:, list(REQUIRED)])
        population = pd.concat(parts, ignore_index=True)
        population["__decision_ts__"] = pd.to_datetime(population["__decision_ts__"], utc=True, errors="raise")
        population["__symbol__"] = population["__symbol__"].astype(str)
        if population["candidate_id"].duplicated().any():
            raise AssertionError("target-free score source has duplicate candidate IDs")
        if not population.candidate_id.astype(str).str.contains("\\|long\\|").all():
            raise AssertionError("frozen rich-policy outcome extension is long-only")
        return population.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable"), paths

    if candidate_root is None or source_policy is None:
        raise ValueError("candidate-root/source-policy are required without candidate-score-root")
    source_ids = pd.read_parquet(source_policy, columns=["candidate_id"])["candidate_id"].astype(str)
    if source_ids.duplicated().any():
        raise AssertionError("source candidate identity panel has duplicate IDs")
    required_ids = set(source_ids)
    pieces: list[pd.DataFrame] = []
    source_parts = sorted((candidate_root / "parts").glob("month=*/side=long.parquet"))
    if not source_parts:
        raise FileNotFoundError(f"no long candidate parts under {candidate_root}")
    for path in source_parts:
        part = pd.read_parquet(path, columns=list(REQUIRED))
        part["candidate_id"] = part["candidate_id"].astype(str)
        part = part.loc[part["candidate_id"].isin(required_ids)]
        if not part.empty:
            pieces.append(part)
    if not pieces:
        raise RuntimeError("none of the fixed policy-universe identities occur in the candidate population")
    population = pd.concat(pieces, ignore_index=True)
    population["__decision_ts__"] = pd.to_datetime(population["__decision_ts__"], utc=True, errors="raise")
    population["__symbol__"] = population["__symbol__"].astype(str)
    if population["candidate_id"].duplicated().any():
        raise AssertionError("candidate identity collision while joining rich-policy source")
    missing = required_ids - set(population["candidate_id"])
    if missing:
        raise AssertionError(f"candidate population is missing {len(missing)} fixed source identities")
    return population.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable"), source_parts


def _invalid_rows(frame: pd.DataFrame, reason: str) -> pd.DataFrame:
    count = len(frame)
    return pd.DataFrame({
        "candidate_id": frame["candidate_id"].astype(str).to_numpy(),
        "policy_path_valid": np.zeros(count, dtype=bool),
        "policy_gross_bps": np.full(count, np.nan),
        "policy_net_bps": np.full(count, np.nan),
        "policy_exit_bar_15m": np.full(count, -1, dtype=np.int16),
        "policy_entry_price": np.full(count, np.nan),
        "policy_exit_price": np.full(count, np.nan),
        "policy_exit_reason": np.full(count, reason, dtype=object),
        "policy_label_available_ts": pd.Series(pd.NaT, index=np.arange(count), dtype="datetime64[ns, UTC]"),
        "policy_cost_bps": np.full(count, np.nan),
        "policy_outcome_source": np.full(count, "frozen_rich_15m_aggregate", dtype=object),
    })


def _materialize_symbol(
    frame: pd.DataFrame,
    *,
    bars_root: Path,
    params: RichPolicyParams,
    median_atr_fraction: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    symbol = str(frame["__symbol__"].iloc[0])
    source = bars_root / _symbol_filename(symbol)
    if not source.is_file():
        return _invalid_rows(frame, "missing_15m_symbol_source"), {"symbol": symbol, "rows": len(frame), "valid_rows": 0, "reason": "missing_15m_symbol_source"}
    bars = pd.read_parquet(source, columns=["open", "high", "low", "close"])
    bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars = bars.loc[~bars.index.isna()].sort_index()
    bars = bars.loc[~bars.index.duplicated(keep="last")]
    if bars.empty:
        return _invalid_rows(frame, "empty_15m_symbol_source"), {"symbol": symbol, "rows": len(frame), "valid_rows": 0, "reason": "empty_15m_symbol_source"}
    bars = bars.apply(pd.to_numeric, errors="coerce")
    atr = _hourly_signal_atr(bars)
    decisions = pd.DatetimeIndex(frame["__decision_ts__"])
    positions = bars.index.get_indexer(decisions)
    offsets = np.arange(HORIZON_BARS, dtype=np.int64)
    matrix_index = positions[:, None] + offsets[None, :]
    in_range = (positions >= 0) & (matrix_index[:, -1] < len(bars))
    atr_values = atr.reindex(decisions).to_numpy(np.float64)
    valid = in_range & np.isfinite(atr_values) & (atr_values > 0.0)
    for column in ("open", "high", "low", "close"):
        values = bars[column].to_numpy(np.float64)
        valid_indices = np.flatnonzero(in_range)
        if len(valid_indices):
            valid[valid_indices] &= np.isfinite(values[matrix_index[valid_indices]]).all(axis=1)
    invalid = _invalid_rows(frame, "incomplete_15m_h12_path_or_causal_atr")
    if not valid.any():
        return invalid, {"symbol": symbol, "rows": len(frame), "valid_rows": 0, "reason": "incomplete_15m_h12_path_or_causal_atr"}
    selected = frame.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    locations = matrix_index[valid]
    entry = bars["open"].to_numpy(np.float64)[locations[:, 0]]
    high = bars["high"].to_numpy(np.float64)[locations]
    low = bars["low"].to_numpy(np.float64)[locations]
    close = bars["close"].to_numpy(np.float64)[locations]
    simulated = simulate_rich_policy(
        entry=entry,
        atr=atr_values[valid],
        highs=high,
        lows=low,
        closes=close,
        params=params,
        median_atr_fraction=float(median_atr_fraction),
        side="long",
    )
    if not np.asarray(simulated["path_valid"], dtype=bool).all():
        raise AssertionError(f"{symbol}: simulator rejected a prevalidated path")
    exit_bar = np.asarray(simulated["exit_bar"], dtype=np.int16)
    gross = np.asarray(simulated["gross_bps"], dtype=np.float64)
    net = np.asarray(simulated["net_bps"], dtype=np.float64)
    if not np.isclose(gross - net, 100.0, rtol=0.0, atol=1e-8).all():
        raise AssertionError(f"{symbol}: frozen rich cost was not applied exactly once")
    # Assign by dtype-compatible columns rather than writing a mixed object
    # matrix into the invalid frame.  That preserves the immutable-row order
    # without relying on pandas' deprecated implicit coercions.
    output = invalid.copy()
    output.loc[valid, "policy_path_valid"] = True
    output.loc[valid, "policy_gross_bps"] = gross
    output.loc[valid, "policy_net_bps"] = net
    output.loc[valid, "policy_exit_bar_15m"] = exit_bar.astype(np.int16)
    output.loc[valid, "policy_entry_price"] = entry
    output.loc[valid, "policy_exit_price"] = entry * (1.0 + gross / 10_000.0)
    output.loc[valid, "policy_exit_reason"] = np.asarray(simulated["exit_reason"], dtype=object)
    output.loc[valid, "policy_label_available_ts"] = (
        selected["__decision_ts__"].to_numpy() + pd.Timedelta(hours=12)
    )
    output.loc[valid, "policy_cost_bps"] = 100.0
    output.loc[valid, "policy_outcome_source"] = "frozen_rich_15m_aggregate"
    output["policy_path_valid"] = output["policy_path_valid"].fillna(False).astype(bool)
    output["policy_exit_bar_15m"] = pd.to_numeric(output["policy_exit_bar_15m"], errors="coerce").fillna(-1).astype(np.int16)
    return output, {"symbol": symbol, "rows": len(frame), "valid_rows": int(valid.sum()), "reason": "ok"}


def run(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    _assert_new_output(out)
    params, median_atr, policy_payload = _load_frozen_policy(args.frozen_policy.resolve())
    population, candidate_parts = _candidate_population(
        candidate_root=args.candidate_root.resolve() if args.candidate_root else None,
        source_policy=args.source_policy.resolve() if args.source_policy else None,
        candidate_score_root=args.candidate_score_root.resolve() if args.candidate_score_root else None,
        candidate_score_file=args.candidate_score_file.resolve() if args.candidate_score_file else None,
    )
    coverage: list[dict[str, object]] = []
    parts_root = out / "policy_parts"
    for symbol, group in population.groupby("__symbol__", sort=True):
        outcome, audit = _materialize_symbol(
            group.reset_index(drop=True), bars_root=args.bars_root.resolve(), params=params, median_atr_fraction=median_atr,
        )
        if len(outcome) != len(group) or outcome["candidate_id"].duplicated().any():
            raise AssertionError(f"{symbol}: rich outcome identity mismatch")
        path = parts_root / f"symbol={str(symbol).replace('/', '_')}" / "policy_labels.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        outcome.to_parquet(path, index=False, compression="zstd")
        coverage.append(audit)
        print(json.dumps({"event": "materialized", **audit}), flush=True)
    coverage_frame = pd.DataFrame(coverage).sort_values("symbol", kind="stable")
    coverage_frame.to_parquet(out / "coverage_by_symbol.parquet", index=False, compression="zstd")
    valid_rows = int(coverage_frame["valid_rows"].sum())
    total_rows = int(coverage_frame["rows"].sum())
    manifest = {
        "schema": SCHEMA,
        "scope": "offline research outcome materialization only",
        "candidate_contract": "fixed source candidate identities before future 15m paths are loaded; invalid paths remain explicit and are excluded only from supervised fitting/replay",
        "policy": {
            "frozen_policy": str(args.frozen_policy.resolve()),
            "frozen_policy_sha256": _sha256([args.frozen_policy.resolve()]),
            "params": params.to_dict(),
            "median_atr_fraction": median_atr,
            "cost_bps_once": 100.0,
            "entry": "decision timestamp 15m open",
            "horizon": "48 completed 15m bars / H12",
            "resolution": "15m aggregate rich-policy proxy; exact one-minute replay is a separately named contract",
        },
        "sources": {
            "candidate_root": str(args.candidate_root.resolve()) if args.candidate_root else None,
            "candidate_part_count": len(candidate_parts),
            "candidate_parts_sha256": _sha256(candidate_parts),
            "source_policy_identity": str(args.source_policy.resolve()) if args.source_policy else None,
            "source_policy_identity_sha256": _sha256([args.source_policy.resolve()]) if args.source_policy else None,
            "candidate_score_root": str(args.candidate_score_root.resolve()) if args.candidate_score_root else None,
            "candidate_score_file": str(args.candidate_score_file.resolve()) if args.candidate_score_file else None,
            "bars_root": str(args.bars_root.resolve()),
        },
        "coverage": {"rows": total_rows, "valid_rows": valid_rows, "valid_fraction": valid_rows / max(total_rows, 1)},
        "prohibitions": ["no_score_selection", "no_live_state", "no_exchange_io", "no_outcome_feature_inputs"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--source-policy", type=Path, help="Identity universe only; its previous outcome columns are never read.")
    parser.add_argument(
        "--candidate-score-root", type=Path,
        help=(
            "Target-free score root with month=YYYY-MM.parquet files. It is "
            "an alternative identity source that never opens a policy ledger."
        ),
    )
    parser.add_argument(
        "--candidate-score-file", type=Path,
        help="Single immutable target-free candidate score file; alternative identity source.",
    )
    parser.add_argument("--bars-root", type=Path, default=ROOT / "15m_ohlcv_perp")
    parser.add_argument("--frozen-policy", type=Path, default=ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json")
    args = parser.parse_args()
    score_modes = int(args.candidate_score_root is not None) + int(args.candidate_score_file is not None)
    legacy_mode = args.candidate_root is not None or args.source_policy is not None
    if score_modes + int(legacy_mode) != 1:
        parser.error("provide exactly one target-free score source or both --candidate-root and --source-policy")
    if not score_modes and (args.candidate_root is None or args.source_policy is None):
        parser.error("--candidate-root and --source-policy must be supplied together")
    return args


if __name__ == "__main__":
    destination = run(parse_args())
    print(destination)
