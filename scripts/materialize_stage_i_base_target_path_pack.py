#!/usr/bin/env python3
"""Build the exact H12 path pack for the selected Stage-I candidate surface.

This adapter reuses the canonical exact-label minute loader and causal Wilder
ATR implementation.  It reads the final selector identities, enters at the
minute-bar open indexed exactly at ``signal __ts__ + 1h``, and stores 720
post-entry minute bars.  Each array row is bound to candidate ID, entry time,
and path-start epoch before the downstream 60-arm label materialiser accepts
it.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_base_target_ablation import file_sha256  # noqa: E402
from scripts.materialize_packb_tp6_sl4_h12_labels import (  # noqa: E402
    _causal_hourly_atr_from_minute,
    _complete_h12_paths,
    _minute_path_pruned,
    _overlapping_minute_fragments,
    _packb_to_kraken_symbol,
)


HORIZON = 720
REGIME_FIELDS = ("is_low_vol_regime", "is_high_vol_regime", "is_trending")


def _canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _materializer_source_fingerprint() -> dict[str, Any]:
    """Hash the full code closure which defines path/ATR/entry semantics."""

    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "materialize_packb_tp6_sl4_h12_labels.py",
        ROOT / "scripts" / "materialize_full_universe_t2_t4_panel.py",
        ROOT / "scripts" / "materialize_full_universe_tp6_sl4_h12_sidecar.py",
    )
    payload = {
        "schema": "stage_i_target_path_materializer_source_v1",
        "files": {str(path.resolve()): file_sha256(path) for path in paths},
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    return payload


def _minute_source_inventory(
    *, minute_root: Path, symbol_values: pd.Series, signal: pd.Series,
) -> dict[str, Any]:
    """Inventory the exact source fragments selected by the minute loader.

    We hash contents rather than only recording mtimes.  Consequently a resume
    cannot reuse a pack if a relevant historical minute fragment was repaired,
    replaced, or silently rewritten.
    """

    rows: list[dict[str, Any]] = []
    for raw_symbol in sorted(symbol_values.astype(str).unique()):
        local_signal = signal.loc[symbol_values.astype(str).eq(raw_symbol)]
        start = local_signal.min() - pd.Timedelta(hours=14)
        end = local_signal.max() + pd.Timedelta(hours=13)
        kraken_symbol = _packb_to_kraken_symbol(raw_symbol)
        fragments = _overlapping_minute_fragments(minute_root, kraken_symbol, start, end)
        if not fragments:
            rows.append({
                "raw_symbol": raw_symbol, "kraken_symbol": kraken_symbol,
                "requested_start": start.isoformat(), "requested_end_exclusive": end.isoformat(),
                "relative_path": None, "bytes": None, "sha256": None,
                "status": "no_overlapping_fragment_discovered",
            })
        for fragment in fragments:
            stat = fragment.stat()
            rows.append({
                "raw_symbol": raw_symbol, "kraken_symbol": kraken_symbol,
                "requested_start": start.isoformat(), "requested_end_exclusive": end.isoformat(),
                "relative_path": str(fragment.resolve().relative_to(minute_root.resolve())),
                "bytes": int(stat.st_size), "sha256": file_sha256(fragment),
                "status": "content_hash_bound",
            })
    rows.sort(key=lambda row: (row["raw_symbol"], str(row["relative_path"])))
    payload = {
        "schema": "stage_i_target_minute_source_inventory_v1",
        "minute_root": str(minute_root.resolve()),
        "rows": rows,
        "content_hash_policy": "all discovered overlapping immutable minute fragments are SHA256-bound",
    }
    payload["inventory_sha256"] = _canonical_sha(payload)
    return payload


def _completed_resume(path: Path, request_sha256: str) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("request_sha256") != request_sha256:
        raise ValueError("completed path-pack request/source lineage drift; a fresh output directory is required")
    inventory = manifest.get("artifact_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("completed path-pack manifest lacks immutable artifact inventory")
    for relative, expected in inventory.items():
        artifact = path / relative
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise ValueError(f"completed path-pack artifact drift: {relative}")
    return manifest


def _regime(features: pd.DataFrame) -> pd.Series:
    missing = sorted(set(REGIME_FIELDS).difference(features.columns))
    if missing:
        raise ValueError(f"selector lacks preregistered causal regime fields: {missing}")
    low = pd.to_numeric(features.is_low_vol_regime, errors="coerce")
    high = pd.to_numeric(features.is_high_vol_regime, errors="coerce")
    trend = pd.to_numeric(features.is_trending, errors="coerce")
    missing_input = pd.concat([low, high, trend], axis=1).isna().any(axis=1).to_numpy()
    volatility = np.where(high.gt(.5), "high_vol", np.where(low.gt(.5), "low_vol", "mid_vol"))
    shape = np.where(trend.gt(.5), "trend", "chop")
    regime = np.char.add(np.char.add(volatility, "__"), shape).astype(object)
    # Missing decision-time context is observable at decision time.  Preserve
    # it explicitly instead of imputing a future-aware state or discarding an
    # otherwise valid path label.
    regime[missing_input] = "causal_unknown"
    return pd.Series(regime, index=features.index, dtype="string")


def materialize(selector_dir: Path, minute_root: Path, output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite path pack: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    ledger_path = selector_dir / "selector_ledger.parquet"
    features_path = selector_dir / "selector_features.parquet"
    manifest_path = selector_dir / "manifest.json"
    contract_path = selector_dir / "selector_feature_contract.json"
    if not all(path.is_file() for path in (ledger_path, features_path, manifest_path, contract_path)):
        raise ValueError("final selector artifact is incomplete")
    selector_manifest = json.loads(manifest_path.read_text())
    selector_contract = json.loads(contract_path.read_text())
    if selector_manifest.get("status") != "complete" or selector_contract.get("max_feature_columns") not in (0, None):
        raise ValueError("path pack requires the final uncapped selector")
    integrity = selector_manifest.get("artifact_integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("schema") != "stage_i_selector_artifact_integrity_v1"
        or integrity.get("selector_ledger_sha256") != file_sha256(ledger_path)
        or integrity.get("selector_features_sha256") != file_sha256(features_path)
    ):
        raise ValueError("selector ledger/features fail immutable artifact-integrity validation")
    if not set(REGIME_FIELDS).issubset(set(map(str, selector_contract.get("feature_columns", ())))):
        raise ValueError("causal regime fields escape the selector inference-feature contract")
    ledger = pd.read_parquet(ledger_path)
    features = pd.read_parquet(features_path, columns=["candidate_id", "__ts__", "__symbol__", *REGIME_FIELDS])
    identity = ["candidate_id", "__ts__", "__symbol__"]
    if not ledger.loc[:, identity].reset_index(drop=True).equals(features.loc[:, identity].reset_index(drop=True)):
        raise ValueError("selector ledger/features identity order drift")
    if ledger.candidate_id.isna().any() or ledger.candidate_id.duplicated().any():
        raise ValueError("selector candidate identities are invalid")
    signal = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise")
    decision = signal + pd.Timedelta(hours=1)
    if "decision_ts" in ledger and not pd.to_datetime(ledger.decision_ts, utc=True, errors="raise").eq(decision).all():
        raise ValueError("selector decision timestamp no longer equals signal +1h")
    minute_inventory = _minute_source_inventory(
        minute_root=minute_root, symbol_values=ledger["__symbol__"], signal=signal,
    )
    source_fingerprint = _materializer_source_fingerprint()
    request = {
        "schema": "stage_i_base_target_exact_h12_path_pack_request_v2",
        "selector_manifest_sha256": file_sha256(manifest_path),
        "selector_contract_sha256": file_sha256(contract_path),
        "selector_ledger_sha256": file_sha256(ledger_path),
        "selector_features_sha256": file_sha256(features_path),
        "source_fingerprint": source_fingerprint,
        "minute_source_inventory_sha256": minute_inventory["inventory_sha256"],
        "entry_convention": "signal_timestamp_plus_1h_exact_minute_open",
        "horizon_minutes": HORIZON,
    }
    request_sha256 = _canonical_sha(request)
    if resume:
        prior = _completed_resume(output_dir, request_sha256)
        if prior is not None:
            return prior
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("partial/unmatched path pack requires a fresh output directory")
    n = len(ledger)
    temporary = Path(tempfile.mkdtemp(prefix="stage_i_target_paths_", dir=str(output_dir.parent)))
    try:
        high = np.lib.format.open_memmap(temporary / "high.npy", mode="w+", dtype=np.float32, shape=(n, HORIZON))
        low = np.lib.format.open_memmap(temporary / "low.npy", mode="w+", dtype=np.float32, shape=(n, HORIZON))
        close = np.lib.format.open_memmap(temporary / "close.npy", mode="w+", dtype=np.float32, shape=(n, HORIZON))
        high[:] = np.nan; low[:] = np.nan; close[:] = np.nan
        entry = np.full(n, np.nan, dtype=np.float64)
        atr = np.full(n, np.nan, dtype=np.float64)
        complete = np.zeros(n, dtype=bool)
        symbol_values = ledger["__symbol__"].astype(str)
        for raw_symbol in sorted(symbol_values.unique()):
            position = np.flatnonzero(symbol_values.eq(raw_symbol).to_numpy())
            local_signal = signal.iloc[position]
            start = local_signal.min() - pd.Timedelta(hours=14)
            end = local_signal.max() + pd.Timedelta(hours=13)
            minute = _minute_path_pruned(
                minute_root, _packb_to_kraken_symbol(raw_symbol), start, end
            )
            local_decision = pd.DatetimeIndex(decision.iloc[position])
            starts = minute.index.get_indexer(local_decision).astype(np.int64)
            atr_hourly = _causal_hourly_atr_from_minute(minute)
            local_atr = atr_hourly.reindex(pd.DatetimeIndex(local_signal)).to_numpy(np.float64)
            local_complete = _complete_h12_paths(minute, starts)
            ohlc = minute[["open", "high", "low", "close"]].to_numpy(np.float64)
            in_bounds = (starts >= 0) & (starts + HORIZON <= len(minute))
            for local_index, global_index in enumerate(position):
                if starts[local_index] >= 0:
                    entry[global_index] = ohlc[starts[local_index], 0]
                atr[global_index] = local_atr[local_index]
                complete[global_index] = bool(
                    in_bounds[local_index] and local_complete[local_index]
                    and np.isfinite(entry[global_index]) and np.isfinite(local_atr[local_index])
                    and local_atr[local_index] > 0.0
                )
                if complete[global_index]:
                    path = ohlc[starts[local_index]:starts[local_index] + HORIZON]
                    high[global_index] = path[:, 1]
                    low[global_index] = path[:, 2]
                    close[global_index] = path[:, 3]
        high.flush(); low.flush(); close.flush()
        # A live historical-data rewrite during materialisation would otherwise
        # produce a path pack whose declared source inventory is stale.
        if _minute_source_inventory(
            minute_root=minute_root, symbol_values=ledger["__symbol__"], signal=signal,
        )["inventory_sha256"] != minute_inventory["inventory_sha256"]:
            raise ValueError("minute-source inventory changed during path-pack materialisation")
        output_dir.mkdir(parents=True, exist_ok=True)
        path_end = decision + pd.Timedelta(minutes=HORIZON)
        candidate = ledger.loc[:, identity + ["side_name"]].copy()
        candidate["decision_ts"] = decision
        candidate["entry_ts"] = decision
        candidate["path_start_ts"] = decision
        candidate["path_end_exclusive"] = path_end
        candidate["label_available_ts"] = path_end
        candidate["entry_price"] = entry
        candidate["atr_1h"] = atr
        candidate["path_complete"] = complete
        candidate["causal_regime"] = _regime(features).to_numpy()
        candidate_path = output_dir / "candidate_paths.parquet"
        candidate.to_parquet(candidate_path, index=False, compression="zstd")
        identity_sha = np.vstack([
            np.frombuffer(sha256(
                (str(row.candidate_id) + "\x1f" + pd.Timestamp(row.entry_ts).isoformat()).encode()
            ).digest(), dtype=np.uint8)
            for row in candidate[["candidate_id", "entry_ts"]].itertuples(index=False)
        ])
        archive_path = output_dir / "h12_paths.npz"
        np.savez(
            archive_path, high=high, low=low, close=close,
            entry_open=entry,
            path_start_ns=decision.astype("int64").to_numpy(np.int64),
            identity_sha256=identity_sha,
        )
        regime_contract = {
            "column": "causal_regime", "causal_at_decision_time": True,
            "diagnostic_noncausal": False,
            "formula": "low/mid/high volatility x trend/chop from three selector decision-time fields",
            "input_fields": list(REGIME_FIELDS),
            "missing_input_policy": "causal_unknown; no imputation; retained in coverage/economics and excluded from supported-regime promotion minimum",
            "source_manifest_sha256": file_sha256(manifest_path),
            "source_feature_contract_sha256": file_sha256(contract_path),
        }
        minute_inventory_path = output_dir / "minute_source_inventory.json"
        minute_inventory_path.write_text(json.dumps(minute_inventory, indent=2, sort_keys=True) + "\n")
        manifest = {
            "schema": "stage_i_base_target_exact_h12_path_pack_v2", "status": "complete",
            "rows": n, "valid_complete_rows": int(complete.sum()),
            "entry_convention": "signal_timestamp_plus_1h_exact_minute_open",
            "horizon_minutes": HORIZON,
            "path_interval": "[entry_ts, entry_ts+H12)",
            "label_availability": "entry_ts+H12 exactly",
            "atr": "canonical causal Wilder ATR14 from 14 completed hourly candles at signal timestamp",
            "causal_regime_contract": regime_contract,
            "selector_manifest_sha256": file_sha256(manifest_path),
            "selector_contract_sha256": file_sha256(contract_path),
            "selector_ledger_sha256": file_sha256(ledger_path),
            "selector_features_sha256": file_sha256(features_path),
            "minute_root": str(minute_root.resolve()),
            "materializer_source_fingerprint": source_fingerprint,
            "minute_source_inventory": {
                "path": "minute_source_inventory.json",
                "inventory_sha256": minute_inventory["inventory_sha256"],
                "policy": "content-hash-bound overlapping source fragments; resume rejects inventory drift",
            },
            "request": request,
            "request_sha256": request_sha256,
            "artifact_sha256": {
                "candidate_paths.parquet": file_sha256(candidate_path),
                "h12_paths.npz": file_sha256(archive_path),
                "minute_source_inventory.json": file_sha256(minute_inventory_path),
            },
        }
        manifest["contract_sha256"] = _canonical_sha({key: value for key, value in manifest.items() if key != "artifact_sha256"})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument(
        "--minute-root", type=Path,
        default=ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.selector_dir, args.minute_root, args.output_dir, resume=args.resume), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
