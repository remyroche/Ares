"""Finalize and audit the targeted OI/funding feature-store repair.

The feature pipeline writes only source-backed symbols and can deliberately
skip a non-critical key when its causal inputs are unavailable.  This pass is
fail-closed: it removes stale values from untouched symbols and skipped keys,
and rewrites the raw ``funding_rate`` alias from the availability-indexed
sidecar with the canonical bounded carry rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SKIPPED_KEYS = {
    "carry_adj_ret_3h", "carry_adj_ret_7h", "carry_adj_ret_self_z_3h",
    "carry_adj_ret_self_z_7h", "carry_adj_short_ret_3h", "carry_adj_short_ret_7h",
    "carry_adj_short_ret_self_z_3h", "carry_adj_short_ret_self_z_7h",
    "fund_flip_x_vol_expansion_3h", "fund_flip_x_vol_expansion_7h",
    "fund_high_neg_mom_3h", "fund_high_neg_mom_7h", "fund_high_neg_mom_self_z_3h",
    "fund_high_neg_mom_self_z_7h", "fund_payment_pressure_3h", "fund_payment_pressure_7h",
    "fund_post_reversal_3h", "fund_post_reversal_7h", "fund_pre_drift_3h",
    "fund_pre_drift_7h", "fund_ret_cond_sign_3h", "fund_ret_cond_sign_7h",
    "funding_crowded_mom_exhaustion_3h", "funding_crowded_mom_exhaustion_7h",
    "funding_crowded_mom_exhaustion_self_z_3h", "funding_crowded_mom_exhaustion_self_z_7h",
    "persistent_neg_funding_failed_breakdown_3h", "persistent_neg_funding_failed_breakdown_7h",
    "persistent_pos_funding_failed_breakout_3h", "persistent_pos_funding_failed_breakout_7h",
}

# These keys are intentionally unavailable under the canonical native-OI
# contract or were absent from every active PF feature file after regeneration.
# Keeping them explicit prevents a future broad backfill from treating an
# uncomputable column as an ordinary sparse feature.
STRUCTURALLY_UNSUPPORTED_KEYS = {
    "bars_since_mkt_oi_trough",
    "cs_rank_oi_value_z_30d",
    "log_oi_to_volume_1d",
    "log_oi_to_volume_1d_cp_absratio_8_32",
    "log_oi_to_volume_1d_cp_logstd_8_32",
    "log_oi_to_volume_1d_cp_z_8_32_96",
    "log_oi_to_volume_7d",
    "mkt_funding_weighted_by_oi",
    "mkt_oi_concentration_btc_eth",
    "mkt_oi_drawdown_from_24h_peak",
    "mkt_oi_drawdown_from_7d_peak",
    "mkt_oi_recovery_from_24h_low",
    "mkt_oi_z_30d",
    "oi_to_volume_1d_z_90d",
    "oi_to_volume_7d_z_180d",
    "oi_value_log_cp_absratio_8_32",
    "oi_value_log_cp_logstd_8_32",
    "oi_value_log_cp_z_8_32_96",
    "oi_value_log_z_30d",
    "oi_value_log_z_90d",
    "oi_value_pct_90d",
    "oi_value_z_30d",
    "oi_value_z_90d",
}
SKIPPED_KEYS |= STRUCTURALLY_UNSUPPORTED_KEYS

PERSISTENT_ELIGIBILITY_COLUMNS = {
    "__oi_fresh_bars__",
    "__funding_fresh_bars__",
    "__oi_exact_available__",
    "__funding_exact_available__",
    "__oi_available__",
    "__funding_alias_available__",
    "__funding_available__",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_symbol(path: Path) -> str:
    return path.stem.removeprefix("symbol=")


def _sidecar_key(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _read_feature(path: Path, requested: list[str]) -> tuple[pd.DataFrame, list[str]]:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in requested if c in names]
    # Eligibility metadata is part of the persistent feature-store contract,
    # not a transient audit column.  Preserve it through every targeted
    # finalization so a later repair cannot silently drop source masks.
    metadata = [c for c in sorted(PERSISTENT_ELIGIBILITY_COLUMNS) if c in names]
    read = [*cols, *metadata]
    if "ts" in names:
        read.append("ts")
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(read)))
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        frame = frame.set_index("ts")
    else:
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    return frame, cols


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.finalize.{os.getpid()}.tmp")
    frame.to_parquet(tmp, engine="pyarrow", compression="zstd")
    os.replace(tmp, path)


def _load_sidecar(root: Path, symbol: str, column: str) -> pd.Series:
    path = root / f"{_sidecar_key(symbol)}.parquet"
    if not path.exists():
        return pd.Series(dtype="float32", index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.read_parquet(path)
    idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
    value_col = column if column in frame.columns else next(iter(frame.columns), None)
    if value_col is None:
        return pd.Series(dtype="float32", index=pd.DatetimeIndex([], tz="UTC"))
    series = pd.to_numeric(frame[value_col], errors="coerce")
    series.index = idx
    series = series.loc[~series.index.isna()]
    return series[~series.index.duplicated(keep="last")].sort_index().astype("float32")


def _source_kind(key: str) -> str:
    text = str(key).lower()
    if "fund" in text:
        return "funding"
    if "oi" in text or "open_interest" in text:
        return "open_interest"
    return "oi_funding_interaction"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--perp-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-start-utc", required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_start = pd.Timestamp(args.run_start_utc, tz="UTC")
    keys = [line.strip() for line in args.keys.read_text().splitlines() if line.strip()]
    all_keys = list(dict.fromkeys([*keys, "funding_rate"]))
    files = sorted(args.feature_dir.glob("symbol=*.parquet"))
    rows: list[dict[str, Any]] = []
    changed_files: list[str] = []
    stale_files: list[str] = []
    skipped_cleared = 0
    funding_rewritten = 0
    sidecar_cache: dict[tuple[str, str], pd.Series] = {}
    for path in files:
        symbol = _safe_symbol(path)
        regenerated = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC") >= run_start
        frame, present = _read_feature(path, all_keys)
        funding = sidecar_cache.setdefault(
            (symbol, "funding"),
            _load_sidecar(args.perp_root / "funding_hourly", symbol, "funding_rate"),
        )
        oi = sidecar_cache.setdefault(
            (symbol, "open_interest"),
            _load_sidecar(args.perp_root / "open_interest_hourly", symbol, "open_interest"),
        )
        # A file with no source sidecar is not a sparse observation: it is an
        # unsupported/static legacy product.  Treat it as stale regardless of
        # mtime so a later finalizer invocation cannot re-admit its columns.
        if not regenerated or (not len(funding) and not len(oi)):
            stale_files.append(symbol)
            regenerated = False
        changed = False
        # All skipped keys are invalid supervision/feature outputs, even on a
        # regenerated symbol.  Untouched symbols are cleared for every target
        # key so old pre-repair semantics cannot survive the repair.
        clear = set(all_keys if not regenerated else SKIPPED_KEYS)
        for key in sorted(clear.intersection(frame.columns)):
            if frame[key].notna().any():
                frame[key] = np.nan
                changed = True
                if key in SKIPPED_KEYS:
                    skipped_cleared += 1

        # Rebuild the raw funding alias from the availability-indexed sidecar.
        # Add the alias when a source exists even if the static feature file
        # did not previously contain the column.  Without this branch, a
        # perfectly backfilled product could carry eligibility masks while
        # silently lacking the canonical raw funding input at inference.
        if len(funding) or "funding_rate" in frame.columns:
            aligned = funding.reindex(frame.index).ffill(limit=2)
            frame["funding_rate"] = aligned.to_numpy(dtype=np.float32)
            changed = True
            funding_rewritten += 1

        if changed:
            _atomic_write(frame, path)
            changed_files.append(symbol)

        for key in all_keys:
            if key not in frame.columns:
                rows.append({"symbol": symbol, "feature": key, "present": False, "rows": int(len(frame)), "finite": 0, "coverage": 0.0, "nunique": 0, "std": None, "constant": None, "regenerated": regenerated, "cleared": key in clear, "source_kind": _source_kind(key)})
                continue
            values = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            finite_values = values[finite]
            nunique = int(pd.Series(finite_values).nunique()) if finite_values.size else 0
            std = float(np.std(finite_values)) if finite_values.size else None
            rows.append({"symbol": symbol, "feature": key, "present": True, "rows": int(len(values)), "finite": int(finite.sum()), "coverage": float(finite.mean()) if len(values) else 0.0, "nunique": nunique, "std": std, "constant": bool(nunique <= 1) if finite_values.size else None, "regenerated": regenerated, "cleared": key in clear, "source_kind": _source_kind(key)})

    coverage = pd.DataFrame(rows)
    coverage.to_parquet(args.out_dir / "oi_funding_feature_coverage.parquet", index=False, compression="zstd")
    summary = []
    for key, group in coverage.groupby("feature", sort=True):
        present = group[group["present"]]
        summary.append({
            "feature": key,
            "source_kind": _source_kind(key),
            "static_files": int(len(files)),
            "present_files": int(len(present)),
            "nonconstant_files": int(present["constant"].eq(False).sum()),
            "all_nan_files": int(present["finite"].eq(0).sum()),
            "median_asset_coverage": float(present["coverage"].median()) if len(present) else 0.0,
            "weighted_coverage": float(present["finite"].sum() / max(1, present["rows"].sum())) if len(present) else 0.0,
        })
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_parquet(args.out_dir / "oi_funding_feature_summary.parquet", index=False, compression="zstd")
    payload = {
        "schema": "oi_funding_feature_store_finalization_v1",
        "status": "COMPLETE",
        "feature_dir": str(args.feature_dir),
        "feature_dir_sha256_sample": _sha(files[0]) if files else None,
        "static_files": len(files),
        "regenerated_files": int(sum(pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC") >= run_start for p in files)),
        "stale_files_cleared": stale_files,
        "changed_files": changed_files,
        "affected_keys": all_keys,
        "skipped_keys_cleared_count": skipped_cleared,
        "skipped_keys": sorted(SKIPPED_KEYS),
        "funding_rate_alias_rewritten_files": funding_rewritten,
        "funding_rate_rule": "exact availability-indexed sidecar join at feature timestamp with max 2-bar carry; no extra shift",
        "oi_rule": "availability-indexed dedicated sidecar; feature loader max 3-bar carry; no embedded quote-notional fallback",
        "entry_rule": "feature bar t is completed at t+1h; decision/entry is t+1h at next candle boundary",
        "timing": {
            "feature_to_decision_hours": 1,
            "sidecar_availability_rule": "observation_ts + 1h",
            "extra_funding_shift_bars": 0,
        },
        "coverage_artifact": "oi_funding_feature_coverage.parquet",
        "summary_artifact": "oi_funding_feature_summary.parquet",
    }
    (args.out_dir / "oi_funding_feature_finalization.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": payload["status"], "static_files": len(files), "changed_files": len(changed_files), "stale_files_cleared": len(stale_files), "funding_rate_alias_rewritten": funding_rewritten, "skipped_keys_cleared": skipped_cleared}, sort_keys=True))


if __name__ == "__main__":
    main()
