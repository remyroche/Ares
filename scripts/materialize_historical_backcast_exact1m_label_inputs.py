#!/usr/bin/env python3
"""Build causal research-only label inputs for a historical request stage.

The adapter reconstructs a signal-time Wilder ATR(14) from canonical hourly
OHLCV and emits the exact identity/context/path-target files expected by the
deployed-policy 1m replay.  The ATR alias is simulator plumbing only: it is
not claimed to be bit-exact deployed historical geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
FORBIDDEN_OUTCOME_TOKENS = (
    "execution_net_ev",
    "execution_gross_ev",
    "label_available",
    "label_end",
    "peak_mfe",
    "future_slope",
    "first_touch",
    "realized",
    "realised",
)

FROZEN_BACKCAST_SCOPE = "frozen_backcast_diagnostic"
FROZEN_BACKCAST_EVIDENCE_SCOPE = "frozen_backcast_diagnostic_not_oof"
FROZEN_BACKCAST_LINEAGE = "historical_frozen_backcast_exact1m_research_only"
INVERSE_GRID_SCOPE = "inverse_pi_market_grid_bootstrap_research"
INVERSE_GRID_EVIDENCE_SCOPE = "inverse_pi_market_grid_bootstrap_research_not_oof"
INVERSE_GRID_LINEAGE = "historical_inverse_pi_market_grid_exact1m_research_only"
INVERSE_GRID_POPULATION_LINEAGE = "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1"
INVERSE_GRID_PRODUCT_LINEAGE = "kraken_inverse_pi_exact_product_binding_v1"
INVERSE_CAUSAL_SCOPE = "inverse_pi_market_grid_causal_features_research"
INVERSE_CAUSAL_EVIDENCE_SCOPE = "inverse_pi_market_grid_causal_features_research_not_oof"
INVERSE_CAUSAL_POPULATION_LINEAGE = "jan_jul_2022_inverse_pi_market_grid_causal_features_v1"
INVERSE_CAUSAL_PARENT_ASSIGNMENT = "explicit_deployed_side_parent_inverse_grid"
INVERSE_CAUSAL_PARENT_POLICY_KEYS = {"long": "long__parent", "short": "short__parent"}
INVERSE_CAUSAL_PRODUCTS = {
    "BTC/USD:BTC": "PI_XBTUSD",
    "ETH/USD:ETH": "PI_ETHUSD",
    "LTC/USD:LTC": "PI_LTCUSD",
    "XRP/USD:XRP": "PI_XRPUSD",
    "BCH/USD:BCH": "PI_BCHUSD",
}


def _validated_stage_lineage(
    stage_manifest: dict[str, Any], stage: pd.DataFrame
) -> dict[str, Any]:
    """Validate and carry the immutable population lineage from the stage.

    Historical legacy manifests predate explicit manifest-level lineage fields,
    so their frozen-backcast values are inferred from the staged rows.  An
    alternate PI population, however, must be explicit in both staged rows and
    (for newly written stages) the stage manifest.  This is propagation, never
    a relabeling mechanism.
    """

    source_scopes = set(stage["evidence_scope"].dropna().astype(str))
    stage_lineages = set(stage["lineage"].dropna().astype(str))
    if len(stage_lineages) != 1:
        raise ValueError("stage must carry exactly one immutable lineage")
    stage_lineage = next(iter(stage_lineages))
    if source_scopes == {FROZEN_BACKCAST_SCOPE}:
        contract = {
            "evidence_scope": FROZEN_BACKCAST_EVIDENCE_SCOPE,
            "lineage": FROZEN_BACKCAST_LINEAGE,
            "candidate_population_lineage": None,
            "product_lineage": None,
            "bootstrap_barrier_data_acquisition_only": False,
            "economics_contract": "frozen_or_current_spread_counterfactual_only",
            "economics": "current_frozen_spread_counterfactual",
            "return_unit": "decimal_notional_return",
            "parent_policy_binding": None,
        }
    elif source_scopes == {INVERSE_GRID_SCOPE}:
        required = {
            "candidate_population_lineage",
            "source_product_lineage",
            "source_product_id",
            "source_contract_family",
            "bootstrap_barrier_data_acquisition_only",
        }
        missing = sorted(required - set(stage.columns))
        if missing:
            raise ValueError(f"inverse PI stage is missing lineage columns: {missing}")
        population = set(stage["candidate_population_lineage"].dropna().astype(str))
        product = set(stage["source_product_lineage"].dropna().astype(str))
        contract_family = set(stage["source_contract_family"].dropna().astype(str))
        if population != {INVERSE_GRID_POPULATION_LINEAGE}:
            raise ValueError("inverse PI stage population lineage is invalid")
        if product != {INVERSE_GRID_PRODUCT_LINEAGE} or contract_family != {"PI"}:
            raise ValueError("inverse PI stage product lineage is invalid")
        if not stage["bootstrap_barrier_data_acquisition_only"].fillna(False).astype(bool).all():
            raise ValueError("inverse PI stage bootstrap barrier must be acquisition-only")
        contract = {
            "evidence_scope": INVERSE_GRID_EVIDENCE_SCOPE,
            "lineage": INVERSE_GRID_LINEAGE,
            "candidate_population_lineage": next(iter(population)),
            "product_lineage": next(iter(product)),
            "bootstrap_barrier_data_acquisition_only": True,
            "economics_contract": "frozen_or_current_spread_counterfactual_only",
            "economics": "current_frozen_spread_counterfactual",
            "return_unit": "decimal_notional_return",
            "parent_policy_binding": None,
        }
    elif source_scopes == {INVERSE_CAUSAL_SCOPE}:
        required = {
            "candidate_population_lineage",
            "source_product_lineage",
            "product_id",
            "policy_archetype_assignment_source",
            "bootstrap_barrier_data_acquisition_only",
        }
        missing = sorted(required - set(stage.columns))
        if missing:
            raise ValueError(f"causal inverse PI stage is missing lineage columns: {missing}")
        population = set(stage["candidate_population_lineage"].dropna().astype(str))
        product = set(stage["source_product_lineage"].dropna().astype(str))
        if population != {INVERSE_CAUSAL_POPULATION_LINEAGE}:
            raise ValueError("causal inverse PI stage population lineage is invalid")
        if product != {INVERSE_GRID_PRODUCT_LINEAGE}:
            raise ValueError("causal inverse PI stage product lineage is invalid")
        if stage["bootstrap_barrier_data_acquisition_only"].fillna(True).astype(bool).any():
            raise ValueError("causal inverse PI stage must not carry a bootstrap barrier")
        if set(stage["archetype_policy_key"].dropna().astype(str)) != {"parent"}:
            raise ValueError("causal inverse PI stage must bind parent policy geometry")
        if set(stage["policy_archetype_assignment_source"].dropna().astype(str)) != {
            INVERSE_CAUSAL_PARENT_ASSIGNMENT
        }:
            raise ValueError("causal inverse PI stage parent policy binding is invalid")
        if not stage["side_name"].astype(str).isin(INVERSE_CAUSAL_PARENT_POLICY_KEYS).all():
            raise ValueError("causal inverse PI stage has an invalid policy side")
        expected_products = stage["symbol"].astype(str).map(INVERSE_CAUSAL_PRODUCTS)
        if expected_products.isna().any() or not stage["product_id"].astype(str).eq(expected_products).all():
            raise ValueError("causal inverse PI stage product binding is invalid")
        contract = {
            "evidence_scope": INVERSE_CAUSAL_EVIDENCE_SCOPE,
            "lineage": INVERSE_GRID_LINEAGE,
            "candidate_population_lineage": INVERSE_CAUSAL_POPULATION_LINEAGE,
            "product_lineage": INVERSE_GRID_PRODUCT_LINEAGE,
            "bootstrap_barrier_data_acquisition_only": False,
            "economics_contract": "inverse_quote_notional_current_spread_counterfactual_only",
            "economics": "inverse_quote_notional_current_spread_counterfactual",
            "return_unit": "quote_notional_price_return_not_inverse_collateral_roe",
            "parent_policy_binding": {
                "assignment_source": INVERSE_CAUSAL_PARENT_ASSIGNMENT,
                "archetype_policy_key": "parent",
                "side_policy_keys": INVERSE_CAUSAL_PARENT_POLICY_KEYS,
            },
        }
    else:
        raise ValueError(f"unexpected stage evidence scope: {sorted(source_scopes)}")
    if stage_lineage != contract["lineage"]:
        raise ValueError("stage row lineage disagrees with its source population")
    # New stages record this in their manifest; tolerate its absence only for
    # legacy frozen-backcast stages so old immutable artifacts remain readable.
    for key in (
        "evidence_scope",
        "lineage",
        "candidate_population_lineage",
        "product_lineage",
        "bootstrap_barrier_data_acquisition_only",
        "economics_contract",
        "return_unit",
        "parent_policy_binding",
    ):
        if key in stage_manifest and stage_manifest[key] != contract[key]:
            raise ValueError(f"stage manifest {key} disagrees with staged lineage")
    return contract


def wilder_atr_fraction(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    period: int = 14,
) -> np.ndarray:
    """Return causal Wilder ATR divided by the contemporaneous close.

    This deliberately stays local to the historical adapter.  Importing the
    Pack-B materializer would also import its AE/GMM training stack, which is
    unrelated to label construction and makes this research artifact depend
    on optional Numba model-training state.
    """

    high_values = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    close_values = np.asarray(close, dtype=np.float64)
    if (
        high_values.ndim != 1
        or high_values.shape != low_values.shape
        or high_values.shape != close_values.shape
        or int(period) < 2
    ):
        raise ValueError("invalid Wilder ATR inputs")
    tr = np.full(len(close_values), np.nan, dtype=np.float64)
    if len(tr):
        tr[0] = high_values[0] - low_values[0]
    if len(tr) > 1:
        previous_close = close_values[:-1]
        tr[1:] = np.maximum(
            high_values[1:] - low_values[1:],
            np.maximum(
                np.abs(high_values[1:] - previous_close),
                np.abs(low_values[1:] - previous_close),
            ),
        )
    atr = (
        pd.Series(tr)
        .ewm(alpha=1.0 / float(period), adjust=False, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    return np.divide(
        atr,
        close_values,
        out=np.full(len(close_values), np.nan, dtype=np.float64),
        where=np.isfinite(close_values) & (close_values > 0.0),
    ).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_parts(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp):
    safe = symbol.replace("/", "_")
    directory = root / "ohlcv" / f"symbol={safe}"
    return [
        part
        for year in range(int(start.year), int(end.year) + 1)
        for part in sorted((directory / f"year={year}").glob("*.parquet"))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--product-map-manifest", type=Path, required=True)
    parser.add_argument(
        "--hourly-root",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures"),
    )
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--warmup-days", type=int, default=90)
    args = parser.parse_args()
    if args.atr_period < 2 or args.warmup_days < 1:
        raise ValueError("ATR period and warmup days must be positive")

    stage_manifest_path = args.stage_dir / "manifest.json"
    staged_path = args.stage_dir / "staged_candidates.parquet"
    stage_manifest = _json(stage_manifest_path)
    if stage_manifest.get("schema") != "historical_backcast_exact1m_request_stage_v2":
        raise ValueError("stage schema must be historical_backcast_exact1m_request_stage_v2")
    expected_stage_hash = (
        stage_manifest.get("outputs", {})
        .get("staged_candidates", {})
        .get("sha256")
    )
    if _sha256(staged_path) != expected_stage_hash:
        raise ValueError("stage manifest does not bind staged candidate bytes")
    product_manifest = _json(args.product_map_manifest)
    if product_manifest.get("schema") != "kraken_historical_product_map_v1":
        raise ValueError("product map manifest schema is invalid")
    if (
        product_manifest.get("stage_candidates", {}).get("sha256")
        != expected_stage_hash
    ):
        raise ValueError("product map is not bound to this exact request stage")

    stage = pd.read_parquet(staged_path)
    forbidden = sorted(
        column
        for column in stage.columns
        if any(token in column.lower() for token in FORBIDDEN_OUTCOME_TOKENS)
    )
    if forbidden:
        raise ValueError(f"stage contains forbidden outcome columns: {forbidden}")
    required = {
        "candidate_id",
        "signal_timestamp",
        "decision_timestamp",
        "path_end_exclusive",
        "symbol",
        "side_name",
        "archetype_policy_key",
        "__barrier_pct__",
        "evidence_scope",
        "lineage",
        "execution_parity_claim",
        "promotion_eligible",
    }
    missing = sorted(required - set(stage.columns))
    if missing:
        raise ValueError(f"stage missing required columns: {missing}")
    for column in ("signal_timestamp", "decision_timestamp", "path_end_exclusive"):
        stage[column] = pd.to_datetime(stage[column], utc=True, errors="raise")
    if stage["candidate_id"].duplicated().any():
        raise ValueError("stage has duplicate candidate IDs")
    if not (
        stage["decision_timestamp"] - stage["signal_timestamp"]
    ).eq(pd.Timedelta(hours=1)).all():
        raise ValueError("decision timestamps are not exactly signal + 1 hour")
    if not (
        stage["path_end_exclusive"] - stage["decision_timestamp"]
    ).eq(pd.Timedelta(hours=12)).all():
        raise ValueError("path windows are not exactly 12 hours")
    lineage_contract = _validated_stage_lineage(stage_manifest, stage)
    if stage["execution_parity_claim"].astype(bool).any():
        raise ValueError("historical request stage may not claim execution parity")
    if stage["promotion_eligible"].astype(bool).any():
        raise ValueError("historical request stage may not be promotion eligible")

    hourly_store = PartitionedOHLCVStore(str(args.hourly_root), timeframe="1h")
    atr_values = np.full(len(stage), np.nan, dtype=np.float32)
    uninterrupted_90d = np.zeros(len(stage), dtype=bool)
    source_hashes: dict[str, str] = {}
    for symbol, row_index in stage.groupby("symbol", sort=True).groups.items():
        positions = np.asarray(list(row_index), dtype=np.int64)
        signals = stage.loc[positions, "signal_timestamp"]
        read_start = signals.min() - pd.Timedelta(days=int(args.warmup_days))
        read_end = signals.max()
        bars = hourly_store.load(
            str(symbol),
            columns=["high", "low", "close"],
            start_ts=read_start,
            end_ts=read_end,
        )
        if bars.empty:
            raise ValueError(f"{symbol}: canonical hourly OHLCV is unavailable")
        bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
        bars = bars.loc[~bars.index.isna(), ["high", "low", "close"]].sort_index()
        if bars.index.duplicated().any():
            raise ValueError(f"{symbol}: duplicate hourly timestamps")
        numeric = bars.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or (values <= 0.0).any()
            or (values[:, 0] < values[:, 1]).any()
        ):
            raise ValueError(f"{symbol}: invalid hourly OHLC values")
        atr = wilder_atr_fraction(
            numeric["high"].to_numpy(dtype=np.float64),
            numeric["low"].to_numpy(dtype=np.float64),
            numeric["close"].to_numpy(dtype=np.float64),
            period=int(args.atr_period),
        )
        aligned = pd.Series(atr, index=bars.index).reindex(
            pd.DatetimeIndex(signals)
        )
        if (
            aligned.isna().any()
            or not np.isfinite(aligned.to_numpy(dtype=float)).all()
            or (aligned.to_numpy(dtype=float) <= 0.0).any()
        ):
            raise ValueError(f"{symbol}: exact signal-time ATR is incomplete")
        atr_values[positions] = aligned.to_numpy(dtype=np.float32)

        bar_ns = bars.index.asi8
        signal_ns = pd.DatetimeIndex(signals).asi8
        left_ns = signal_ns - int(pd.Timedelta(days=args.warmup_days).value)
        left = np.searchsorted(bar_ns, left_ns, side="left")
        right = np.searchsorted(bar_ns, signal_ns, side="right")
        expected = int(args.warmup_days) * 24 + 1
        uninterrupted_90d[positions] = (right - left) == expected
        for part in _source_parts(args.hourly_root, str(symbol), read_start, read_end):
            source_hashes[str(part.resolve())] = _sha256(part)

    if not np.isfinite(atr_values).all() or (atr_values <= 0.0).any():
        raise ValueError("causal ATR reconstruction is incomplete")

    identity = pd.DataFrame(
        {
            "__ts__": stage["signal_timestamp"],
            "__symbol__": stage["symbol"].astype(str),
            "side_name": stage["side_name"].astype(str),
            "candidate_id": stage["candidate_id"].astype(str),
        }
    )
    context = identity.copy()
    # The deployed policy's local geometry keys were frozen from the observable
    # ``side__archetype`` value.  Passing only the raw archetype silently
    # normalizes to a different key and sends every row to side-parent geometry.
    context["policy_archetype"] = (
        stage["side_name"].astype(str)
        + "__"
        + stage["archetype_policy_key"].astype(str)
    )
    path_targets = identity.copy()
    path_targets["__barrier_pct__"] = pd.to_numeric(
        stage["__barrier_pct__"], errors="raise"
    ).astype(np.float32)
    path_targets["__historical_backcast_wilder14_atr_fraction__"] = atr_values
    path_targets["__path_auxiliary_atr_fraction__"] = atr_values
    path_targets["atr_available_at"] = stage["signal_timestamp"]
    path_targets["atr_90d_uninterrupted_history"] = uninterrupted_90d

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidates.parquet"
    context_path = output / "context.parquet"
    targets_path = output / "path_targets.parquet"
    identity.to_parquet(candidates_path, index=False)
    context.to_parquet(context_path, index=False)
    path_targets.to_parquet(targets_path, index=False)
    manifest = {
        "schema": "historical_backcast_exact1m_label_inputs_v1",
        "status": "causal_label_inputs_materialized",
        "rows": int(len(stage)),
        "symbols": int(stage["symbol"].nunique()),
        "evidence_scope": lineage_contract["evidence_scope"],
        "lineage": lineage_contract["lineage"],
        "candidate_population_lineage": lineage_contract[
            "candidate_population_lineage"
        ],
        "product_lineage": lineage_contract["product_lineage"],
        "bootstrap_barrier_data_acquisition_only": lineage_contract[
            "bootstrap_barrier_data_acquisition_only"
        ],
        "oof_status": "not_oof",
        "promotion_eligible": False,
        "execution_parity_claim": False,
        "economics": lineage_contract["economics"],
        "return_unit": lineage_contract["return_unit"],
        "parent_policy_binding": lineage_contract["parent_policy_binding"],
        "historical_l2_spread_available": False,
        "atr_contract": (
            "causal historical Wilder ATR14 diagnostic reconstruction; "
            "simulator alias only; not bit-exact deployed ATR geometry"
        ),
        "atr_period": int(args.atr_period),
        "atr_warmup_days_loaded": int(args.warmup_days),
        "atr_exact_signal_rows": int(np.isfinite(atr_values).sum()),
        "atr_90d_uninterrupted_rows": int(uninterrupted_90d.sum()),
        "atr_90d_uninterrupted_fraction": float(uninterrupted_90d.mean()),
        "decision_to_path": "[signal+1h, signal+1h+12h)",
        "same_minute_conflict": "conservative_adverse_first_required",
        "stage_manifest": {
            "path": str(stage_manifest_path.resolve()),
            "sha256": _sha256(stage_manifest_path),
        },
        "product_map_manifest": {
            "path": str(args.product_map_manifest.resolve()),
            "sha256": _sha256(args.product_map_manifest),
        },
        "policy_json": {
            "path": str(args.policy_json.resolve()),
            "sha256": _sha256(args.policy_json),
        },
        "hourly_source_parts": {
            "count": int(len(source_hashes)),
            "sha256_by_path": source_hashes,
        },
        "outputs": {
            "candidates": {
                "path": str(candidates_path.resolve()),
                "sha256": _sha256(candidates_path),
            },
            "context": {
                "path": str(context_path.resolve()),
                "sha256": _sha256(context_path),
            },
            "path_targets": {
                "path": str(targets_path.resolve()),
                "sha256": _sha256(targets_path),
            },
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
