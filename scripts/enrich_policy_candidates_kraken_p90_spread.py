#!/usr/bin/env python3
"""Join a frozen per-symbol Kraken p90 spread map onto policy candidates.

This script is intentionally an enrichment boundary.  It does not rank, filter,
calibrate, admit, size, or cost candidates.  In particular, it never deducts a
fee or spread from an EV/return column.  Downstream replay owns that arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


SCHEMA = "kraken_p90_spread_candidate_enrichment_v1"
DEFAULT_SPREAD_COLUMN = "p90_spread_bps"
OUTPUT_SPREAD_BPS_COLUMN = "kraken_p90_spread_bps"
OUTPUT_SPREAD_RETURN_COLUMN = "kraken_p90_spread_return"
OUTPUT_MAPPING_HASH_COLUMN = "kraken_p90_spread_mapping_sha256"
OUTPUT_POLICY_SPREAD_BPS_COLUMN = "expected_spread_bps"
OUTPUT_COMPAT_P90_SPREAD_BPS_COLUMN = "p90_spread_bps"

DuplicatePolicy = Literal["reject", "require_equal", "first", "last"]
MissingPolicy = Literal["reject", "allow_null"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_kraken_symbol(value: object) -> str:
    """Normalize harmless separator/case differences without changing market identity.

    ``AAA/USD:USD`` and ``aaa_usd:usd`` are the same Kraken perpetual.  Spot
    ``AAA/USD`` remains distinct from the perpetual on purpose: treating the two
    as interchangeable would hide an ambiguous spread source.
    """

    raw = str(value or "").strip().upper().replace(" ", "")
    if not raw:
        return ""
    return raw.replace("_", "/").replace("-", "/")


def _select_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: tuple[str, ...],
    source: str,
) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"{source} does not contain requested column '{requested}'")
        return requested
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"{source} requires one of {list(candidates)} or an explicit column")


def _select_spread_symbol_column(
    frame: pd.DataFrame,
    requested: str | None,
    spread_column: str,
) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"eligible spread CSV does not contain requested column '{requested}'")
        return requested
    for column in ("symbol", "__symbol__", "instrument"):
        if column in frame.columns:
            return column
    remaining = [column for column in frame.columns if column != spread_column]
    if len(remaining) == 1:
        return remaining[0]
    raise ValueError(
        "eligible spread CSV needs a symbol column; pass --spread-symbol-column "
        f"(available columns: {list(frame.columns)})"
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    """Hash existing candidate values and order without coercing their dtypes."""

    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("|".join(map(str, frame.dtypes)).encode("utf-8"))
    # hash_pandas_object is deterministic for the unchanged input frame and
    # preserves row order.  We retain the source file hash separately as well.
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=False)
    digest.update(hashed.to_numpy(dtype=np.uint64, copy=False).tobytes())
    return digest.hexdigest()


def _mapping_hash(mapping: pd.DataFrame) -> str:
    ordered = mapping.sort_values("normalized_symbol", kind="stable")
    payload = "\n".join(
        f"{symbol}\t{spread:.17g}"
        for symbol, spread in zip(
            ordered["normalized_symbol"].astype(str),
            ordered[DEFAULT_SPREAD_COLUMN].to_numpy(dtype=np.float64, copy=False),
        )
    )
    return _sha256_text(payload)


def build_spread_mapping(
    eligible_spreads: pd.DataFrame,
    *,
    symbol_column: str | None = None,
    spread_column: str = DEFAULT_SPREAD_COLUMN,
    duplicate_policy: DuplicatePolicy = "reject",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and collapse an explicit spread table into one normalized row/symbol."""

    if spread_column not in eligible_spreads.columns:
        raise ValueError(
            f"eligible spread CSV does not contain '{spread_column}' "
            f"(available columns: {list(eligible_spreads.columns)})"
        )
    if duplicate_policy not in {"reject", "require_equal", "first", "last"}:
        raise ValueError(f"unsupported duplicate policy: {duplicate_policy}")

    source_symbol = _select_spread_symbol_column(eligible_spreads, symbol_column, spread_column)
    mapping = eligible_spreads.loc[:, [source_symbol, spread_column]].copy()
    mapping.columns = ["source_symbol", DEFAULT_SPREAD_COLUMN]
    mapping["source_symbol"] = mapping["source_symbol"].astype(str)
    mapping["normalized_symbol"] = mapping["source_symbol"].map(normalize_kraken_symbol)
    mapping[DEFAULT_SPREAD_COLUMN] = pd.to_numeric(
        mapping[DEFAULT_SPREAD_COLUMN], errors="coerce"
    ).astype(np.float64)

    invalid = (
        mapping["normalized_symbol"].eq("")
        | ~np.isfinite(mapping[DEFAULT_SPREAD_COLUMN].to_numpy(dtype=np.float64, copy=False))
        | mapping[DEFAULT_SPREAD_COLUMN].lt(0.0)
    )
    if invalid.any():
        sample = mapping.loc[invalid, ["source_symbol", DEFAULT_SPREAD_COLUMN]].head(5)
        raise ValueError(f"eligible spread CSV has invalid symbol/spread rows: {sample.to_dict('records')}")

    duplicate = mapping["normalized_symbol"].duplicated(keep=False)
    duplicate_rows = int(duplicate.sum())
    if duplicate.any():
        ambiguous = mapping.loc[duplicate].groupby("normalized_symbol", sort=False)[DEFAULT_SPREAD_COLUMN].nunique(
            dropna=False
        )
        if duplicate_policy == "reject":
            sample = mapping.loc[duplicate, ["source_symbol", "normalized_symbol", DEFAULT_SPREAD_COLUMN]].head(10)
            raise ValueError(
                "eligible spread CSV has duplicate/ambiguous normalized symbol mappings; "
                "set --duplicate-policy explicitly to allow a defined resolution: "
                f"{sample.to_dict('records')}"
            )
        if duplicate_policy == "require_equal" and ambiguous.gt(1).any():
            bad = ambiguous.loc[ambiguous.gt(1)].index.tolist()[:10]
            raise ValueError(
                "duplicate normalized symbol mappings disagree on p90 spread: "
                f"{bad}; require_equal only permits identical values"
            )
        keep = "first" if duplicate_policy in {"require_equal", "first"} else "last"
        mapping = mapping.drop_duplicates("normalized_symbol", keep=keep)

    mapping = mapping.reset_index(drop=True)
    mapping_sha256 = _mapping_hash(mapping)
    return mapping, {
        "source_symbol_column": source_symbol,
        "source_spread_column": spread_column,
        "duplicate_policy": duplicate_policy,
        "source_rows": int(len(eligible_spreads)),
        "mapping_rows": int(len(mapping)),
        "duplicate_source_rows": duplicate_rows,
        "mapping_sha256": mapping_sha256,
    }


def enrich_candidates(
    candidates: pd.DataFrame,
    spread_mapping: pd.DataFrame,
    *,
    candidate_symbol_column: str | None = None,
    missing_policy: MissingPolicy = "reject",
    mapping_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a same-order candidate frame with only p90-spread provenance added."""

    if missing_policy not in {"reject", "allow_null"}:
        raise ValueError(f"unsupported missing policy: {missing_policy}")
    for column in ("normalized_symbol", DEFAULT_SPREAD_COLUMN):
        if column not in spread_mapping.columns:
            raise ValueError(f"spread mapping is missing required column '{column}'")
    if spread_mapping["normalized_symbol"].duplicated().any():
        raise ValueError("spread mapping has duplicate normalized symbols")

    symbol_column = _select_column(
        candidates,
        candidate_symbol_column,
        ("symbol", "__symbol__", "instrument"),
        "candidate parquet",
    )
    protected = candidates.copy(deep=True)
    protected_hash_before = _frame_hash(protected)
    normalized = candidates[symbol_column].map(normalize_kraken_symbol)
    lookup = spread_mapping.set_index("normalized_symbol")[DEFAULT_SPREAD_COLUMN]
    spread_bps = normalized.map(lookup).astype(np.float64)
    missing = normalized.eq("") | ~np.isfinite(spread_bps.to_numpy(dtype=np.float64, copy=False))
    if missing.any() and missing_policy == "reject":
        examples = candidates.loc[missing, symbol_column].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{int(missing.sum())} candidate rows have no unambiguous p90 spread mapping; "
            f"examples: {examples}. Set --missing-policy allow_null only explicitly."
        )

    output = candidates.copy(deep=True)
    output[OUTPUT_SPREAD_BPS_COLUMN] = spread_bps
    output[OUTPUT_SPREAD_RETURN_COLUMN] = spread_bps / 10_000.0
    output[OUTPUT_MAPPING_HASH_COLUMN] = str(mapping_sha256 or _mapping_hash(spread_mapping))
    canonical_columns_added: list[str] = []
    for canonical_column in (
        OUTPUT_POLICY_SPREAD_BPS_COLUMN,
        OUTPUT_COMPAT_P90_SPREAD_BPS_COLUMN,
    ):
        if canonical_column in candidates.columns:
            existing = pd.to_numeric(candidates[canonical_column], errors="coerce").to_numpy(
                dtype=np.float64,
                copy=False,
            )
            expected = spread_bps.to_numpy(dtype=np.float64, copy=False)
            comparable = np.isfinite(existing) & np.isfinite(expected)
            if not bool(comparable.all()) or not bool(
                np.allclose(existing, expected, rtol=0.0, atol=1e-9)
            ):
                raise ValueError(
                    f"candidate column {canonical_column!r} conflicts with the "
                    "explicit Kraken p90 spread mapping"
                )
            continue
        output[canonical_column] = spread_bps
        canonical_columns_added.append(canonical_column)
    protected_hash_after = _frame_hash(output.loc[:, protected.columns])
    if protected_hash_before != protected_hash_after or not output.loc[:, protected.columns].equals(protected):
        raise RuntimeError("candidate enrichment modified an existing candidate column")

    return output, {
        "candidate_symbol_column": symbol_column,
        "missing_policy": missing_policy,
        "rows": int(len(output)),
        "missing_spread_rows": int(missing.sum()),
        "protected_columns": list(protected.columns),
        "protected_columns_sha256_before": protected_hash_before,
        "protected_columns_sha256_after": protected_hash_after,
        "row_order_preserved": bool(output.index.equals(candidates.index)),
        "canonical_optimizer_spread_column": OUTPUT_POLICY_SPREAD_BPS_COLUMN,
        "canonical_columns_added": canonical_columns_added,
        "costs_applied": False,
        "cost_contract": "enrichment_only_no_fee_or_spread_deduction",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Input policy candidate parquet.")
    parser.add_argument("--eligible-spreads", type=Path, required=True, help="Explicit Kraken eligible-symbol/p90-spread CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Enriched candidate parquet.")
    parser.add_argument("--manifest", type=Path, help="Output lineage manifest JSON. Defaults beside --output.")
    parser.add_argument("--candidate-symbol-column")
    parser.add_argument("--spread-symbol-column")
    parser.add_argument("--spread-column", default=DEFAULT_SPREAD_COLUMN)
    parser.add_argument(
        "--duplicate-policy",
        choices=("reject", "require_equal", "first", "last"),
        default="reject",
        help="Explicit resolution for duplicate normalized source symbols; default rejects.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=("reject", "allow_null"),
        default="reject",
        help="Explicit behavior for candidate symbols absent from the source map; default rejects.",
    )
    args = parser.parse_args(argv)

    candidates = pd.read_parquet(args.candidates)
    eligible_spreads = pd.read_csv(args.eligible_spreads)
    mapping, mapping_lineage = build_spread_mapping(
        eligible_spreads,
        symbol_column=args.spread_symbol_column,
        spread_column=args.spread_column,
        duplicate_policy=args.duplicate_policy,
    )
    output, enrichment = enrich_candidates(
        candidates,
        mapping,
        candidate_symbol_column=args.candidate_symbol_column,
        missing_policy=args.missing_policy,
        mapping_sha256=mapping_lineage["mapping_sha256"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "candidates": str(args.candidates),
        "eligible_spreads": str(args.eligible_spreads),
        "output": str(args.output),
        "candidate_input_sha256": _sha256_file(args.candidates),
        "eligible_spreads_input_sha256": _sha256_file(args.eligible_spreads),
        "output_sha256": _sha256_file(args.output),
        "added_columns": [
            OUTPUT_SPREAD_BPS_COLUMN,
            OUTPUT_SPREAD_RETURN_COLUMN,
            OUTPUT_MAPPING_HASH_COLUMN,
            *enrichment["canonical_columns_added"],
        ],
        "mapping": mapping_lineage,
        "enrichment": enrichment,
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(manifest_path),
                "rows": enrichment["rows"],
                "missing_spread_rows": enrichment["missing_spread_rows"],
                "mapping_sha256": mapping_lineage["mapping_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
