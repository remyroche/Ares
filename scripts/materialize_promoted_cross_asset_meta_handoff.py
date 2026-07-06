#!/usr/bin/env python3
"""Materialize promoted cross-asset representation features into a meta handoff.

This is a handoff materializer, not a model-training step.  It joins only the
representation columns selected by the V2 promotion artifact, preserving the
OOF/prior-fold contract from the representation runner:

* promoted columns are joined by row keys only;
* no outcome/path columns from the representation prediction file are copied;
* rows without an OOF/prior-fold representation are left as NaN;
* the source ledger is copied unchanged for downstream smoke/replay scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cross_asset_representation_meta_ablation_v2 import (  # noqa: E402
    KEY_COLUMNS,
    REPRESENTATION_COLUMNS,
)


DEFAULT_HANDOFF_DIR = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1"
    "/s52_trailing_regime_meta_handoff_xmarket_v1"
)
DEFAULT_REPRESENTATION_PREDICTIONS = (
    DEFAULT_HANDOFF_DIR
    / "cross_asset_archetype_representation_v1"
    / "cross_asset_representation_v1_predictions.parquet"
)
DEFAULT_PROMOTION_JSON = (
    DEFAULT_HANDOFF_DIR
    / "cross_asset_representation_meta_ablation_v2_conditional_control_v2"
    / "cross_asset_representation_meta_ablation_v2_promotion.json"
)
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_DIR / "train_meta_handoff_promoted_cross_asset_v1"

HANDOFF_NAME = "train_meta_regime_handoff.parquet"
LEDGER_NAME = "s52_trailing_regime_scored_ledger.parquet"
CONTRACT_NAME = "train_meta_regime_handoff_contract.json"
CELL_INTERACTION_SIDE_COL = "side_name"
CELL_INTERACTION_ARCHETYPE_COL = "source_semantic_family"


def _safe_token(value: Any, *, max_len: int = 28) -> str:
    text = str(value).strip().lower()
    token = "".join(ch if ch.isalnum() else "_" for ch in text)
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        token = "missing"
    if len(token) > max_len:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        token = f"{token[: max_len - 9]}_{digest}"
    return token


def _add_cell_interactions(
    frame: pd.DataFrame,
    promoted_cols: list[str],
    *,
    min_cell_rows: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = {CELL_INTERACTION_SIDE_COL, CELL_INTERACTION_ARCHETYPE_COL}
    if not required.issubset(frame.columns) or not promoted_cols:
        return frame, []
    counts = (
        frame.groupby([CELL_INTERACTION_SIDE_COL, CELL_INTERACTION_ARCHETYPE_COL], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    supported = counts[counts >= int(min_cell_rows)]
    if supported.empty:
        return frame, []
    out = frame.copy()
    registry: list[dict[str, Any]] = []
    side_values = out[CELL_INTERACTION_SIDE_COL].astype(str)
    family_values = out[CELL_INTERACTION_ARCHETYPE_COL].astype(str)
    used_names: set[str] = set(out.columns)
    for (side, family), rows in supported.items():
        side_key = str(side)
        family_key = str(family)
        token_base = f"{_safe_token(side_key, max_len=10)}__{_safe_token(family_key, max_len=34)}"
        digest = hashlib.sha1(f"{side_key}|{family_key}".encode("utf-8")).hexdigest()[:8]
        token = f"{token_base}__{digest}"
        mask = side_values.eq(side_key) & family_values.eq(family_key)
        for col in promoted_cols:
            base_name = f"{col}__sxsf__{token}"
            name = base_name
            if name in used_names:
                suffix = 1
                while f"{base_name}_{suffix}" in used_names:
                    suffix += 1
                name = f"{base_name}_{suffix}"
            used_names.add(name)
            values = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)
            out[name] = values.where(mask, 0.0).astype(np.float32)
            registry.append(
                {
                    "interaction_column": name,
                    "base_column": col,
                    "side_name": side_key,
                    "source_semantic_family": family_key,
                    "cell_rows": int(rows),
                    "construction": "base_promoted_oof_score_masked_by_pre_entry_side_x_source_semantic_family",
                }
            )
    return out, registry


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_promotion(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Promotion artifact not found: {path}")
    payload = json.loads(path.read_text())
    promoted = payload.get("promote_to_deeper_meta_eval") or []
    if not promoted:
        raise ValueError(f"Promotion artifact has no promoted representation variants: {path}")
    return payload


def _promoted_columns(promotion: dict[str, Any], *, preferred_variant: str | None) -> tuple[list[str], list[dict[str, Any]]]:
    promoted = list(promotion.get("promote_to_deeper_meta_eval") or [])
    if preferred_variant:
        selected = [item for item in promoted if str(item.get("variant")) == str(preferred_variant)]
        if not selected:
            raise ValueError(f"Preferred variant {preferred_variant!r} is not promoted by the promotion artifact.")
    else:
        selected = promoted
    allowed = set(REPRESENTATION_COLUMNS)
    cols: list[str] = []
    for item in selected:
        for col in item.get("feature_columns") or []:
            if col in allowed and col not in cols:
                cols.append(str(col))
    if not cols:
        raise ValueError("No promoted representation columns survived the safe representation-column filter.")
    return cols, selected


def materialize(
    *,
    handoff_dir: Path,
    representation_predictions: Path,
    promotion_json: Path,
    out_dir: Path,
    preferred_variant: str | None,
    add_cell_interactions: bool = False,
    min_cell_interaction_rows: int = 250,
) -> dict[str, Any]:
    handoff_path = handoff_dir / HANDOFF_NAME
    ledger_path = handoff_dir / LEDGER_NAME
    source_contract_path = handoff_dir / CONTRACT_NAME
    if not handoff_path.exists():
        raise FileNotFoundError(f"Source handoff not found: {handoff_path}")
    if not representation_predictions.exists():
        raise FileNotFoundError(f"Representation predictions not found: {representation_predictions}")
    promotion = _read_promotion(promotion_json)
    promoted_cols, promoted_variants = _promoted_columns(promotion, preferred_variant=preferred_variant)

    handoff = pd.read_parquet(handoff_path)
    rep_cols = list(KEY_COLUMNS) + promoted_cols
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(representation_predictions).names)
    except Exception:
        available = set(pd.read_parquet(representation_predictions).columns)
    missing = [col for col in rep_cols if col not in available]
    if missing:
        raise ValueError(f"Representation prediction file is missing required columns: {missing}")
    reps = pd.read_parquet(representation_predictions, columns=rep_cols)
    reps = reps.drop_duplicates(list(KEY_COLUMNS), keep="last")
    before_cols = set(handoff.columns)
    collisions = [col for col in promoted_cols if col in before_cols]
    if collisions:
        handoff = handoff.drop(columns=collisions)
    materialized = handoff.merge(
        reps,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    cell_interaction_registry: list[dict[str, Any]] = []
    if add_cell_interactions:
        materialized, cell_interaction_registry = _add_cell_interactions(
            materialized,
            promoted_cols,
            min_cell_rows=int(min_cell_interaction_rows),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_handoff = out_dir / HANDOFF_NAME
    materialized.to_parquet(out_handoff, index=False)
    copied_ledger = False
    if ledger_path.exists():
        shutil.copy2(ledger_path, out_dir / LEDGER_NAME)
        copied_ledger = True
    source_contract: dict[str, Any] = {}
    if source_contract_path.exists():
        try:
            source_contract = json.loads(source_contract_path.read_text())
        except Exception:
            source_contract = {"unparsed_source_contract": str(source_contract_path)}
    non_null = materialized.loc[:, promoted_cols].notna().all(axis=1)
    any_non_null = materialized.loc[:, promoted_cols].notna().any(axis=1)
    contract = {
        **source_contract,
        "promoted_cross_asset_representation": {
            "status": "materialized",
            "source_handoff_dir": str(handoff_dir),
            "source_handoff_sha256": _sha256_file(handoff_path),
            "source_ledger_copied": bool(copied_ledger),
            "representation_predictions": str(representation_predictions),
            "representation_predictions_sha256": _sha256_file(representation_predictions),
            "promotion_json": str(promotion_json),
            "promotion_json_sha256": _sha256_file(promotion_json),
            "promotion_status": promotion.get("status"),
            "preferred_variant": preferred_variant,
            "promoted_variants": promoted_variants,
            "promoted_columns": promoted_cols,
            "promoted_column_count": int(len(promoted_cols)),
            "cell_interaction_features": {
                "enabled": bool(add_cell_interactions),
                "side_column": CELL_INTERACTION_SIDE_COL,
                "archetype_column": CELL_INTERACTION_ARCHETYPE_COL,
                "min_cell_rows": int(min_cell_interaction_rows),
                "interaction_column_count": int(len(cell_interaction_registry)),
                "cell_count": int(len({(r["side_name"], r["source_semantic_family"]) for r in cell_interaction_registry})),
                "registry": cell_interaction_registry,
                "leakage_contract": (
                    "Interactions are deterministic masks using pre-entry side_name and "
                    "source_semantic_family only; no outcomes or validation metrics are used."
                ),
            },
            "row_count": int(len(materialized)),
            "rows_with_all_promoted_columns": int(non_null.sum()),
            "rows_with_any_promoted_column": int(any_non_null.sum()),
            "coverage_all_promoted_columns": float(non_null.mean()) if len(materialized) else 0.0,
            "coverage_any_promoted_column": float(any_non_null.mean()) if len(materialized) else 0.0,
            "no_in_sample_backfill": True,
            "missing_representation_policy": "leave_nan_for_rows_without_oof_prior_fold_representation",
            "leakage_contract": (
                "Only promoted OOF/prior-fold representation predictions are joined. "
                "Outcome/path columns from representation predictions are not copied."
            ),
        },
    }
    (out_dir / CONTRACT_NAME).write_text(json.dumps(_json_safe(contract), indent=2, sort_keys=True))
    manifest = {
        "generated_by": "materialize_promoted_cross_asset_meta_handoff",
        "out_dir": str(out_dir),
        "handoff_path": str(out_handoff),
        "contract_path": str(out_dir / CONTRACT_NAME),
        "ledger_path": str(out_dir / LEDGER_NAME) if copied_ledger else None,
        "source_handoff_dir": str(handoff_dir),
        "row_count": int(len(materialized)),
        "input_column_count": int(len(handoff.columns)),
        "output_column_count": int(len(materialized.columns)),
        "promoted_columns": promoted_cols,
        "cell_interaction_column_count": int(len(cell_interaction_registry)),
        "cell_interaction_cell_count": int(len({(r["side_name"], r["source_semantic_family"]) for r in cell_interaction_registry})),
        "rows_with_all_promoted_columns": int(non_null.sum()),
        "rows_with_any_promoted_column": int(any_non_null.sum()),
        "promotion_status": promotion.get("status"),
        "promoted_variants": [item.get("variant") for item in promoted_variants],
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--representation-predictions", type=Path, default=DEFAULT_REPRESENTATION_PREDICTIONS)
    parser.add_argument("--promotion-json", type=Path, default=DEFAULT_PROMOTION_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--preferred-variant",
        default="m1b_cross_lgbm_risk_only_meta",
        help="Promoted variant to materialize. Use empty string to materialize the union of all promoted variants.",
    )
    parser.add_argument(
        "--add-cell-interactions",
        action="store_true",
        help="Add deterministic side x source_semantic_family interaction features for promoted columns.",
    )
    parser.add_argument("--min-cell-interaction-rows", type=int, default=250)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preferred = str(args.preferred_variant).strip() or None
    manifest = materialize(
        handoff_dir=args.handoff_dir,
        representation_predictions=args.representation_predictions,
        promotion_json=args.promotion_json,
        out_dir=args.out_dir,
        preferred_variant=preferred,
        add_cell_interactions=bool(args.add_cell_interactions),
        min_cell_interaction_rows=int(args.min_cell_interaction_rows),
    )
    print(json.dumps(_json_safe({"event": "promoted_cross_asset_meta_handoff_materialized", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
