#!/usr/bin/env python3
"""Attach archetype fields to replay-ready simple-policy candidate rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _attach_policy_archetype_column,
    _json_safe,
)


DEFAULT_CANDIDATES = Path(
    "data_perp/reports/ae_gmm_archetype_ablation_existing_source_export_20260704_v1/"
    "g5_meta_side_arch_local_cap15/gmm_train_meta_path_filter_simple_policy_candidates.parquet"
)
DEFAULT_ARCHETYPE_LEDGER = Path(
    "data_perp/reports/ae_gmm_archetype_ablation_existing_source_export_20260704_v1/"
    "g5_meta_side_arch_local_cap15/gmm_train_meta_path_filter_smoke_selected_rows.parquet"
)
DEFAULT_OUT_DIR = Path("data_perp/reports/replay_candidates_side_archetype_materialized")

REPLAY_REQUIRED_COLUMNS = {
    "timestamp",
    "symbol",
    "side",
    "strategy_id",
    "rank_pct",
    "calibrated_score",
    "barrier_pct",
    "base_strategy_threshold",
}
DEFAULT_JOIN_KEYS = (
    "timestamp",
    "symbol",
    "side",
    "meta_variant",
    "meta_score_rank_pct",
)
ARCHETYPE_COLUMNS = (
    "local_side_archetype",
    "local_archetype_support",
    "local_archetype_quality",
    "local_archetype_quality_rank",
    "local_archetype_bad_prior",
    "local_archetype_timeout_prior",
    "local_archetype_mean_u_prior",
    "archetype",
    "policy_archetype",
    "policy_archetype_source",
)
DEFAULT_DEDUPE_KEYS = ("timestamp", "symbol", "strategy_id")
DEFAULT_DEDUPE_SORT_COLUMNS = (
    "meta_score_rank_pct_selected",
    "meta_score_rank_pct",
    "calibrated_score",
    "rank_pct",
)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _normalise_join_key(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for key in keys:
        if key not in out.columns:
            continue
        if key == "timestamp":
            out[key] = pd.to_datetime(out[key], utc=True, errors="coerce")
        elif key == "side":
            side = pd.to_numeric(out[key], errors="coerce")
            text = out[key].astype(str).str.lower()
            out[key] = np.where(
                side < 0.0,
                "short",
                np.where(side > 0.0, "long", np.where(text.str.startswith("short"), "short", "long")),
            )
        elif pd.api.types.is_numeric_dtype(out[key]):
            out[key] = pd.to_numeric(out[key], errors="coerce").round(12)
        else:
            out[key] = out[key].astype(str)
    return out


def _available_join_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    requested: Sequence[str],
) -> List[str]:
    keys = [str(k) for k in requested if str(k) in left.columns and str(k) in right.columns]
    if not keys:
        raise ValueError("No requested join keys are available in both candidate tables")
    return keys


def _validate_unique(frame: pd.DataFrame, keys: Sequence[str], *, name: str) -> None:
    duplicate_count = int(frame.duplicated(list(keys)).sum())
    if duplicate_count:
        raise ValueError(f"{name} has {duplicate_count} duplicate rows on join keys {list(keys)}")


def _join_archetype_ledger(
    candidates: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    join_keys: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    keys = _available_join_keys(candidates, ledger, join_keys)
    left = _normalise_join_key(candidates, keys)
    right = _normalise_join_key(ledger, keys)
    _validate_unique(left[keys], keys, name="candidates")
    _validate_unique(right[keys], keys, name="archetype_ledger")

    cols = list(keys)
    cols.extend(
        col
        for col in ARCHETYPE_COLUMNS
        if col in right.columns and col not in cols
    )
    right = right[cols].copy()
    joined = left.merge(right, on=list(keys), how="left", validate="one_to_one")
    matched = joined[[col for col in ARCHETYPE_COLUMNS if col in joined.columns]].notna().any(axis=1)
    report = {
        "join_keys": list(keys),
        "ledger_rows": int(len(ledger)),
        "candidate_rows": int(len(candidates)),
        "matched_rows": int(matched.sum()),
        "matched_fraction": float(matched.mean()) if len(joined) else 0.0,
    }
    return joined, report


def _materialize_policy_archetype(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "local_side_archetype" in out.columns:
        labels = out["local_side_archetype"].astype("string")
        out["policy_archetype"] = labels.fillna("missing").astype(str)
        out["policy_archetype_source"] = "local_side_archetype"
        return out
    return _attach_policy_archetype_column(out, strategy_id="__candidate_table__")


def _deduplicate_decisions(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
    sort_columns: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    available_keys = [key for key in keys if key in frame.columns]
    if not available_keys:
        return frame, {"enabled": False, "reason": "no_available_keys"}
    work = frame.copy()
    for key in available_keys:
        if key == "timestamp":
            work[key] = pd.to_datetime(work[key], utc=True, errors="coerce")
        else:
            work[key] = work[key].astype(str)
    before = int(len(work))
    duplicate_rows = int(work.duplicated(available_keys).sum())
    if duplicate_rows == 0:
        return work, {
            "enabled": True,
            "keys": list(available_keys),
            "before_rows": before,
            "after_rows": before,
            "duplicate_rows": 0,
            "removed_rows": 0,
        }
    available_sort = [col for col in sort_columns if col in work.columns]
    for col in available_sort:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if available_sort:
        work = work.sort_values(
            [*available_keys, *available_sort],
            ascending=[True] * len(available_keys) + [False] * len(available_sort),
            kind="mergesort",
        )
    else:
        work = work.sort_values(available_keys, kind="mergesort")
    deduped = work.drop_duplicates(available_keys, keep="first").reset_index(drop=True)
    return deduped, {
        "enabled": True,
        "keys": list(available_keys),
        "sort_columns": available_sort,
        "before_rows": before,
        "after_rows": int(len(deduped)),
        "duplicate_rows": duplicate_rows,
        "removed_rows": int(before - len(deduped)),
    }


def _summary(frame: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "policy_archetype_present": bool("policy_archetype" in frame.columns),
        "policy_archetype_source": (
            str(frame["policy_archetype_source"].dropna().iloc[0])
            if "policy_archetype_source" in frame.columns
            and frame["policy_archetype_source"].notna().any()
            else None
        ),
    }
    if "policy_archetype" in frame.columns:
        counts = frame["policy_archetype"].astype(str).value_counts()
        out["policy_archetype_count"] = int(counts.size)
        out["policy_archetype_counts"] = {
            str(key): int(value) for key, value in counts.head(50).items()
        }
    if {"side", "policy_archetype"}.issubset(frame.columns):
        work = frame.copy()
        side_num = pd.to_numeric(work["side"], errors="coerce")
        side_text = work["side"].astype(str).str.lower()
        work["side_label"] = np.where(
            side_num < 0.0,
            "short",
            np.where(
                side_num > 0.0,
                "long",
                np.where(side_text.str.startswith("short"), "short", "long"),
            ),
        )
        counts = work.groupby(["side_label", "policy_archetype"], dropna=False).size()
        out["side_archetype_counts"] = {
            f"{str(side)}|{str(arch)}": int(value)
            for (side, arch), value in counts.sort_values(ascending=False).head(100).items()
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--archetype-ledger", type=Path, default=DEFAULT_ARCHETYPE_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--join-keys",
        default=",".join(DEFAULT_JOIN_KEYS),
        help="Comma-separated keys used to join an optional archetype ledger.",
    )
    parser.add_argument(
        "--dedupe-decision-keys",
        default=",".join(DEFAULT_DEDUPE_KEYS),
        help="Comma-separated keys used to freeze one replay row per decision.",
    )
    parser.add_argument(
        "--dedupe-sort-columns",
        default=",".join(DEFAULT_DEDUPE_SORT_COLUMNS),
        help="Descending score columns used to choose the retained duplicate row.",
    )
    args = parser.parse_args()

    candidates = _read(args.candidates)
    missing = sorted(REPLAY_REQUIRED_COLUMNS.difference(candidates.columns))
    if missing:
        raise ValueError(f"{args.candidates} is not replay-ready; missing {missing}")

    join_report: Dict[str, Any] = {"enabled": False}
    if args.archetype_ledger and args.archetype_ledger.exists():
        ledger = _read(args.archetype_ledger)
        candidates, join_report = _join_archetype_ledger(
            candidates,
            ledger,
            join_keys=[part.strip() for part in str(args.join_keys).split(",") if part.strip()],
        )
        join_report["enabled"] = True
        if float(join_report["matched_fraction"]) < 0.99:
            raise ValueError(
                "Archetype ledger join coverage is too low: "
                f"{join_report['matched_fraction']:.2%}"
            )

    candidates = _materialize_policy_archetype(candidates)
    candidates, dedupe_report = _deduplicate_decisions(
        candidates,
        keys=[
            part.strip()
            for part in str(args.dedupe_decision_keys).split(",")
            if part.strip()
        ],
        sort_columns=[
            part.strip()
            for part in str(args.dedupe_sort_columns).split(",")
            if part.strip()
        ],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "simple_policy_candidates_with_archetypes.parquet"
    candidates.to_parquet(out_path, index=False)
    manifest = {
        "generated_by": "materialize_replay_candidates_with_archetypes",
        "source_candidates": str(args.candidates),
        "source_archetype_ledger": str(args.archetype_ledger),
        "output_candidates": str(out_path),
        "join_report": join_report,
        "dedupe_report": dedupe_report,
        "summary": _summary(candidates),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2))
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
