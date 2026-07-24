#!/usr/bin/env python3
"""Materialize the canonical pre-entry Pack-B downstream context.

The context is derived only from the strict side-local base outer-OOF stream.
It supplies the five auxiliary heads with the same base score, rank, margin,
and trainable base-archetype context while excluding every realised outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (  # noqa: E402
    candidate_identity_sha256,
)
from scripts.run_packb_pre_march_side_fs_hpo import _git_revision  # noqa: E402

SCHEMA = "packb_downstream_preentry_context_v1"
DEFAULT_TOP40 = (
    ROOT / "data_perp/artifacts/packb_side_local_top40_20260724_v1_31_8/"
    "base_candidate_population.parquet"
)
DEFAULT_TOP40_MANIFEST = DEFAULT_TOP40.with_name("manifest.json")
DEFAULT_OUTER = (
    ROOT / "data_perp/artifacts/packb_side_local_outer_oof_20260724_v1_31_8/"
    "oof_predictions.parquet"
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/packb_downstream_context_20260724_v1_31_8"
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")
FORBIDDEN_PREFIXES = (
    "path_arch_",
    "__peak_",
    "__mae_",
    "__mfe_",
    "__future_",
    "__first_touch_",
)


class DownstreamContextError(RuntimeError):
    """Raised when the strict base-OOF context contract cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_side(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    if not text.isin(("long", "short")).all():
        raise DownstreamContextError("base OOF contains an invalid side")
    return text


def build_context(
    top40: pd.DataFrame,
    outer: pd.DataFrame,
) -> pd.DataFrame:
    """Derive pre-entry score context without using an outcome column."""

    top_required = {
        "candidate_id",
        "side_name",
        "__ts__",
        "__symbol__",
        "prediction",
        "base_candidate_rank_timestamp_side",
        "base_candidate_rank_pct_timestamp_side",
        "base_candidate_group_rows",
        "selected_top40",
        "prediction_source",
    }
    outer_required = {
        "candidate_id",
        "side_name",
        "__ts__",
        "__symbol__",
        "prediction",
    }
    missing_top = sorted(top_required.difference(top40.columns))
    missing_outer = sorted(outer_required.difference(outer.columns))
    if missing_top or missing_outer:
        raise DownstreamContextError(
            f"missing top40={missing_top} outer={missing_outer}"
        )
    selected = top40.loc[:, sorted(top_required)].copy()
    population = outer.loc[:, sorted(outer_required)].copy()
    for frame in (selected, population):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["side_name"] = _canonical_side(frame["side_name"])
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["__symbol__"] = frame["__symbol__"].astype(str)
        frame["prediction"] = pd.to_numeric(
            frame["prediction"], errors="coerce"
        ).astype(float)
    if (
        selected["candidate_id"].duplicated().any()
        or population["candidate_id"].duplicated().any()
        or not selected["selected_top40"].astype(bool).all()
        or set(selected["prediction_source"].astype(str)) != {"outer_oof_fold_model"}
        or not np.isfinite(selected["prediction"]).all()
        or not np.isfinite(population["prediction"]).all()
    ):
        raise DownstreamContextError(
            "base OOF identity, selection, or score is invalid"
        )
    outer_by_id = population.set_index("candidate_id")
    aligned = outer_by_id.reindex(selected["candidate_id"])
    if aligned["prediction"].isna().any():
        raise DownstreamContextError("top40 identities are outside the base outer OOF")
    if not np.allclose(
        selected["prediction"].to_numpy(float),
        aligned["prediction"].to_numpy(float),
        rtol=0.0,
        atol=1e-7,
    ):
        raise DownstreamContextError("top40 scores differ from the base outer OOF")

    group_keys = ["__ts__", "side_name"]
    population["base_rank_decile"] = np.clip(
        np.floor(
            population.groupby(group_keys, sort=False)["prediction"].rank(
                method="first", ascending=False, pct=True
            )
            * 10.0
        ).astype(int),
        0,
        9,
    )
    group_stats = population.groupby(group_keys, sort=False)["prediction"].agg(
        group_score_mean="mean",
        group_score_std="std",
    )
    decile_stats = population.groupby([*group_keys, "base_rank_decile"], sort=False)[
        "prediction"
    ].agg(
        decile_score_mean="mean",
        decile_score_std="std",
    )
    cutoff = (
        selected.groupby(group_keys, sort=False)["prediction"]
        .min()
        .rename("base_cutoff_score")
    )
    output = selected.copy()
    output["base_rank_decile"] = np.clip(
        np.floor(
            pd.to_numeric(
                output["base_candidate_rank_pct_timestamp_side"], errors="raise"
            )
            * 10.0
        ).astype(int),
        0,
        9,
    )
    output = output.join(group_stats, on=group_keys)
    output = output.join(decile_stats, on=[*group_keys, "base_rank_decile"])
    output = output.join(cutoff, on=group_keys)
    output["score"] = output["prediction"].astype(np.float32)
    output["base_oof_score"] = output["prediction"].astype(np.float32)
    output["base_margin_to_cutoff"] = (
        output["prediction"] - output["base_cutoff_score"]
    ).astype(np.float32)
    safe_group_std = output["group_score_std"].where(
        output["group_score_std"].gt(1e-12)
    )
    output["base_margin_to_cutoff_z"] = (
        (output["base_margin_to_cutoff"] / safe_group_std)
        .fillna(0.0)
        .astype(np.float32)
    )
    safe_decile_std = output["decile_score_std"].where(
        output["decile_score_std"].gt(1e-12)
    )
    output["base_signal_zscore_within_archetype"] = (
        ((output["prediction"] - output["decile_score_mean"]) / safe_decile_std)
        .fillna(0.0)
        .astype(np.float32)
    )
    output["base_score_z_timestamp_side"] = (
        ((output["prediction"] - output["group_score_mean"]) / safe_group_std)
        .fillna(0.0)
        .astype(np.float32)
    )
    output["archetype"] = output["base_rank_decile"].map(
        lambda value: f"base_rank_decile_{int(value)}"
    )
    output["archetype_label_family"] = output["archetype"]
    output["archetype_policy_key"] = output["archetype"]
    output["policy_archetype"] = output["archetype"]
    output["local_side_archetype"] = (
        output["side_name"].astype(str) + "__" + output["archetype"].astype(str)
    )
    output["side"] = output["side_name"]
    keep = [
        "__ts__",
        "__symbol__",
        "side",
        "side_name",
        "candidate_id",
        "selected_top40",
        "prediction_source",
        "score",
        "base_oof_score",
        "base_candidate_rank_timestamp_side",
        "base_candidate_rank_pct_timestamp_side",
        "base_candidate_group_rows",
        "base_cutoff_score",
        "base_margin_to_cutoff",
        "base_margin_to_cutoff_z",
        "base_score_z_timestamp_side",
        "base_signal_zscore_within_archetype",
        "base_rank_decile",
        "archetype",
        "archetype_label_family",
        "archetype_policy_key",
        "policy_archetype",
        "local_side_archetype",
    ]
    output = output.loc[:, keep].sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="mergesort"
    )
    if (
        output["candidate_id"].duplicated().any()
        or output.loc[
            :,
            [
                "score",
                "base_margin_to_cutoff",
                "base_margin_to_cutoff_z",
                "base_signal_zscore_within_archetype",
            ],
        ]
        .isna()
        .any(axis=None)
        or any(str(column).startswith(FORBIDDEN_PREFIXES) for column in output.columns)
    ):
        raise DownstreamContextError("derived context is not finite and pre-entry only")
    return output.reset_index(drop=True)


def run(
    *,
    top40_path: Path,
    top40_manifest_path: Path,
    outer_path: Path,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite downstream context: {destination}"
        )
    manifest = json.loads(top40_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("output", {}).get("sha256") != _sha256(top40_path)
        or manifest.get("source", {}).get("sha256") != _sha256(outer_path)
        or manifest.get("selected_rows") != 300315
        or manifest.get("source_rows") != 744251
    ):
        raise DownstreamContextError("canonical top40 source binding changed")
    revision = _git_revision()
    top40 = pd.read_parquet(top40_path)
    outer = pd.read_parquet(
        outer_path,
        columns=["candidate_id", "side_name", "__ts__", "__symbol__", "prediction"],
    )
    context = build_context(top40, outer)
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True)
        output_path = stage / "context.parquet"
        context.to_parquet(
            output_path, index=False, compression="zstd", compression_level=5
        )
        result = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_STRICT_BASE_OOF_PREENTRY_CONTEXT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": revision,
            "top40": {
                "path": str(top40_path),
                "sha256": _sha256(top40_path),
                "manifest_sha256": _sha256(top40_manifest_path),
            },
            "outer_oof": {
                "path": str(outer_path),
                "sha256": _sha256(outer_path),
            },
            "output": {
                "path": str(destination / output_path.name),
                "sha256": _sha256(output_path),
                "rows": len(context),
                "columns": len(context.columns),
                "candidate_identity_sha256": candidate_identity_sha256(
                    context, columns=IDENTITY_COLUMNS
                ),
            },
            "side_rows": {
                side: int(context["side_name"].eq(side).sum())
                for side in ("long", "short")
            },
            "archetype_contract": (
                "pre-entry base rank decile within UTC timestamp and side; "
                "derived only from strict outer-OOF scores"
            ),
            "feature_contract": {
                "mandatory_auxiliary_handoff_features": [
                    "score",
                    "base_margin_to_cutoff",
                    "base_margin_to_cutoff_z",
                    "base_signal_zscore_within_archetype",
                ],
                "realized_outcome_columns": [],
                "all_values_available_at_signal_decision": True,
            },
        }
        _write_json(stage / "manifest.json", result)
        os.replace(stage, destination)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top40", type=Path, default=DEFAULT_TOP40)
    parser.add_argument("--top40-manifest", type=Path, default=DEFAULT_TOP40_MANIFEST)
    parser.add_argument("--outer-oof", type=Path, default=DEFAULT_OUTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        top40_path=args.top40,
        top40_manifest_path=args.top40_manifest,
        outer_path=args.outer_oof,
        destination=args.output_dir,
    )
    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
