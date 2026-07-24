#!/usr/bin/env python3
"""Package the historical V9 tail policy over the deployed meta-score domain."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.meta_historical_rank import HistoricalScoreRankReference
from extreme_price_movements.v9_meta_score_tail_bundle import MetaScoreV9TailBundle


DEFAULT_ARTIFACT = Path(
    "data_perp/artifacts/"
    "s59_s52_finalfit_meta_repairedcoverage_v9tail95_mlp_hierev_20260715_v3"
)
DEFAULT_V9 = Path(
    "data_perp/reports/"
    "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_20260712_v9"
)
DEFAULT_RANK_SHARDS = Path(
    "data_perp/artifacts/20260713_meta_fullhistory_old55_expandedpool/"
    "prediction_shards"
)
DEFAULT_STATE = Path(
    "data_perp/reports/"
    "residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/"
    "oos_residual_event_states.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_reference(
    artifact: Path,
    *,
    prediction_shards: Path | None,
    end_exclusive: pd.Timestamp,
    scope: str,
    rank_method: str,
) -> HistoricalScoreRankReference:
    reference = HistoricalScoreRankReference(
        score_col="score_meta_base_soft_label",
        side_col="side_name",
        rank_method=rank_method,
    )
    if prediction_shards is not None:
        parts: list[pd.DataFrame] = []
        raw_paths = (
            [prediction_shards]
            if prediction_shards.is_file()
            else [
                Path(path)
                for path in sorted(glob.glob(str(prediction_shards / "*.parquet")))
            ]
        )
        for raw_path in raw_paths:
            part = pd.read_parquet(
                raw_path,
                columns=["__ts__", "side_name", "score_meta_base_soft_label"],
            )
            part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
            part = part.loc[part["__ts__"].lt(end_exclusive)]
            if not part.empty:
                parts.append(part)
        if not parts:
            raise ValueError(
                f"No prior-OOS rank scores found in {prediction_shards} "
                f"before {end_exclusive}"
            )
        scores = pd.concat(parts, ignore_index=True, copy=False)
        reference.fit_start = str(scores["__ts__"].min())
        reference.fit_end = str(scores["__ts__"].max())
        values = pd.to_numeric(
            scores["score_meta_base_soft_label"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        reference.sorted_scores_global = np.sort(values[np.isfinite(values)])
        if scope == "side":
            reference.sorted_scores_by_side = {}
            for side, group in scores.groupby("side_name", observed=True, sort=True):
                values = pd.to_numeric(
                    group["score_meta_base_soft_label"], errors="coerce"
                ).to_numpy(dtype=np.float32)
                reference.sorted_scores_by_side[str(side)] = np.sort(
                    values[np.isfinite(values)]
                )
        elif scope == "global":
            reference.sorted_scores_by_side = {}
        else:
            raise ValueError(f"Unsupported rank-reference scope: {scope!r}")
        return reference

    by_side: dict[str, np.ndarray] = {}
    for side in ("long", "short"):
        path = (
            artifact
            / "meta_oof"
            / f"meta_score_reference_{side}_s52_meta_threshold_handoff.parquet"
        )
        values = pd.to_numeric(pd.read_parquet(path)["score"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        by_side[side] = np.sort(values[np.isfinite(values)])
    reference.sorted_scores_by_side = by_side
    reference.sorted_scores_global = np.sort(
        np.concatenate(list(by_side.values())).astype(np.float32, copy=False)
    )
    reference.fit_start = "frozen_meta_oof_reference"
    reference.fit_end = "2026-06-30T23:00:00+00:00"
    return reference


def _local_references(
    state_path: Path,
    selected_path: Path,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[dict[tuple[str, str, str], list[tuple[str, float, np.ndarray]]], dict]:
    selected = pd.read_csv(selected_path)
    selected = selected.sort_values(
        ["side_name", "archetype_policy_key", "event"], kind="stable"
    ).groupby(
        ["side_name", "archetype_policy_key", "event"],
        observed=True,
        sort=True,
    ).head(1)
    features = selected["feature"].astype(str).drop_duplicates().tolist()
    available = set(pq.read_schema(state_path).names)
    required = ["__ts__", "side_name", "archetype_policy_key", *features]
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"Residual state source missing V9 features: {missing}")
    state = pd.read_parquet(state_path, columns=required)
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True, errors="coerce")
    state = state.loc[state["__ts__"].ge(train_start) & state["__ts__"].lt(train_end)]
    references: dict[
        tuple[str, str, str], list[tuple[str, float, np.ndarray]]
    ] = {}
    support: dict[str, int] = {}
    for row in selected.itertuples(index=False):
        side = str(row.side_name)
        archetype = str(row.archetype_policy_key)
        event = str(row.event)
        feature = str(row.feature)
        direction = float(row.direction)
        local = state.loc[
            state["side_name"].astype(str).eq(side)
            & state["archetype_policy_key"].astype(str).eq(archetype)
        ]
        values = direction * pd.to_numeric(local[feature], errors="coerce").to_numpy(
            dtype=np.float32
        )
        values = np.sort(values[np.isfinite(values)])
        if len(values) < 200:
            raise ValueError(
                f"Insufficient V9 reference support for {side}|{archetype}|{event}: "
                f"{len(values)}"
            )
        references[(side, archetype, event)] = [(feature, direction, values)]
        support[f"{side}|{archetype}|{event}"] = int(len(values))
    return references, support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--v9-report", type=Path, default=DEFAULT_V9)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--train-end", default="2026-04-01")
    parser.add_argument(
        "--rank-reference-shards",
        type=Path,
        default=DEFAULT_RANK_SHARDS,
        help="Prior-OOS old55 meta prediction shards used by the historical rank.",
    )
    parser.add_argument("--rank-reference-end-exclusive", default="2026-07-01")
    parser.add_argument("--rank-scope", choices=("global", "side"), default="global")
    parser.add_argument("--rank-method", choices=("midrank", "right"), default="midrank")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.v9_report / "selected_local_features_strict.csv"
    forced = pd.read_csv(args.v9_report / "forced_nonzero_tail_summary.csv")
    row = forced.loc[forced["selector"].astype(str).str.endswith("tail_0.950")].iloc[0]
    references, support = _local_references(
        args.state,
        selected,
        train_start=pd.Timestamp(args.train_start, tz="UTC"),
        train_end=pd.Timestamp(args.train_end, tz="UTC"),
    )
    bundle = MetaScoreV9TailBundle(
        historical_rank_reference=_rank_reference(
            args.artifact,
            prediction_shards=args.rank_reference_shards,
            end_exclusive=pd.Timestamp(args.rank_reference_end_exclusive, tz="UTC"),
            scope=args.rank_scope,
            rank_method=args.rank_method,
        ),
        local_references=references,
        threshold=float(row["train_threshold"]),
        alpha_down=float(row["train_alpha_down"]),
        alpha_up=0.0,
        fit_through="2026-06-30T23:00:00+00:00",
        metadata={
            "policy_id": str(row["selector"]),
            "selected_feature_source": str(selected),
            "selected_feature_sha256": _sha256(selected),
            "state_reference_source": str(args.state),
            "state_reference_support": support,
            "score_reference_artifact": str(args.artifact),
            "score_reference_shards": str(args.rank_reference_shards),
            "score_reference_end_exclusive": args.rank_reference_end_exclusive,
            "score_reference_scope": args.rank_scope,
            "score_reference_rank_method": args.rank_method,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    manifest = bundle.manifest()
    manifest["output"] = str(args.output)
    manifest["output_sha256"] = _sha256(args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
