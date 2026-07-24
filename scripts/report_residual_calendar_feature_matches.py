#!/usr/bin/env python3
"""Map recognized residual-calendar cells to their observable composite inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from extreme_price_movements.residual_leaf_state_discovery import _leaf_paths


def _definition_map(root: Path, models: Path, scope: str) -> dict[tuple[str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for item in json.loads((root / "composite_definitions.json").read_text()):
        side = str(item["side_name"])
        archetype = str(item["archetype_policy_key"])
        model_name = f"{side}__{archetype}.joblib" if archetype != "__side_global__" else f"{side}____side_global__.joblib"
        bundle = joblib.load(models / model_name)
        paths = _leaf_paths(bundle["model"], bundle["features"]).set_index(["tree_index", "leaf_index"])
        for composite in item.get("lgbm_leaf_composites", []):
            features: set[str] = set()
            for coordinate in composite["leaf_coordinates"]:
                try:
                    value = paths.loc[tuple(coordinate), "path_features"]
                except KeyError:
                    continue
                features.update(feature for feature in str(value).split("|") if feature)
            output_name = str(composite["output_name"])
            result[(side, archetype, output_name)] = {
                "scope": scope,
                "source": "lgbm_leaf_pattern",
                "features": "|".join(sorted(features)),
                "interaction": "LGBM leaf paths: " + " + ".join(sorted(features)),
            }
        for index, composite in enumerate(item.get("unsupervised_pair_composites", [])):
            features = [str(composite.get("feature", "")), str(composite.get("feature_b", ""))]
            features = [feature for feature in features if feature]
            bins = [str(composite.get("feature_bin", "")), str(composite.get("feature_b_bin", ""))]
            output_name = f"residual_episode_unsup_composite_{index}"
            result[(side, archetype, output_name)] = {
                "scope": scope,
                "source": "unsupervised_pair_intensity",
                "features": "|".join(features),
                "interaction": " AND ".join(
                    f"{feature} [{bins[position]}]" for position, feature in enumerate(features)
                ),
            }
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    mapping = _definition_map(args.local, args.local_models, "side_archetype_local")
    mapping.update(_definition_map(args.shared, args.shared_models, "side_global_shared"))
    rows: list[dict[str, object]] = []
    for row in calendar.itertuples(index=False):
        composites = [value for value in str(getattr(row, "matching_composites", "")).split("|") if value and value != "nan"]
        matches: list[dict[str, str]] = []
        for composite in composites:
            local_key = (str(row.side_name), str(row.archetype_policy_key), composite)
            shared_key = (str(row.side_name), "__side_global__", composite)
            if local_key in mapping:
                matches.append(mapping[local_key])
            if shared_key in mapping:
                matches.append(mapping[shared_key])
        features = sorted({feature for match in matches for feature in match["features"].split("|") if feature})
        interactions = sorted({match["interaction"] for match in matches if match["interaction"]})
        sources = sorted({match["scope"] + ":" + match["source"] for match in matches})
        rows.append(
            {
                "day": row.day,
                "side_name": row.side_name,
                "archetype_policy_key": row.archetype_policy_key,
                "status": row.status,
                "evidence_scope": row.evidence_scope,
                "recognition_sources": getattr(row, "recognition_sources", ""),
                "matching_composites": getattr(row, "matching_composites", ""),
                "matching_feature_count": len(features),
                "matching_features": "|".join(features),
                "matching_interactions": " || ".join(interactions),
                "resolved_sources": "|".join(sources),
            }
        )
    cells = pd.DataFrame(rows).sort_values(["day", "side_name", "archetype_policy_key"], kind="stable")
    cells.to_csv(args.output / "calendar_cells_with_feature_matches.csv", index=False)
    daily = (
        cells.groupby("day", observed=True)
        .agg(
            calendar_cells=("status", "size"),
            recognized_cells=("status", lambda values: int(values.eq("recognized").sum())),
            ignored_cells=("status", lambda values: int(values.eq("ignored").sum())),
            recognized_side_archetypes=("archetype_policy_key", lambda values: "|".join(sorted(set(values[cells.loc[values.index, "status"].eq("recognized")])))),
            ignored_side_archetypes=("archetype_policy_key", lambda values: "|".join(sorted(set(values[cells.loc[values.index, "status"].eq("ignored")])))),
            useful_features=("matching_features", lambda values: "|".join(sorted({feature for text in values for feature in str(text).split("|") if feature and feature != "nan"}))),
            useful_interactions=("matching_interactions", lambda values: " || ".join(sorted({text for text in values if text and text != "nan"}))),
        )
        .reset_index()
    )
    daily["day_status"] = "partially_recognized"
    daily.loc[daily["ignored_cells"].eq(0), "day_status"] = "fully_recognized"
    daily.loc[daily["recognized_cells"].eq(0), "day_status"] = "not_recognized"
    daily.to_csv(args.output / "calendar_days_recognized_vs_not.csv", index=False)
    manifest = {
        "schema": "residual_calendar_feature_matches_v1",
        "calendar_cells": int(len(cells)),
        "recognized_cells": int(cells["status"].eq("recognized").sum()),
        "ignored_cells": int(cells["status"].eq("ignored").sum()),
        "calendar_days": int(len(daily)),
        "fully_recognized_days": int(daily["day_status"].eq("fully_recognized").sum()),
        "partially_recognized_days": int(daily["day_status"].eq("partially_recognized").sum()),
        "not_recognized_days": int(daily["day_status"].eq("not_recognized").sum()),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/residual_episode_recognition_calendar_20260712_v1/calendar_recognized_vs_ignored.csv"))
    parser.add_argument("--local", type=Path, default=Path("data_perp/reports/residual_episode_composite_discovery_20260712_v2_leakagesafe"))
    parser.add_argument("--local-models", type=Path, default=Path("data_perp/reports/residual_calendar_leaf_state_discovery_20260712_v1/models"))
    parser.add_argument("--shared", type=Path, default=Path("data_perp/reports/residual_side_global_episode_composite_discovery_20260712_v1"))
    parser.add_argument("--shared-models", type=Path, default=Path("data_perp/reports/side_global_calendar_leaf_models_20260712_v1/models"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/residual_calendar_feature_matches_20260712_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
