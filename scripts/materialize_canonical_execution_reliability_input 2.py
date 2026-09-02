#!/usr/bin/env python3
"""Materialize exact identities, labels, and causal inputs for reliability heads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONVERSION = ROOT / "data_perp/artifacts/v5_conversion_residual_input_20260730_v3"
FULL_PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
REPAIR = ROOT / "data_perp/artifacts/canonical_full_base_opportunity_ablation_20260730_v2"
REPAIR_SOURCE = ROOT / "data_perp/artifacts/canonical_full_base_opportunity_ablation_20260729_v1"
PATH_LABELS = ROOT / "data_perp/artifacts/febapr2025_top40_exact1m_path_head_labels_20260727_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v2"
IDENTITY = ("candidate_id", "side_name")
EXPECTED_ROWS = 110_730
OPPORTUNITY_TEMPERATURE = 0.0025

SCORE_CONTEXT = (
    "base_oof_score",
    "base_rank_pct_timestamp_side",
    "base_score_z_timestamp_side",
    "base_group_rows_timestamp_side",
    "base_margin_to_top40_cutoff",
    "base_margin_to_top40_cutoff_z",
    "base_rank_pct_timestamp_global",
    "base_score_z_timestamp_global",
    "base_group_rows_timestamp_global",
)
STATE_LEVELS = (
    "range_24h_pct",
    "__meta_raw__volatility_zscore",
    "trend_r2_24",
    "jump_intensity",
    "__meta_raw__chop_score",
)
TRANSITIONS = tuple(
    f"preentry_transition__{name}__delta_{horizon}h"
    for name in (
        "range_24h_pct",
        "meta_raw__volatility_zscore",
        "trend_r2_24",
        "jump_intensity",
        "meta_raw__chop_score",
    )
    for horizon in (3, 12)
)
REGIME_COMPOSITES = (
    "__regime_source_shock_impulse_score__",
    "__regime_source_execution_quality_score__",
    "__regime_source_execution_risk_score__",
    "__regime_source_oi_agreement_score__",
    "__regime_source_compression_score__",
    "__regime_source_loud_breakout_impulse_score__",
    "__regime_source_dirty_shock_avoid_score__",
    "__regime_source_clean_execution_context_score__",
)
TRANSITION_INTERACTION_SOURCES = tuple(
    f"preentry_transition__{name}__delta_12h"
    for name in (
        "range_24h_pct",
        "meta_raw__volatility_zscore",
        "trend_r2_24",
        "jump_intensity",
        "meta_raw__chop_score",
    )
)
V4_SCORES = (
    "raw_score",
    "score_base_alpha",
    "score_residual_expected_ev",
    "direct_q25_return",
)
PATH_LABEL_COLUMNS = (
    "__decision_ts__",
    "__label_end_ts__",
    "__meaningful_mfe_reached_12h__",
    "__soft_tb_upper_hit_12h__",
    "__soft_tb_lower_hit_12h__",
    "__soft_tb_first_event__",
    "__soft_tb_order_ambiguous__",
    "__peak_mfe_return_12h__",
    "__peak_mfe_atr_12h__",
)


class MaterializationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(safe(dict(payload)), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_seal(root: Path, schema: str) -> dict[str, Any]:
    manifest = root / "manifest.json"
    seal = root / "manifest.sha256"
    if not manifest.is_file() or not seal.is_file():
        raise MaterializationError(f"sealed source is missing: {root}")
    if sha256(manifest) != seal.read_text().split()[0]:
        raise MaterializationError(f"manifest seal mismatch: {root}")
    payload = json.loads(manifest.read_text())
    if payload.get("schema") != schema:
        raise MaterializationError(f"source schema mismatch: {root}")
    return payload


def utc(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    return result


def join_exact(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    if right.duplicated(list(IDENTITY)).any():
        raise MaterializationError(f"{name} identities are not unique")
    timestamp = f"__{name}_ts__"
    selected = right.rename(columns={"__ts__": timestamp})
    result = left.merge(
        selected,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    if result[timestamp].isna().any():
        raise MaterializationError(f"{name} does not cover every identity")
    if not result.__ts__.eq(result[timestamp]).all():
        raise MaterializationError(f"{name} timestamp parity failed")
    return result.drop(columns=timestamp)


def selected_configs(repair_manifest: Mapping[str, Any]) -> list[str]:
    rows = repair_manifest["repair"]["selected_configs"]
    configs = [
        "__".join((str(row["target"]), str(row["arm"]), str(row["geometry"])))
        for row in rows
    ]
    if len(configs) != 8 or len(set(configs)) != 8:
        raise MaterializationError("repair must provide eight unique support configs")
    return configs


def load_support_sidecars(
    conversion: pd.DataFrame,
    repair_root: Path,
    repair_source: Path,
    configs: Sequence[str],
) -> pd.DataFrame:
    development = pd.read_parquet(
        repair_source / "development_oof_predictions.parquet"
    )
    development = utc(development, ["__ts__"])
    march_columns = [*IDENTITY, "__ts__", *[f"raw__{config}" for config in configs]]
    missing = set(march_columns).difference(development.columns)
    if missing:
        raise MaterializationError(f"development support columns missing: {sorted(missing)}")
    march = development.loc[:, march_columns].rename(
        columns={f"raw__{config}": f"support__{config}" for config in configs}
    )

    april_long = pd.read_parquet(
        repair_root / "april_predictions.parquet",
        columns=[*IDENTITY, "__ts__", "config", "raw_score"],
    )
    april_long = utc(april_long, ["__ts__"])
    if april_long.duplicated([*IDENTITY, "config"]).any():
        raise MaterializationError("April repaired supports overlap")
    if set(april_long.config.astype(str).unique()) != set(configs):
        raise MaterializationError("April repaired support configs differ from manifest")
    april = (
        april_long.pivot(
            index=[*IDENTITY, "__ts__"],
            columns="config",
            values="raw_score",
        )
        .rename(columns={config: f"support__{config}" for config in configs})
        .reset_index()
    )
    supports = pd.concat([march, april], ignore_index=True, sort=False)
    selected = supports.loc[
        supports.candidate_id.astype(str).isin(
            set(conversion.candidate_id.astype(str))
        )
    ].copy()
    if len(selected) != len(conversion):
        raise MaterializationError(
            f"support identity count mismatch: {len(selected)} != {len(conversion)}"
        )
    return selected


def load_path_labels(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    if index.get("schema") != "febapr2025_exact1m_path_head_shard_index_v1":
        raise MaterializationError("wrong exact path-label index")
    if not index.get("coverage", {}).get("complete"):
        raise MaterializationError("exact path labels are incomplete")
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {"index.json": sha256(index_path)}
    for item in index["shards"]:
        path = ROOT / str(item["labels"])
        actual = sha256(path)
        if actual != item["sha256"]:
            raise MaterializationError(f"path-label shard hash mismatch: {path}")
        hashes[str(path.relative_to(ROOT))] = actual
        pieces.append(
            pd.read_parquet(
                path,
                columns=["candidate_id", "side_name", "__ts__", *PATH_LABEL_COLUMNS],
            )
        )
    result = utc(
        pd.concat(pieces, ignore_index=True),
        ["__ts__", "__decision_ts__", "__label_end_ts__"],
    )
    if len(result) != int(index["coverage"]["expected_rows"]):
        raise MaterializationError("path-label row count drift")
    return result, hashes


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    net = result.execution_net_ev_12h.to_numpy(float)
    gross = result.execution_gross_ev_12h.to_numpy(float)
    cost = result.execution_cost_return.to_numpy(float)
    mfe = result.execution_mfe_return_12h.to_numpy(float)
    if np.max(np.abs(gross - cost - net)) > 1e-7:
        raise MaterializationError("gross-cost-net accounting failed")
    event = result.__soft_tb_first_event__.astype(str)
    allowed = {"favorable_first", "adverse_first_or_conflict", "timeout"}
    observed = set(event.unique())
    if not observed or not observed.issubset(allowed):
        raise MaterializationError("triple-barrier competing classes changed")
    ambiguous = result.__soft_tb_order_ambiguous__.astype(bool)
    if (ambiguous & ~event.eq("adverse_first_or_conflict")).any():
        raise MaterializationError("ambiguous ordering escaped adverse/conflict class")
    result["target_meaningful_mfe"] = result.__meaningful_mfe_reached_12h__.astype(np.int8)
    result["target_clean_favorable_first"] = event.eq("favorable_first").astype(np.int8)
    result["target_competing_class"] = event
    margin = mfe - cost
    result["target_economic_opportunity_hard"] = (margin > 0).astype(np.int8)
    result["target_economic_opportunity_soft_25bps"] = 1.0 / (
        1.0 + np.exp(-np.clip(margin / OPPORTUNITY_TEMPERATURE, -40.0, 40.0))
    )
    result["target_net_positive"] = (net > 0).astype(np.int8)
    result["target_net_positive_given_opportunity_valid"] = (margin > 0).astype(np.int8)
    result["target_positive_net_magnitude"] = np.maximum(net, 0)
    result["target_adverse_loss_magnitude"] = np.maximum(-net, 0)
    result["target_severe_loss_100bps"] = (net <= -0.01).astype(np.int8)
    result["target_direct_net"] = net
    result["target_full_horizon_gross_to_mfe_ratio_diagnostic_only"] = np.clip(
        np.divide(np.maximum(gross, 0), np.maximum(mfe, 0.0001)),
        0,
        1,
    )
    return result


def feature_roles(configs: Sequence[str]) -> dict[str, Any]:
    support = [f"support__{config}" for config in configs]
    default = [*V4_SCORES, *support, *SCORE_CONTEXT, *STATE_LEVELS, *TRANSITIONS, *REGIME_COMPOSITES]
    forbidden = []
    for name in default:
        lowered = name.lower()
        if (
            lowered.startswith(("execution_", "target_"))
            or any(token in lowered for token in ("mfe", "mae", "time_to", "wait", "price"))
        ):
            forbidden.append(name)
    if forbidden:
        raise MaterializationError(f"forbidden default EV feature: {forbidden}")
    return {
        "default_ev_inputs": default,
        "repaired_full_base_support_sidecars": support,
        "transition_interaction_sources": list(TRANSITION_INTERACTION_SOURCES),
        "interaction_contract": (
            "within each fold/side, standardize each 12h transition source on "
            "training rows only, multiply by base_oof_score, clip to [-1,1]"
        ),
        "target_only_never_features": [
            "target_meaningful_mfe",
            "target_clean_favorable_first",
            "target_competing_class",
            "target_economic_opportunity_hard",
            "target_economic_opportunity_soft_25bps",
            "target_net_positive",
            "target_net_positive_given_opportunity_valid",
            "target_positive_net_magnitude",
            "target_adverse_loss_magnitude",
            "target_severe_loss_100bps",
            "target_direct_net",
            "target_full_horizon_gross_to_mfe_ratio_diagnostic_only",
        ],
        "separate_action_layer": [
            "time-to-MFE and hit-by-horizon predictions",
            "bars-to-adverse-trough predictions",
            "MAE path timing",
            "target-price and wait/reprice actions",
        ],
        "explicitly_unavailable": [
            "archetype-relative z-score in the canonical 2025 lineage",
            "proper pre-exit capture target/head",
        ],
        "excluded_from_default_ev": [
            "DAE and GMM geometry outputs",
            "mapped coordinates and mapping support fields",
            "realized path/auxiliary labels",
            "timing, MAE, target-price and wait fields",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    conversion_manifest = verify_seal(args.conversion, "v5_conversion_residual_input_v3")
    full_manifest = verify_seal(args.full_panel, "canonical_opportunity_payoff_trust_panel_v2")
    repair_manifest = verify_seal(
        args.repair, "canonical_full_base_opportunity_ablation_raw_oof_repair_v2"
    )
    configs = selected_configs(repair_manifest)
    conversion = utc(
        pd.read_parquet(args.conversion / "panel.parquet"),
        ["__ts__", "execution_decision_utc", "execution_label_end_utc"],
    )
    if len(conversion) != EXPECTED_ROWS or conversion.duplicated(list(IDENTITY)).any():
        raise MaterializationError("conversion identity contract failed")

    supports = load_support_sidecars(
        conversion, args.repair, args.repair_source, configs
    )
    panel = join_exact(conversion, supports, "support")

    full_columns = [
        *IDENTITY,
        "__ts__",
        *SCORE_CONTEXT,
        *STATE_LEVELS,
        *TRANSITIONS,
        *REGIME_COMPOSITES,
    ]
    full = utc(
        pd.read_parquet(args.full_panel / "panel.parquet", columns=full_columns),
        ["__ts__"],
    )
    overlap = set(full_columns).intersection(panel.columns) - set(IDENTITY) - {"__ts__"}
    full = full.rename(columns={name: f"{name}__individual" for name in overlap})
    panel = join_exact(panel, full, "individual_context")
    for name in overlap:
        source = f"{name}__individual"
        if name == "base_oof_score":
            if not np.array_equal(
                panel.score_base_alpha.to_numpy(float),
                panel[source].to_numpy(float),
            ):
                raise MaterializationError("full-panel/base score parity failed")
        panel[name] = panel.pop(source)

    path_labels, path_hashes = load_path_labels(args.path_labels)
    selected_labels = path_labels.loc[
        path_labels.candidate_id.astype(str).isin(set(panel.candidate_id.astype(str)))
    ].copy()
    panel = join_exact(panel, selected_labels, "path_labels")
    if not panel.__decision_ts__.eq(panel.execution_decision_utc).all():
        raise MaterializationError("path-label decision timestamp parity failed")
    if not panel.__label_end_ts__.eq(panel.execution_label_end_utc).all():
        raise MaterializationError("path-label resolution timestamp parity failed")
    panel = add_targets(panel)
    roles = feature_roles(configs)
    default_features = roles["default_ev_inputs"]
    if panel.loc[:, default_features].isna().any().any():
        raise MaterializationError("default EV input contains missing values")
    if not np.isfinite(panel.loc[:, default_features].to_numpy(float)).all():
        raise MaterializationError("default EV input contains non-finite values")
    if len(panel) != EXPECTED_ROWS or panel.duplicated(list(IDENTITY)).any():
        raise MaterializationError("final reliability identity contract failed")

    stage = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        panel.to_parquet(stage / "panel.parquet", index=False, compression="zstd")
        write_json(stage / "feature_roles.json", roles)
        target_support = pd.DataFrame(
            [
                {
                    "target": name,
                    "rows": int(panel[name].notna().sum()),
                    "positive_or_valid_rows": (
                        int(panel[name].sum())
                        if pd.api.types.is_integer_dtype(panel[name])
                        else int(np.isfinite(pd.to_numeric(panel[name], errors="coerce")).sum())
                    ),
                }
                for name in roles["target_only_never_features"]
                if name in panel and name != "target_competing_class"
            ]
        )
        target_support.to_csv(stage / "target_support.csv", index=False)
        (
            panel.target_competing_class.value_counts(dropna=False)
            .rename_axis("class")
            .reset_index(name="rows")
            .sort_values("class")
            .to_csv(stage / "competing_class_support.csv", index=False)
        )
        outputs = {
            path.name: sha256(path) for path in stage.iterdir() if path.is_file()
        }
        manifest = {
            "schema": "canonical_execution_reliability_input_v2",
            "run_id": args.output_dir.name,
            "status": "SEALED_RESEARCH_INPUT_READY_NO_MODEL_SELECTION_NO_PROMOTION",
            "promotion_eligible": False,
            "rows": len(panel),
            "side_rows": panel.groupby("side_name").size().to_dict(),
            "period_rows": panel.groupby(panel.__ts__.dt.strftime("%Y-%m")).size().to_dict(),
            "identity_contract": "candidate_id + side_name; exact UTC timestamp assertion",
            "label_contract": (
                "exact decision+12h ATR-normalized triple barrier plus exact "
                "deployed-policy gross-cost-net; cost appears once"
            ),
            "feature_contract": roles,
            "support_configs": configs,
            "input_sha256": {
                "conversion_manifest": sha256(args.conversion / "manifest.json"),
                "conversion_panel": conversion_manifest["outputs_sha256"]["panel.parquet"],
                "full_panel_manifest": sha256(args.full_panel / "manifest.json"),
                "full_panel": full_manifest["outputs_sha256"]["panel.parquet"],
                "repair_manifest": sha256(args.repair / "manifest.json"),
                "repair_april_predictions": repair_manifest["outputs_sha256"]["april_predictions.parquet"],
                "repair_source_development_oof": sha256(
                    args.repair_source / "development_oof_predictions.parquet"
                ),
                **path_hashes,
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "April is reused diagnostic evidence, never promotion evidence.",
                "Full-horizon gross/MFE ratio is diagnostic only and is not a proper pre-exit capture target.",
                "A proper 2025 pre-exit capture label and OOF head remain to be materialized.",
                "Archetype-relative z-score is unavailable and is not synthesized.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            sha256(stage / "manifest.json") + "  manifest.json\n"
        )
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--conversion", type=Path, default=CONVERSION)
    command.add_argument("--full-panel", type=Path, default=FULL_PANEL)
    command.add_argument("--repair", type=Path, default=REPAIR)
    command.add_argument("--repair-source", type=Path, default=REPAIR_SOURCE)
    command.add_argument("--path-labels", type=Path, default=PATH_LABELS)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
