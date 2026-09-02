#!/usr/bin/env python3
"""Seal exact, target-only execution reliability labels derived from v4.

The pack deliberately separates pre-exit executable opportunity from realised
deployed-policy exits.  Every outcome remains unavailable until the complete
canonical 12-hour label has resolved; this artifact is neither a feature store
nor promotion evidence.
"""
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

SOURCE = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v4"
CONFIG = ROOT / "configs/canonical_execution_reliability_workstream_20260730_v2.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_execution_reliability_target_pack_20260730_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TIME = "execution_decision_utc"
END = "execution_label_end_utc"
SIDES = ("long", "short")
OPPORTUNITY_BUFFERS = (("0bps", 0.0), ("25bps", 0.0025), ("50bps", 0.005))
SEVERE_THRESHOLD = 0.01
EXIT_CLASSES = ("successful_trailing", "trailing_nonpositive", "hard_adverse", "timeout")


class TargetPackError(RuntimeError):
    """Raised for an invalid source, label, or sealed target-pack contract."""


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
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_sealed(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise TargetPackError(f"sealed artifact missing: {root}")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise TargetPackError(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise TargetPackError(f"schema mismatch: {manifest.get('schema')}")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or not outputs:
        raise TargetPackError("source has no output-hash ledger")
    for name, expected in outputs.items():
        output = root / str(name)
        if not output.is_file() or sha256(output) != str(expected):
            raise TargetPackError(f"output hash mismatch: {output}")
    return manifest


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema") != "canonical_execution_reliability_workstream_v2":
        raise TargetPackError("frozen reliability v2 config is required")
    folds = config.get("outer_folds")
    if not isinstance(folds, list) or not folds:
        raise TargetPackError("outer folds are missing")
    for fold in folds:
        if not all(key in fold for key in ("name", "validation_start_utc", "validation_end_utc")):
            raise TargetPackError("outer fold is incomplete")
    return config


def build_labels(source: pd.DataFrame) -> pd.DataFrame:
    """Create all targets without retaining outcome primitives as columns."""

    required = {
        *IDENTITY,
        TIME,
        END,
        "pre_exit_mfe_return",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_class",
        "target_pre_exit_economic_opportunity",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise TargetPackError(f"source lacks target primitives: {missing}")
    result = source.loc[:, [*IDENTITY, TIME, END]].copy()
    for column in ("__ts__", TIME, END):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if source.duplicated(["candidate_id", "side_name"]).any():
        raise TargetPackError("source identity is not unique")
    if not source.side_name.isin(SIDES).all():
        raise TargetPackError("source side is noncanonical")
    if not result[END].eq(result[TIME] + pd.Timedelta(hours=12)).all():
        raise TargetPackError("labels do not have the exact H12 availability contract")

    mfe = source["pre_exit_mfe_return"].to_numpy(float)
    cost = source["execution_cost_return"].to_numpy(float)
    net = source["execution_net_ev_12h"].to_numpy(float)
    exit_class = source["execution_exit_class"].astype(str)
    if not np.isfinite(np.column_stack([mfe, cost, net])).all():
        raise TargetPackError("target primitives are non-finite")
    if not exit_class.isin(("trailing", "full_stop", "timeout", "adverse_exit")).all():
        raise TargetPackError("unexpected deployed exit class")

    for suffix, buffer in OPPORTUNITY_BUFFERS:
        result[f"target_pre_exit_opportunity_{suffix}"] = (mfe > cost + buffer).astype(np.int8)
    if not np.array_equal(
        result["target_pre_exit_opportunity_0bps"].to_numpy(),
        source["target_pre_exit_economic_opportunity"].to_numpy(np.int8),
    ):
        raise TargetPackError("pre-exit 0bps opportunity diverges from sealed capture target")

    successful_trailing = exit_class.eq("trailing") & (net > 0.0)
    trailing_nonpositive = exit_class.eq("trailing") & ~successful_trailing
    full_stop = exit_class.eq("full_stop")
    timeout = exit_class.eq("timeout")
    other_adverse = exit_class.eq("adverse_exit")
    hard_adverse = full_stop | other_adverse
    result["target_successful_deployed_trailing"] = successful_trailing.astype(np.int8)
    result["target_deployed_full_stop"] = full_stop.astype(np.int8)
    result["target_deployed_timeout"] = timeout.astype(np.int8)
    result["target_deployed_other_adverse_exit_attribution_only"] = other_adverse.astype(np.int8)
    result["target_deployed_hard_adverse"] = hard_adverse.astype(np.int8)

    severe = net <= -SEVERE_THRESHOLD
    result["target_severe_loss_100bps"] = severe.astype(np.int8)
    conditional = np.full(len(result), np.nan, dtype=np.float32)
    conditional[severe] = np.log1p((-net[severe]) / SEVERE_THRESHOLD).astype(np.float32)
    result["target_conditional_severe_loss_log1p_100bps"] = conditional
    result["target_conditional_severe_loss_mask"] = severe.astype(np.int8)

    outcome_class = np.select(
        [successful_trailing, trailing_nonpositive, hard_adverse, timeout],
        EXIT_CLASSES,
        default="",
    )
    if not np.isin(outcome_class, EXIT_CLASSES).all():
        raise TargetPackError("deployed exit/economics classes are not exhaustive")
    result["target_deployed_exit_economics_class"] = pd.Categorical(
        outcome_class, categories=list(EXIT_CLASSES)
    )
    if (pd.get_dummies(result["target_deployed_exit_economics_class"]).sum(axis=1) != 1).any():
        raise TargetPackError("deployed exit/economics classes are not exclusive")
    if result.loc[~severe, "target_conditional_severe_loss_log1p_100bps"].notna().any():
        raise TargetPackError("conditional severe-loss target escaped its mask")
    result["label_available_at_utc"] = result[END]
    return result.sort_values([TIME, "candidate_id", "side_name"], kind="stable").reset_index(drop=True)


def target_columns() -> list[str]:
    return [
        *(f"target_pre_exit_opportunity_{suffix}" for suffix, _ in OPPORTUNITY_BUFFERS),
        "target_successful_deployed_trailing",
        "target_deployed_full_stop",
        "target_deployed_timeout",
        "target_deployed_other_adverse_exit_attribution_only",
        "target_deployed_hard_adverse",
        "target_severe_loss_100bps",
        "target_conditional_severe_loss_log1p_100bps",
        "target_conditional_severe_loss_mask",
        "target_deployed_exit_economics_class",
    ]


def support_ledgers(labels: pd.DataFrame, folds: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    binary_targets = [
        column for column in target_columns()
        if column not in {
            "target_conditional_severe_loss_log1p_100bps",
            "target_deployed_exit_economics_class",
        }
    ]
    support_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for fold in folds:
        start = pd.Timestamp(fold["validation_start_utc"])
        end = pd.Timestamp(fold["validation_end_utc"])
        valid = labels.loc[labels[TIME].ge(start) & labels[TIME].lt(end)]
        train = labels.loc[labels[TIME].lt(start) & labels[END].lt(start)]
        if len(valid) == 0 or len(train) == 0:
            raise TargetPackError(f"empty train/validation sidecar for {fold['name']}")
        for split, frame in (("train", train), ("valid", valid)):
            for side in SIDES:
                part = frame.loc[frame.side_name.eq(side)]
                if len(part) == 0:
                    raise TargetPackError(f"missing {split}/{side} rows for {fold['name']}")
                for target in binary_targets:
                    values = part[target]
                    support_rows.append(
                        {
                            "fold": fold["name"],
                            "split": split,
                            "side_name": side,
                            "target": target,
                            "target_kind": "binary" if target != "target_conditional_severe_loss_mask" else "mask",
                            "rows": int(len(part)),
                            "eligible_rows": int(values.notna().sum()),
                            "positive_rows": int(values.sum()),
                            "prevalence": float(values.mean()),
                            "mean_target": float(values.mean()),
                        }
                    )
                severity = part["target_conditional_severe_loss_log1p_100bps"]
                support_rows.append(
                    {
                        "fold": fold["name"],
                        "split": split,
                        "side_name": side,
                        "target": "target_conditional_severe_loss_log1p_100bps",
                        "target_kind": "conditional_regression",
                        "rows": int(len(part)),
                        "eligible_rows": int(severity.notna().sum()),
                        "positive_rows": int(severity.notna().sum()),
                        "prevalence": float(severity.notna().mean()),
                        "mean_target": float(severity.mean()),
                    }
                )
                for class_name in EXIT_CLASSES:
                    class_rows.append(
                        {
                            "fold": fold["name"],
                            "split": split,
                            "side_name": side,
                            "target": "target_deployed_exit_economics_class",
                            "class_name": class_name,
                            "rows": int(len(part)),
                            "class_rows": int(part["target_deployed_exit_economics_class"].eq(class_name).sum()),
                            "prevalence": float(part["target_deployed_exit_economics_class"].eq(class_name).mean()),
                        }
                    )
    return pd.DataFrame(support_rows), pd.DataFrame(class_rows)


def target_roles() -> dict[str, Any]:
    targets = target_columns()
    return {
        "schema": "canonical_execution_reliability_target_roles_v1",
        "identity_and_timestamps": [*IDENTITY, TIME, END, "label_available_at_utc"],
        "target_only_never_features": targets,
        "outcome_or_mask_columns_never_features": targets,
        "input_prohibition": "No target, mask, outcome class, or target-pack column may be used as an execution-EV, timing, action, mapping, policy, or portfolio input.",
        "availability": "Every target is available only at execution_label_end_utc (canonical decision + 12h), including rows whose deployed exit occurs earlier.",
        "pre_exit_opportunity": {
            "source": "spread-adjusted 1m executable pre-exit MFE truncated inclusive at deployed exit minute",
            "formula": "1[pre_exit_mfe_return > execution_cost_return + buffer]",
            "buffers_return": {suffix: buffer for suffix, buffer in OPPORTUNITY_BUFFERS},
            "primary_head": "target_pre_exit_opportunity_25bps",
            "robustness_only": ["target_pre_exit_opportunity_0bps", "target_pre_exit_opportunity_50bps"],
            "parity_mask": "not applied: opportunity is distinct from capture and remains valid on capture-parity failures",
        },
        "deployed_exit_targets": {
            "successful_trailing": "exit_class == trailing AND canonical net > 0",
            "full_stop": "exit_class == full_stop",
            "timeout": "exit_class == timeout",
            "hard_adverse": "exit_class in {full_stop, adverse_exit}",
            "other_adverse": {
                "column": "target_deployed_other_adverse_exit_attribution_only",
                "rule": "exit_class == adverse_exit",
                "training": "FORBIDDEN_STANDALONE_SIDE_LOCAL_HEAD",
                "reason": "March short support is zero and March long support is only 27; use realised attribution or merge into hard_adverse only.",
            },
        },
        "severe_loss_hurdle": {
            "event": "target_severe_loss_100bps = 1[canonical net <= -0.01]",
            "conditional_target": "target_conditional_severe_loss_log1p_100bps = log1p((-canonical net)/0.01), defined only where severe event = 1",
            "inverse": "0.01 * expm1(prediction)",
            "metrics": "conditioned MAE/RMSE after inverse transform, conditioned rank IC and bias; never score missing/non-severe rows as zero severity",
        },
        "exhaustive_class": {
            "column": "target_deployed_exit_economics_class",
            "classes": list(EXIT_CLASSES),
            "semantics": "successful trailing / nonpositive trailing / full-stop-or-adverse / timeout; exactly one class per row",
        },
        "promotion": "TARGET_ONLY_RESEARCH_NO_PROMOTION_NO_POLICY_OR_PORTFOLIO_REPLAY",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    source_manifest = verify_sealed(args.source, "canonical_execution_reliability_input_v4")
    config = load_config(args.config)
    source_panel = args.source / "panel.parquet"
    labels = build_labels(pd.read_parquet(source_panel))
    if len(labels) != 110_730 or labels.duplicated(["candidate_id", "side_name"]).any():
        raise TargetPackError("exact v4 identity coverage failed")
    support, class_support = support_ledgers(labels, config["outer_folds"])
    roles = target_roles()
    if set(target_columns()).intersection(roles["identity_and_timestamps"]):
        raise TargetPackError("target escaped into identity/timestamp contract")

    stage = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        labels.to_parquet(stage / "labels.parquet", index=False, compression="zstd")
        support.to_csv(stage / "support_by_fold_side.csv", index=False)
        class_support.to_csv(stage / "class_support_by_fold_side.csv", index=False)
        write_json(stage / "target_roles.json", roles)
        outputs = {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}
        manifest = {
            "schema": "canonical_execution_reliability_target_pack_v1",
            "run_id": args.output_dir.name,
            "status": "SEALED_TARGET_ONLY_NO_MODEL_INPUT_NO_PROMOTION_NO_POLICY_OR_PORTFOLIO_REPLAY",
            "promotion_eligible": False,
            "rows": int(len(labels)),
            "input_sha256": {
                "source_manifest": sha256(args.source / "manifest.json"),
                "source_panel": source_manifest["outputs_sha256"]["panel.parquet"],
                "source_feature_roles": source_manifest["outputs_sha256"]["feature_roles.json"],
            },
            "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
            "target_contract": roles,
            "outputs_sha256": outputs,
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "limitations": [
                "Other adverse exit is attribution-only and is forbidden as a standalone side-local head.",
                "All labels are future outcomes resolved at canonical +12h and are never model inputs.",
                "The 0bps and 50bps opportunity labels are fixed robustness labels; only +25bps is the primary candidate head.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "  manifest.json\n")
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source", type=Path, default=SOURCE)
    command.add_argument("--config", type=Path, default=CONFIG)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
