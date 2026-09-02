#!/usr/bin/env python3
"""Seal a no-reranking pre-entry action-learning handoff.

The handoff starts from the already frozen pooled-global monthly book.  It
adds only authorised decision-time inputs, exact future action targets, and
the exact 720x1m counterfactual path.  Selection identities, ranks, weights,
and sizing are immutable; future outcomes are explicitly target-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
PANEL_ROOT = ART / "canonical_execution_reliability_input_20260730_v4"
TARGET_ROOT = ART / "execution_action_target_pack_20260730_v2"
PATH_ROOT = ART / "febapr2025_top40_exact1m_paths_20260727_v1"
BOOK_ROOT = ART / "frozen_exit_state_action_ablation_20260730_v4"
POLICY_INPUT_ROOT = ART / "febapr2025_execution_ev_deployed_policy_inputs_20260727_v1"
OUT = ART / "frozen_entry_action_handoff_20260730_v2"

SCHEMA = "frozen_entry_action_handoff_v2"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
PATH_JOIN = ("candidate_id", "side_name")
WEIGHTS = ("weight_top_01", "weight_top_05", "weight_top_10", "weight_top_20")
BOOK_COLUMNS = (
    *IDENTITY,
    "execution_decision_utc",
    "candidate_month",
    "mapped_score",
    "mapped_eligible",
    "gross__deployed",
    "net__deployed",
    "cost__deployed",
    "execution_exit_reason",
    *WEIGHTS,
)
PATH_COLUMNS = (
    *IDENTITY,
    "execution_future_path",
    "atr_1h",
    "decision_price",
    "fee",
    "entry_spread",
    "exit_spread",
)


class ContractError(RuntimeError):
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def verify_seal(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file():
        raise ContractError(f"missing manifest: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise ContractError(f"schema mismatch: {root}")
    if seal_path.is_file():
        if sha256(manifest_path) != seal_path.read_text().split()[0]:
            raise ContractError(f"manifest seal mismatch: {root}")
        for name, expected in manifest.get("outputs_sha256", {}).items():
            if sha256(root / name) != expected:
                raise ContractError(f"sealed output mismatch: {root / name}")
    else:
        # Historical exact-path packs predate manifest.sha256.  Their signed
        # prediction-role manifest binds the exact parquet payload instead.
        signed = manifest.get("prediction_role_manifest_sha256")
        canonical = {
            str(key): safe(value)
            for key, value in manifest.items()
            if key != "prediction_role_manifest_sha256"
        }
        actual_signature = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if not isinstance(signed, str) or signed != actual_signature:
            raise ContractError(f"signed manifest mismatch: {root}")
        artifact_hash = manifest.get("source_artifact_sha256")
        if not isinstance(artifact_hash, str) or sha256(root / "paths.parquet") != artifact_hash:
            raise ContractError(f"signed path output mismatch: {root / 'paths.parquet'}")
    return manifest


def _utc(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ContractError(f"{name} contains invalid UTC values")
    return result


def _identity(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ContractError(f"{source} missing identity columns: {missing}")
    work = frame.copy()
    work["candidate_id"] = work["candidate_id"].astype("string").str.strip()
    work["side_name"] = work["side_name"].astype("string").str.strip().str.lower()
    work["__symbol__"] = work["__symbol__"].astype("string").str.strip()
    work["__ts__"] = _utc(work["__ts__"], f"{source}.__ts__")
    if (
        work["candidate_id"].isna().any()
        or work["candidate_id"].eq("").any()
        or work["__symbol__"].isna().any()
        or work["__symbol__"].eq("").any()
        or not work["side_name"].isin(("long", "short")).all()
    ):
        raise ContractError(f"{source} has invalid identity values")
    duplicates = int(work.duplicated(list(IDENTITY), keep=False).sum())
    if duplicates:
        raise ContractError(f"{source} has {duplicates} duplicate exact identities")
    return work


def _normal_symbol(value: Any) -> str:
    return str(value).strip().replace("_", "/", 1)


def _assert_exact_coverage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    source: str,
) -> None:
    coverage = left.loc[:, list(keys)].merge(
        right.loc[:, list(keys)],
        on=list(keys),
        how="outer",
        indicator=True,
        sort=False,
    )
    missing = int(coverage["_merge"].eq("left_only").sum())
    extra = int(coverage["_merge"].eq("right_only").sum())
    if missing or extra:
        raise ContractError(
            f"{source} exact coverage mismatch: missing={missing}, unexpected={extra}"
        )


def _finite(frame: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ContractError(f"{source}.{column} must be finite")


def _feature_provenance(features: Sequence[str]) -> dict[str, dict[str, Any]]:
    support = {name for name in features if name.startswith("support__")}
    predictive = {
        "raw_score",
        "score_base_alpha",
        "score_residual_expected_ev",
        "direct_q25_return",
        "base_oof_score",
        *support,
    }
    records: dict[str, dict[str, Any]] = {}
    for name in features:
        if name in support:
            source = "sealed side-local support OOF in March; frozen-forward in April"
            available = "score_available_utc"
            fold = "support_fold"
            cutoff = "support_train_decision_max_utc"
        elif name in predictive:
            source = "sealed upstream outer-OOF/residual score lineage"
            available = "score_available_utc"
            fold = "residual_fold_x"
            cutoff = "fold_train_cutoff_utc"
        else:
            source = "deterministic decision-time candidate/regime context"
            available = "__decision_ts__"
            fold = None
            cutoff = None
        records[name] = {
            "role": "model_input",
            "source": source,
            "pre_entry": True,
            "available_at_col": available,
            "oof_fold_col": fold,
            "source_train_cutoff_col": cutoff,
            "april_status": (
                "frozen_forward_oos" if name in predictive else "decision_time_observable"
            ),
        }
    return records


def materialize(
    panel_root: Path,
    target_root: Path,
    path_root: Path,
    book_root: Path,
    policy_input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    panel_manifest = verify_seal(panel_root, "canonical_execution_reliability_input_v4")
    target_manifest = verify_seal(target_root, "execution_action_target_pack_v2")
    path_manifest = verify_seal(path_root, "execution_entry_timing_1m_paths_v1")
    book_manifest = verify_seal(book_root, "frozen_exit_state_action_ablation_v4")
    policy_input_manifest_path = policy_input_root / "manifest.json"
    if not policy_input_manifest_path.is_file():
        raise ContractError(f"missing policy input manifest: {policy_input_root}")
    policy_input_manifest = json.loads(policy_input_manifest_path.read_text())
    if policy_input_manifest.get("schema") != "historical_execution_ev_deployed_policy_inputs_v1":
        raise ContractError("policy input schema mismatch")
    for record in policy_input_manifest.get("outputs", {}).values():
        source = policy_input_root / Path(str(record["path"])).name
        if sha256(source) != record.get("sha256"):
            raise ContractError(f"policy input output mismatch: {source}")
    policy_output_hashes = {
        Path(str(record["path"])).name: str(record["sha256"])
        for record in policy_input_manifest.get("outputs", {}).values()
    }
    if output_root.exists():
        raise ContractError(f"output already exists: {output_root}")

    feature_roles = json.loads((panel_root / "feature_roles.json").read_text())
    features = tuple(feature_roles["default_ev_inputs"])
    if len(features) != len(set(features)):
        raise ContractError("default feature list contains duplicates")
    forbidden_tokens = ("target_", "execution_future", "realized", "label_")
    leaked = [name for name in features if any(token in name.lower() for token in forbidden_tokens)]
    if leaked:
        raise ContractError(f"future/target columns entered model feature list: {leaked}")

    book = _identity(
        pd.read_parquet(book_root / "paired_candidates.parquet", columns=list(BOOK_COLUMNS)),
        "frozen book",
    )
    _finite(book, (*WEIGHTS, "mapped_score"), "frozen book")
    if not book["mapped_eligible"].astype(bool).all():
        raise ContractError("frozen book contains mapped-ineligible rows")
    for weight in WEIGHTS:
        values = book[weight].to_numpy(dtype=float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ContractError(f"{weight} must be in [0,1]")

    panel_columns = list(
        dict.fromkeys(
            [
                *IDENTITY,
                "execution_decision_utc",
                "execution_label_end_utc",
                "candidate_month",
                "score_available_utc",
                "fold_train_cutoff_utc",
                "training_label_resolved_max_utc",
                "residual_fold_x",
                "residual_is_oof",
                "upstream_scores_are_outer_oof",
                "candidate_score_is_oof",
                "candidate_score_is_forward_oos",
                "support_fold",
                "support_train_decision_max_utc",
                "support_train_label_end_max_utc",
                "support_scores_are_chronological_oof",
                "support_scores_are_frozen_forward",
                *features,
            ]
        )
    )
    panel = _identity(
        pd.read_parquet(panel_root / "panel.parquet", columns=panel_columns),
        "canonical panel",
    )
    selected_panel = book.loc[:, list(IDENTITY)].merge(
        panel, on=list(IDENTITY), how="left", validate="one_to_one", sort=False
    )
    if selected_panel["execution_decision_utc"].isna().any():
        raise ContractError("frozen book has identities absent from canonical panel")
    for column in ("execution_decision_utc", "execution_label_end_utc", "score_available_utc"):
        selected_panel[column] = _utc(selected_panel[column], f"panel.{column}")
    if not (
        selected_panel["execution_label_end_utc"]
        == selected_panel["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise ContractError("panel label horizon is not exactly 12 hours")
    if not (
        selected_panel["score_available_utc"] <= selected_panel["execution_decision_utc"]
    ).all():
        raise ContractError("a model score is available after action decision time")
    if not (
        selected_panel["residual_is_oof"].astype(bool)
        & selected_panel["upstream_scores_are_outer_oof"].astype(bool)
    ).all():
        raise ContractError("selected rows lack upstream OOF evidence")
    _finite(selected_panel, features, "selected panel")

    targets = _identity(
        pd.read_parquet(target_root / "labels.parquet"),
        "action target pack",
    )
    _assert_exact_coverage(panel, targets, IDENTITY, "panel/target pack")
    target_columns = [name for name in targets.columns if name not in IDENTITY]
    selected = book.merge(
        selected_panel,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
        suffixes=("_book", ""),
        sort=False,
    )
    selected = selected.merge(
        targets,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        suffixes=("", "_target"),
        sort=False,
    )
    if selected["canonical_cost_return"].isna().any():
        raise ContractError("selected book lacks action targets")

    paths = _identity(
        pd.read_parquet(path_root / "paths.parquet", columns=list(PATH_COLUMNS)),
        "exact 1m paths",
    )
    if paths.duplicated(list(PATH_JOIN), keep=False).any():
        raise ContractError("exact path source is not unique by candidate_id and side")
    path_payload = paths.rename(
        columns={"__symbol__": "path_symbol", "__ts__": "path_signal_utc"}
    )
    selected = selected.merge(
        path_payload,
        on=list(PATH_JOIN),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if selected["execution_future_path"].isna().any():
        raise ContractError("selected book lacks exact path rows")
    signal = _utc(selected["path_signal_utc"], "paths.__ts__")
    if not (signal == selected["__ts__"]).all():
        raise ContractError("path signal timestamp differs from canonical identity")
    symbol_match = np.fromiter(
        (
            _normal_symbol(left) == _normal_symbol(right)
            for left, right in zip(selected["__symbol__"], selected["path_symbol"])
        ),
        dtype=bool,
        count=len(selected),
    )
    if not symbol_match.all():
        raise ContractError("normalized path symbol differs from canonical identity")
    _finite(
        selected,
        ("atr_1h", "decision_price", "fee", "entry_spread", "exit_spread"),
        "exact paths",
    )
    if (selected["atr_1h"] <= 0.0).any() or (selected["decision_price"] <= 0.0).any():
        raise ContractError("path ATR and decision price must be positive")

    policy_context = _identity(
        pd.read_parquet(policy_input_root / "context.parquet"),
        "deployed policy context",
    )
    policy_targets = _identity(
        pd.read_parquet(policy_input_root / "path_targets.parquet"),
        "deployed policy path targets",
    )
    if policy_context["candidate_id"].duplicated().any():
        raise ContractError("deployed policy context candidate_id is not unique")
    if policy_targets["candidate_id"].duplicated().any():
        raise ContractError("deployed policy target candidate_id is not unique")
    execution_inputs = policy_context.loc[
        :, [*IDENTITY, "policy_archetype"]
    ].merge(
        policy_targets.loc[:, [*IDENTITY, "__barrier_pct__", "__path_auxiliary_atr_fraction__"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_context", ""),
    )
    selected = selected.merge(
        execution_inputs,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_policy"),
    )
    if selected["__barrier_pct__"].isna().any():
        raise ContractError("selected book lacks deployed-policy execution inputs")
    for column in ("side_name", "__ts__", "__symbol__"):
        left = selected[column]
        right = selected[f"{column}_policy"]
        if column == "__ts__":
            equal = _utc(left, column).eq(_utc(right, f"{column}_policy"))
        elif column == "__symbol__":
            equal = pd.Series(
                [_normal_symbol(a) == _normal_symbol(b) for a, b in zip(left, right)]
            )
        else:
            equal = left.astype(str).eq(right.astype(str))
        if not equal.all():
            raise ContractError(f"deployed-policy {column} differs from selected identity")
        selected = selected.drop(columns=[f"{column}_policy"])
    selected = selected.drop(
        columns=[
            name
            for name in (
                "__ts___context",
                "__symbol___context",
                "side_name_context",
            )
            if name in selected
        ]
    )
    _finite(
        selected,
        ("__barrier_pct__", "__path_auxiliary_atr_fraction__"),
        "deployed policy inputs",
    )
    if (
        (selected["__barrier_pct__"] <= 0.0).any()
        or (selected["__path_auxiliary_atr_fraction__"] <= 0.0).any()
    ):
        raise ContractError("deployed-policy barrier and ATR inputs must be positive")

    selected["__decision_ts__"] = selected["execution_decision_utc"]
    # Protect one canonical name for each duplicated book/panel field.
    for column in ("execution_decision_utc", "candidate_month"):
        book_column = f"{column}_book"
        if book_column in selected:
            if column.endswith("_utc"):
                equal = _utc(selected[book_column], book_column).eq(
                    _utc(selected[column], column)
                )
            else:
                equal = selected[book_column].astype(str).eq(selected[column].astype(str))
            if not equal.all():
                raise ContractError(f"frozen book {column} differs from canonical panel")
            selected = selected.drop(columns=[book_column])

    target_only = sorted(
        {
            *target_columns,
            "execution_future_path",
            "atr_1h",
            "decision_price",
            "fee",
            "entry_spread",
            "exit_spread",
            "path_symbol",
            "path_signal_utc",
            "gross__deployed",
            "net__deployed",
            "cost__deployed",
            "execution_exit_reason",
        }
    )
    execution_only = [
        "__barrier_pct__",
        "__path_auxiliary_atr_fraction__",
        "policy_archetype",
    ]
    selection_only = [
        "mapped_score",
        "mapped_eligible",
        *WEIGHTS,
    ]
    if (
        set(features).intersection(target_only)
        or set(features).intersection(selection_only)
        or set(features).intersection(execution_only)
    ):
        raise ContractError(
            "model inputs overlap target, frozen-selection, or execution-only fields"
        )

    audit = (
        selected.groupby(["candidate_month", "side_name"], observed=True)
        .agg(
            rows=("candidate_id", "size"),
            top01_weight=("weight_top_01", "sum"),
            top05_weight=("weight_top_05", "sum"),
            top10_weight=("weight_top_10", "sum"),
            top20_weight=("weight_top_20", "sum"),
            first_decision=("execution_decision_utc", "min"),
            last_decision=("execution_decision_utc", "max"),
        )
        .reset_index()
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        handoff_path = temporary / "handoff.parquet"
        selected.to_parquet(handoff_path, index=False, compression="zstd")
        audit.to_csv(temporary / "coverage.csv", index=False)
        roles = {
            "schema": "frozen_entry_action_feature_roles_v2",
            "model_inputs": list(features),
            "feature_provenance": _feature_provenance(features),
            "selection_only_never_model_inputs": selection_only,
            "target_only_never_model_inputs": target_only,
            "execution_only_never_model_inputs": execution_only,
            "contract": {
                "ranking": "frozen pooled-global monthly book; no action reranking or backfill",
                "availability": "all model inputs are available no later than execution decision",
                "training": "March candidate/support predictions are chronological OOF; April is frozen forward OOS",
                "labels": "future paths and all target fields are train/report-only",
                "exit_policy": "exact deployed simple-policy barrier/archetype inputs retained only for counterfactual replay",
            },
        }
        write_json(temporary / "feature_roles.json", roles)
        outputs = {
            name: sha256(temporary / name)
            for name in ("handoff.parquet", "coverage.csv", "feature_roles.json")
        }
        identity_digest = hashlib.sha256(
            pd.util.hash_pandas_object(
                selected.loc[:, [*IDENTITY, *WEIGHTS]], index=False
            ).to_numpy(dtype=np.uint64).tobytes()
        ).hexdigest()
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_DIAGNOSTIC_ONLY_FROZEN_GLOBAL_BOOK_NO_RERANK_NO_PROMOTION",
            "rows": int(len(selected)),
            "identity_weight_digest": identity_digest,
            "contract": roles["contract"],
            "feature_count": len(features),
            "target_only_count": len(target_only),
            "promotion_eligible": False,
            "limitations": [
                "March/April are reused diagnostic months and cannot select a deployable action.",
                "The handoff materializes pre-entry actions only; learned post-entry routing still requires causal prefix-state snapshots.",
                "April upstream support is frozen-forward rather than row-level chronological OOF.",
            ],
            "input_provenance": {
                "panel_manifest_sha256": sha256(panel_root / "manifest.json"),
                "panel_sha256": panel_manifest["outputs_sha256"]["panel.parquet"],
                "target_manifest_sha256": sha256(target_root / "manifest.json"),
                "targets_sha256": target_manifest["outputs_sha256"]["labels.parquet"],
                "path_manifest_sha256": sha256(path_root / "manifest.json"),
                "paths_sha256": path_manifest.get("source_artifact_sha256"),
                "book_manifest_sha256": sha256(book_root / "manifest.json"),
                "book_sha256": book_manifest["outputs_sha256"]["paired_candidates.parquet"],
                "policy_input_manifest_sha256": sha256(policy_input_manifest_path),
                "policy_context_sha256": policy_output_hashes["context.parquet"],
                "policy_targets_sha256": policy_output_hashes["path_targets.parquet"],
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(
            f"{sha256(temporary / 'manifest.json')}  manifest.json\n"
        )
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    result.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    result.add_argument("--path-root", type=Path, default=PATH_ROOT)
    result.add_argument("--book-root", type=Path, default=BOOK_ROOT)
    result.add_argument("--policy-input-root", type=Path, default=POLICY_INPUT_ROOT)
    result.add_argument("--output-root", type=Path, default=OUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            safe(
                materialize(
                    args.panel_root,
                    args.target_root,
                    args.path_root,
                    args.book_root,
                    args.policy_input_root,
                    args.output_root,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
