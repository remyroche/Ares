#!/usr/bin/env python3
"""Run a fixed-contract meta ablation for short-breakout path context.

This is deliberately narrower than a general meta retrain.  It holds the
candidate universe, side-local selected feature lists, LightGBM parameters,
labels, OOS folds, and cost/outcome ledger fixed.  The only varying inputs are
the leakage-safe OOF breakout path probabilities and their pre-entry
reliability measures.

The path fields are exposed *only* on ``short_breakout_precision`` rows.  They
remain missing for other archetypes rather than being globally zero-filled.
LightGBM can route missing values without turning an inactive field into a
synthetic signal for unrelated archetypes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    _fit_base_soft_label_model,
    _predict,
    _projected_handoff_columns_for_selected,
    run_smoke,
)


DEFAULT_PARENT_MANIFEST = Path(
    "data_perp/reports/meta_v9_recovery_20260713/side_archetype_mda_hpo150/"
    "incumbent_full_oos/manifest.json"
)
DEFAULT_CONTEXT = Path(
    "data_perp/reports/breakout_path_quality_context_20260713_v1/"
    "path_quality_context_oof_2025q3_2026q2.parquet"
)
DEFAULT_OUT_ROOT = Path(
    "data_perp/reports/breakout_path_quality_meta_ablation_20260713_v1"
)
TARGET_SIDE = "short"
TARGET_ARCHETYPE = "short_breakout_precision"

RAW_PATH_FIELDS = (
    "breakout_rapid_reversal_probability_ebm",
    "breakout_rapid_reversal_probability_reliability",
    "breakout_severe_retention_probability_ebm",
    "breakout_severe_retention_probability_reliability",
)
RELIABLE_RISK_FIELDS = (
    "breakout_rapid_reversal_reliable_risk",
    "breakout_severe_retention_reliable_risk",
)
UNCERTAIN_RISK_FIELDS = (
    "breakout_rapid_reversal_uncertain_risk",
    "breakout_severe_retention_uncertain_risk",
)


@dataclass(frozen=True)
class Arm:
    name: str
    fields: tuple[str, ...]
    description: str
    local_target_model: bool = False


ARMS = {
    "baseline": Arm("baseline", (), "Frozen parent feature contract."),
    "raw_path_fields": Arm(
        "raw_path_fields",
        RAW_PATH_FIELDS,
        "Four OOF path probability/reliability fields.",
    ),
    "reliable_risk": Arm(
        "reliable_risk",
        RAW_PATH_FIELDS + RELIABLE_RISK_FIELDS,
        "Raw fields plus high-probability/high-reliability interactions.",
    ),
    "full_path_context": Arm(
        "full_path_context",
        RAW_PATH_FIELDS + RELIABLE_RISK_FIELDS + UNCERTAIN_RISK_FIELDS,
        "Raw fields plus reliable- and uncertain-risk interactions.",
    ),
    "local_short_breakout_baseline": Arm(
        "local_short_breakout_baseline",
        (),
        "Local short-breakout model using only the frozen parent feature contract.",
        local_target_model=True,
    ),
    "local_short_breakout_path": Arm(
        "local_short_breakout_path",
        RAW_PATH_FIELDS,
        "Local short-breakout model with the four frozen path-context fields.",
        local_target_model=True,
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalise_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip()


class BreakoutPathContext:
    """Frozen OOF context, keyed by pre-entry candidate identity."""

    key_columns = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")

    def __init__(self, path: Path) -> None:
        raw = pd.read_parquet(path)
        required = {
            "__ts__",
            "__symbol__",
            "side_name",
            "__archetype_policy_key__",
            *RAW_PATH_FIELDS,
        }
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise KeyError(f"Path context missing required fields: {missing}")
        context = raw.loc[:, [
            "__ts__",
            "__symbol__",
            "side_name",
            "__archetype_policy_key__",
            *RAW_PATH_FIELDS,
        ]].copy()
        context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True, errors="coerce")
        context["side_name"] = _normalise_key(context["side_name"]).str.lower()
        context["archetype_policy_key"] = _normalise_key(
            context.pop("__archetype_policy_key__")
        )
        context["__symbol__"] = _normalise_key(context["__symbol__"])
        if context.duplicated(list(self.key_columns)).any():
            raise ValueError("Path context is not unique on its declared pre-entry key.")
        for field in RAW_PATH_FIELDS:
            context[field] = pd.to_numeric(context[field], errors="coerce").astype(np.float32)
        self._context = context.set_index(list(self.key_columns)).sort_index()
        self.path = Path(path)

    @staticmethod
    def _archetype(frame: pd.DataFrame) -> pd.Series:
        for name in ("archetype_policy_key", "__archetype_policy_key__", "policy_archetype"):
            if name in frame.columns:
                return _normalise_key(frame[name])
        raise KeyError("Meta handoff has no archetype policy key for path-context scoping.")

    def attach(self, frame: pd.DataFrame, fields: Iterable[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        requested = tuple(dict.fromkeys(str(field) for field in fields))
        out = frame.copy(deep=False)
        if not requested:
            # The baseline arm must be a literal frozen-parent replay.  Do not
            # even materialize inactive path columns merely to report coverage.
            return out, {
                "active_rows": 0,
                "matched_rows": 0,
                "coverage_active_rows": float("nan"),
                "inactive_non_null_rows": 0,
                "status": "not_applicable_baseline",
            }
        # Preserve all non-target rows as missing.  This is deliberate: a zero
        # would mean known benign risk and leak the active-archetype identity.
        for field in requested:
            out[field] = np.float32(np.nan)

        side = _normalise_key(out["side_name"]).str.lower()
        archetype = self._archetype(out)
        active = side.eq(TARGET_SIDE) & archetype.eq(TARGET_ARCHETYPE)
        if bool(active.any()):
            keys = pd.MultiIndex.from_arrays(
                [
                    pd.to_datetime(out.loc[active, "__ts__"], utc=True, errors="coerce"),
                    _normalise_key(out.loc[active, "__symbol__"]),
                    side.loc[active],
                    archetype.loc[active],
                ],
                names=list(self.key_columns),
            )
            matched = self._context.reindex(keys)
            raw = matched.loc[:, list(RAW_PATH_FIELDS)].copy()
            raw["breakout_rapid_reversal_reliable_risk"] = (
                raw["breakout_rapid_reversal_probability_ebm"]
                * raw["breakout_rapid_reversal_probability_reliability"]
            )
            raw["breakout_severe_retention_reliable_risk"] = (
                raw["breakout_severe_retention_probability_ebm"]
                * raw["breakout_severe_retention_probability_reliability"]
            )
            raw["breakout_rapid_reversal_uncertain_risk"] = (
                raw["breakout_rapid_reversal_probability_ebm"]
                * (1.0 - raw["breakout_rapid_reversal_probability_reliability"])
            )
            raw["breakout_severe_retention_uncertain_risk"] = (
                raw["breakout_severe_retention_probability_ebm"]
                * (1.0 - raw["breakout_severe_retention_probability_reliability"])
            )
            for field in requested:
                out.loc[active, field] = raw[field].to_numpy(dtype=np.float32, copy=False)

        active_rows = int(active.sum())
        matched_rows = int(out.loc[active, list(RAW_PATH_FIELDS)].notna().all(axis=1).sum()) if active_rows else 0
        # All rows outside scope must remain absent, never encoded as a zero.
        if requested and out.loc[~active, list(requested)].notna().any(axis=None):
            raise AssertionError("Path context escaped short_breakout_precision scope.")
        return out, {
            "active_rows": active_rows,
            "matched_rows": matched_rows,
            "coverage_active_rows": float(matched_rows / active_rows) if active_rows else float("nan"),
            "inactive_non_null_rows": int(out.loc[~active, list(requested)].notna().any(axis=1).sum()) if requested else 0,
        }


def _fixed_features(parent: dict[str, Any], arm: Arm) -> dict[str, list[str]]:
    source = parent.get("selected_features_by_side") or {}
    if not {"long", "short"}.issubset(source):
        raise ValueError("Parent manifest lacks fixed long/short selected feature lists.")
    out = {side: list(dict.fromkeys(map(str, features))) for side, features in source.items()}
    out["short"] = list(dict.fromkeys([*out["short"], *arm.fields]))
    return {side: out[side] for side in ("long", "short")}


def _fold_builder(context: BreakoutPathContext, arm: Arm):
    def build(*, train, valid, fold, month, valid_start, valid_end, selected_col):
        del month, valid_start, valid_end, selected_col
        train_out, train_coverage = context.attach(train, arm.fields)
        valid_out, valid_coverage = context.attach(valid, arm.fields)
        return train_out, valid_out, list(arm.fields), {
            "fold": str(fold),
            "path_context": {
                "scope": {"side": TARGET_SIDE, "archetype": TARGET_ARCHETYPE},
                "train": train_coverage,
                "valid": valid_coverage,
                "fields": list(arm.fields),
                "source": str(context.path),
            },
        }

    return build


def _local_short_breakout_override(arm: Arm):
    """Replace only the target archetype's parent score with a local model.

    The local prediction is deliberately trained on the same soft label and
    parent feature contract.  It receives no realized validation outcomes and
    leaves every non-target row on the frozen side-level parent score.
    """

    def archetype(frame: pd.DataFrame) -> pd.Series:
        return BreakoutPathContext._archetype(frame)

    def override(
        *,
        x_train: pd.DataFrame,
        train: pd.DataFrame,
        x_valid: pd.DataFrame,
        scored: pd.DataFrame,
        base_target: pd.Series,
        feature_names_by_side: dict[str, list[str]],
        classifier_params: dict[str, Any],
        fold: str,
        seed: int,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]], dict[str, Any]]:
        side_features = list(feature_names_by_side[TARGET_SIDE])
        train_target = (
            train["side_name"].astype(str).str.lower().eq(TARGET_SIDE)
            & archetype(train).eq(TARGET_ARCHETYPE)
        )
        valid_target = (
            scored["side_name"].astype(str).str.lower().eq(TARGET_SIDE)
            & archetype(scored).eq(TARGET_ARCHETYPE)
        )
        if int(train_target.sum()) < 100:
            raise RuntimeError(
                f"{fold}: only {int(train_target.sum())} short-breakout train rows; "
                "local ablation requires at least 100."
            )
        out = scored.copy(deep=False)
        out["score_meta_base_soft_label_parent"] = pd.to_numeric(
            out["score_meta_base_soft_label"], errors="coerce"
        ).astype(np.float32)
        out["score_meta_base_soft_label_local_short_breakout"] = np.float32(np.nan)
        local_model = _fit_base_soft_label_model(
            x_train.loc[train_target, side_features],
            base_target.loc[train_target],
            train.loc[train_target],
            int(seed) + 70_001,
            lgbm_params=classifier_params,
        )
        if bool(valid_target.any()):
            local_score = _predict(
                local_model,
                x_valid.loc[valid_target, side_features],
                classifier=False,
            ).to_numpy(dtype=np.float32)
            out.loc[valid_target, "score_meta_base_soft_label_local_short_breakout"] = local_score
            out.loc[valid_target, "score_meta_base_soft_label"] = local_score
        return (
            out,
            {"base_soft_label_short_breakout_local": local_model},
            {"base_soft_label_short_breakout_local": side_features},
            {
                "scope": {"side": TARGET_SIDE, "archetype": TARGET_ARCHETYPE},
                "train_rows": int(train_target.sum()),
                "valid_rows": int(valid_target.sum()),
                "feature_count": int(len(side_features)),
                "score_contract": (
                    "Target-archetype rows use the local soft-label model; all other rows retain "
                    "the frozen parent side-model score."
                ),
            },
        )

    return override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--path-context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--eval-months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--model-train-max-rows", type=int, default=0)
    parser.add_argument("--minimal-artifacts", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent = _load_json(args.parent_manifest)
    context = BreakoutPathContext(args.path_context)
    requested = [name.strip() for name in str(args.arms).split(",") if name.strip()]
    unknown = sorted(set(requested).difference(ARMS))
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}; available={sorted(ARMS)}")
    eval_months = [part.strip() for part in str(args.eval_months).split(",") if part.strip()]
    handoff_path = Path(parent["handoff_path"])
    ledger_path = Path(parent["ledger_path"])
    base_features = parent.get("selected_feature_union") or []
    args.out_root.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, Any] = {
        "generated_by": "run_breakout_path_quality_meta_ablation",
        "parent_manifest": str(args.parent_manifest),
        "handoff_path": str(handoff_path),
        "ledger_path": str(ledger_path),
        "path_context": str(args.path_context),
        "eval_months": eval_months,
        "arms": [],
        "fixed_contract": {
            "candidate_universe": "parent selected_top30 handoff",
            "feature_selection": "disabled; parent side-local selected lists reused",
            "hpo": "disabled; parent LightGBM parameters reused",
            "outcome_and_cost_ledger": "identical parent ledger",
            "path_context_scope": {"side": TARGET_SIDE, "archetype": TARGET_ARCHETYPE},
            "path_context_non_target_rows": "missing; never zero-filled",
            "path_context_oos": "frozen EBM probability/reliability outputs only",
        },
    }
    for name in requested:
        arm = ARMS[name]
        out_dir = args.out_root / arm.name
        done = out_dir / "manifest.json"
        if done.exists() and not args.rerun_complete:
            matrix["arms"].append({"arm": arm.name, "out_dir": str(out_dir), "skipped_complete": True})
            continue
        features_by_side = _fixed_features(parent, arm)
        projected = _projected_handoff_columns_for_selected(
            handoff_path,
            [*features_by_side["long"], *features_by_side["short"]],
        )
        if projected is not None:
            projected.extend(["candidate_id", "archetype_policy_key", "__archetype_policy_key__"])
            projected = list(dict.fromkeys(projected))
        print(json.dumps({
            "event": "breakout_path_meta_ablation_arm_start",
            "arm": arm.name,
            "fields": list(arm.fields),
            "short_feature_count": len(features_by_side["short"]),
            "long_feature_count": len(features_by_side["long"]),
            "eval_months": eval_months,
        }, sort_keys=True), flush=True)
        manifest = run_smoke(
            handoff_dir=handoff_path.parent,
            handoff_path=handoff_path,
            ledger_path=ledger_path,
            out_dir=out_dir,
            frontier="top30",
            seed=int(args.seed),
            train_scope="selected",
            enable_base_prior_features=bool(parent.get("enable_base_prior_features", True)),
            enable_reliability_features=bool(parent.get("enable_reliability_features", True)),
            enable_support_drift_features=bool(parent.get("enable_support_drift_features", True)),
            enable_hit_surprise_features=bool(parent.get("enable_hit_surprise_features", True)),
            feature_selection_top_n=0,
            feature_selection_method="lgbm_pipeline",
            validation_scope="chronological",
            model_train_max_rows=int(args.model_train_max_rows),
            model_params={
                "classifier": dict(parent["classifier_params"]),
                "regressor": dict(parent.get("regressor_params") or parent["classifier_params"]),
            },
            model_profile_name=f"breakout_path_context_{arm.name}_fixed_parent",
            meta_head_mode="single_base_soft_label",
            fixed_selected_features_by_side=features_by_side,
            eval_months=eval_months,
            fold_feature_builder=_fold_builder(context, arm),
            fold_feature_profile_name=f"breakout_path_context_{arm.name}",
            extra_prediction_columns=[
                *arm.fields,
                *(
                    [
                        "score_meta_base_soft_label_parent",
                        "score_meta_base_soft_label_local_short_breakout",
                    ]
                    if arm.local_target_model
                    else []
                ),
            ],
            force_prediction_shards=True,
            combine_prediction_shards=False,
            save_fold_models=True,
            minimal_artifacts=bool(args.minimal_artifacts),
            handoff_columns=projected,
            ood_reference_features=base_features,
            single_head_score_override=(
                _local_short_breakout_override(arm) if arm.local_target_model else None
            ),
        )
        manifest["breakout_path_context_ablation"] = {
            "arm": arm.name,
            "description": arm.description,
            "fields": list(arm.fields),
            "scope": {"side": TARGET_SIDE, "archetype": TARGET_ARCHETYPE},
            "local_target_model": bool(arm.local_target_model),
            "contract": matrix["fixed_contract"],
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
        )
        matrix["arms"].append({
            "arm": arm.name,
            "out_dir": str(out_dir),
            "fields": list(arm.fields),
            "selected_short_feature_count": len(features_by_side["short"]),
            "selected_long_feature_count": len(features_by_side["long"]),
        })
        (args.out_root / "matrix_manifest.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "breakout_path_meta_ablation_done", "out_root": str(args.out_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
