#!/usr/bin/env python3
"""Execute the standalone Stage-I base automatic/20/30/40/60/full ladder.

No meta parameters, arms, target heads, or candidate handoff are accepted.  The
command is a fixed-source-HPO diagnostic and writes a per-count HPO/refit
request instead of selecting a winner.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_base_feature_ladder import (
    BaseFeatureLadderConfig,
    BaseFeatureLadderInput,
    run_pooled_base_feature_ladder,
    run_side_base_feature_ladder,
)
from extreme_price_movements.stage_i_nested_feature_challenger import (
    checkpoint_nested_feature_plan,
    load_completed_stage_i_base_selection,
    materialize_nested_feature_challenge,
)
from extreme_price_movements.stage_i_nested_lgbm_hooks import (
    FixedLGBMContract,
    fixed_lgbm_base_predictor,
)
from extreme_price_movements.stage_i_r3_contract import (
    frame_content_sha256,
    require_r3_label_economics_contract,
    selector_validity_mask,
)


IDENTITY = ("candidate_id", "__ts__", "__symbol__")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _valid_target_rows(ledger: pd.DataFrame) -> pd.Series:
    """Keep invalid/unresolved paths out of every supervised base arm."""
    valid = pd.Series(True, index=ledger.index)
    for column, expected in (("target_invalid", False), ("label_valid", True), ("path_complete", True)):
        if column in ledger:
            values = ledger[column].astype(bool)
            valid &= values if expected else ~values
    return valid


def _require_selector_integrity(selector_dir: Path, ledger: pd.DataFrame) -> dict[str, Any]:
    manifest = _json(selector_dir / "manifest.json")
    integrity = manifest.get("artifact_integrity")
    if not isinstance(integrity, dict) or integrity.get("schema") != "stage_i_selector_artifact_integrity_v1":
        raise ValueError("selector manifest lacks immutable artifact-integrity evidence")
    files = {
        "selector_ledger_sha256": selector_dir / "selector_ledger.parquet",
        "selector_features_sha256": selector_dir / "selector_features.parquet",
        "exact_coverage_audit_sha256": selector_dir / "selector_exact_feature_coverage_audit.parquet",
    }
    detail = selector_dir / "selector_exact_feature_month_side_coverage.parquet"
    for key, path in files.items():
        if not path.is_file() or integrity.get(key) != _sha(path):
            raise ValueError(f"selector artifact integrity drift: {key}")
    expected_detail = integrity.get("exact_coverage_month_side_audit_sha256")
    if expected_detail is None:
        if detail.is_file():
            raise ValueError("selector detail coverage audit unexpectedly appeared")
    elif not detail.is_file() or expected_detail != _sha(detail):
        raise ValueError("selector artifact integrity drift: exact_coverage_month_side_audit_sha256")
    current = require_r3_label_economics_contract(
        ledger, str(integrity.get("r3_label_economics_contract_sha256", "")),
    )
    if integrity.get("r3_label_economics_contract") != current:
        raise ValueError("selector R3 label/economics payload drift")
    return integrity


def load_side_input(selector_dir: Path, *, side: str) -> BaseFeatureLadderInput:
    ledger_path = selector_dir / "selector_ledger.parquet"
    matrix_path = selector_dir / "selector_features.parquet"
    manifest_path = selector_dir / "manifest.json"
    contract_path = selector_dir / "selector_feature_contract.json"
    if not all(path.is_file() for path in (ledger_path, matrix_path, manifest_path, contract_path)):
        raise ValueError("corrected selector ledger/matrix/contract inputs are incomplete")
    if _json(manifest_path).get("status") != "complete":
        raise ValueError("selector sample is not complete")
    ledger, matrix = pd.read_parquet(ledger_path), pd.read_parquet(matrix_path)
    integrity = _require_selector_integrity(selector_dir, ledger)
    if not ledger.loc[:, list(IDENTITY)].reset_index(drop=True).equals(matrix.loc[:, list(IDENTITY)].reset_index(drop=True)):
        raise ValueError("corrected selector ledger and matrix identity order differs")
    local = ledger.side_name.astype(str).str.lower().eq(side)
    valid_target = _valid_target_rows(ledger)
    valid = local & valid_target
    if not valid.any():
        raise ValueError(f"{side}: no valid complete-path base target rows")
    local_ledger = ledger.loc[valid].reset_index(drop=True)
    local_matrix = matrix.loc[valid].reset_index(drop=True).drop(columns=list(IDENTITY))
    required = {"side_name", "r3_class", "exact_net_bps", "label_available_ts"}
    if missing := sorted(required.difference(local_ledger.columns)):
        raise ValueError(f"{side}: selector ledger lacks base target contract {missing}")
    raw = pd.concat(
        [local_ledger.loc[:, list(IDENTITY) + ["side_name", "r3_class", "exact_net_bps", "label_available_ts"]], local_matrix],
        axis=1,
    )
    raw["decision_ts"] = pd.to_datetime(raw["__ts__"], utc=True, errors="raise") + pd.Timedelta(hours=1)
    full_side = ledger.loc[local].reset_index(drop=True)
    full_identity = frame_content_sha256(full_side, IDENTITY)
    validity = pd.DataFrame({"supervised_valid": valid_target.loc[local].astype(np.int8).to_numpy()})
    return BaseFeatureLadderInput(
        side=side, frame=raw, base_feature_universe=tuple(local_matrix.columns),
        full_candidate_rows=int(len(full_side)),
        invalid_or_incomplete_rows=int((~valid_target.loc[local]).sum()),
        full_candidate_identity_sha256=full_identity,
        full_candidate_validity_sha256=frame_content_sha256(validity, ("supervised_valid",)),
        source_integrity={
            "selector_ledger_sha256": integrity["selector_ledger_sha256"],
            "selector_features_sha256": integrity["selector_features_sha256"],
            "exact_coverage_audit_sha256": integrity["exact_coverage_audit_sha256"],
            "exact_coverage_month_side_audit_sha256": integrity["exact_coverage_month_side_audit_sha256"],
            "r3_label_economics_contract_sha256": integrity["r3_label_economics_contract_sha256"],
            "r3_label_economics_value_sha256": integrity["r3_label_economics_contract"]["value_sha256"],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-protected-json", type=Path)
    parser.add_argument("--n-validation-folds", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite base ladder output: {args.output_dir}")
    policy = _json(args.required_protected_json) if args.required_protected_json else {}
    if not isinstance(policy, dict) or not set(policy).issubset({"required_features", "protected_features"}):
        raise ValueError("required/protected JSON permits only required_features/protected_features")
    config = BaseFeatureLadderConfig(
        n_validation_folds=args.n_validation_folds, min_train_rows=args.min_train_rows,
    )
    selector_ledger = pd.read_parquet(args.selector_dir / "selector_ledger.parquet")
    selector_integrity = _require_selector_integrity(args.selector_dir, selector_ledger)
    results: dict[str, Any] = {}
    for side in ("long", "short"):
        selection_dir = args.base_selection_dir / side
        source = load_completed_stage_i_base_selection(selection_dir, side=side)
        plan = materialize_nested_feature_challenge(
            source,
            required_features=policy.get("required_features", ()),
            protected_features=policy.get("protected_features", ()),
        )
        checkpoint_nested_feature_plan(plan, args.output_dir / side / "plan")
        manifest = _json(selection_dir / "manifest.json")
        if (
            manifest.get("schema") != "stage_i_base_feature_selection_v1"
            or manifest.get("status") != "complete"
            or str(manifest.get("side", "")).lower() != side
            or manifest.get("selector_sample_manifest_sha256") != _sha(args.selector_dir / "manifest.json")
            or manifest.get("selector_feature_contract_sha256") != _sha(args.selector_dir / "selector_feature_contract.json")
            or manifest.get("selector_artifact_integrity") != selector_integrity
            or not isinstance(manifest.get("best_params"), dict)
        ):
            raise ValueError(f"{side}: base selection manifest does not bind to selector input/feature contract")
        predictor = fixed_lgbm_base_predictor(
            FixedLGBMContract(base_params=manifest["best_params"])
        )
        results[side] = run_side_base_feature_ladder(
            load_side_input(args.selector_dir, side=side), plan,
            base_predictor=predictor, source_base_params=manifest["best_params"],
            source_base_manifest_sha256=_sha(selection_dir / "manifest.json"),
            output_dir=args.output_dir / side / "execution", config=config, resume=args.resume,
        )
    pooled = run_pooled_base_feature_ladder(
        long_dir=args.output_dir / "long" / "execution",
        short_dir=args.output_dir / "short" / "execution",
        output_dir=args.output_dir / "pooled_global", resume=args.resume,
    )
    print(json.dumps({
        "status": "complete", "output_dir": str(args.output_dir.resolve()),
        "sides": list(results),
        "sets": ["automatic_sparse", "top20", "top30", "top40", "top60", "full_input_control"],
        "pooled_global_ranking": "after_side_local_21d_common_bps_mapping",
        "freeze_disposition": "count_specific_base_HPO_and_refit_required",
        "pooled_request_sha256": pooled["request_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
