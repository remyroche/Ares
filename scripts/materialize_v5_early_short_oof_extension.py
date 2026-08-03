#!/usr/bin/env python3
"""Extend the sealed v5 March score ledger with legal early short OOF scores.

The frozen v5 short architecture starts on March 20 only because its second
stage selected new cut points after discarding the robust layer's warm-up.
All frozen score/support inputs are already strict OOF from March 1.  This
runner fits the already-selected B_peak_slope/tail=2 short architecture at
the robust layer's March 13 and March 19 cutoffs, uses those scores only
through March 19, and leaves every sealed v5 row from March 20 onward intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_bounded_robust_auxiliary_contribution_ablation as base
from scripts import run_bounded_short_conditional_payoff_ablation as short
from scripts.run_bounded_side_local_support_composition import strict_mae


V5 = ROOT / "data_perp/artifacts/short_winner_causal_recent_ev_mapping_20260730_v5"
WINNER = ROOT / "data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2"
FINAL_SEAL = ROOT / "data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v3_final_seal"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/v5_early_short_oof_extension_20260730_v1"
TIME = base.TIME
END = base.END
IDENTITY = tuple(base.ID)
CUTS = (
    pd.Timestamp("2025-03-13T00:00:00Z"),
    pd.Timestamp("2025-03-19T00:00:00Z"),
)
SEALED_START = pd.Timestamp("2025-03-20T00:00:00Z")


class ExtensionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def verified_manifest(root: Path, schema: str | None = None) -> dict[str, Any]:
    manifest = root / "manifest.json"
    seal = root / "manifest.sha256"
    if not manifest.is_file():
        raise ExtensionError(f"missing manifest: {manifest}")
    if seal.is_file() and seal.read_text().split()[0] != sha256(manifest):
        raise ExtensionError(f"manifest seal mismatch: {root}")
    payload = json.loads(manifest.read_text())
    if schema is not None and payload.get("schema") != schema:
        raise ExtensionError(f"unexpected manifest schema under {root}")
    return payload


def frozen_contract(winner: Mapping[str, Any]) -> tuple[list[str], float]:
    selected = winner.get("frozen_winner", {})
    if (
        selected.get("arm") != "B_peak_slope"
        or selected.get("key") != "B_peak_slope__tail_2"
        or float(selected.get("short_tail_weight", np.nan)) != 2.0
    ):
        raise ExtensionError("sealed short winner contract changed")
    arms = winner.get("contract", {}).get("arms", {})
    features = list(map(str, arms.get("B_peak_slope", ())))
    expected = [
        *base.F,
        "peak_contribution",
        "pred_future_slope_atr_per_hour__diagnostic",
    ]
    if features != expected:
        raise ExtensionError("frozen B_peak_slope feature order changed")
    return features, 2.0


def load_training_frame(args: argparse.Namespace) -> pd.DataFrame:
    load_args = SimpleNamespace(
        source=args.source,
        peak=args.peak,
        slope=args.slope,
    )
    frame = base.load(load_args)
    mae, status = strict_mae(args.mae)
    if mae is None:
        raise ExtensionError(f"strict MAE support unavailable: {status}")
    mae["__ts__"] = pd.to_datetime(mae["__ts__"], utc=True, errors="raise")
    frame = frame.merge(mae, on=list(IDENTITY), validate="one_to_one")
    development, _, _ = base.reconstruct(frame)
    development["peak_contribution"] = (
        development["pred_peak_mfe_12h_atr__p_hit"]
        * development["pred_peak_mfe_12h_atr__conditional_mean"]
    )
    development["adverse_severity"] = (
        development["pred_mae_before_meaningful_mfe_atr__p_hit"]
        * development["pred_mae_before_meaningful_mfe_atr__if_hit"]
        + (
            1.0
            - development["pred_mae_before_meaningful_mfe_atr__p_hit"]
        )
        * development["pred_mae_before_meaningful_mfe_atr__if_no_hit"]
    )
    return development


def early_short_rows(
    development: pd.DataFrame,
    *,
    features: Sequence[str],
    tail_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for cut, end in (
        (CUTS[0], CUTS[1]),
        (CUTS[1], SEALED_START),
    ):
        train = development.loc[
            development.side_name.eq("short")
            & development[TIME].lt(cut)
            & development[END].lt(cut)
        ].copy()
        valid = development.loc[
            development.side_name.eq("short")
            & development[TIME].ge(cut)
            & development[TIME].lt(end)
        ].copy()
        if len(train) < 10_000 or valid.empty:
            raise ExtensionError(
                f"insufficient early short support at {cut}: {len(train)}/{len(valid)}"
            )
        probability, favorable, adverse, score = short.fit_decomp(
            train,
            valid,
            features,
            tail_weight,
        )
        if not np.isfinite(score).all():
            raise ExtensionError("early short score contains non-finite values")
        valid["p_positive"] = probability
        valid["conditional_favorable_payoff"] = favorable
        valid["conditional_adverse_loss"] = adverse
        valid["raw_score"] = score
        valid["validation_start_utc"] = cut
        valid["validation_end_utc"] = end
        valid["fold_train_cutoff_utc"] = cut
        valid["training_label_resolved_max_utc"] = train[END].max()
        valid["score_available_utc"] = valid[TIME]
        valid["candidate_score_is_oof"] = True
        valid["upstream_scores_are_outer_oof"] = True
        valid["candidate_score_is_forward_oos"] = False
        valid["candidate_score_head"] = "short_conditional_payoff"
        valid["candidate_score_config"] = "B_peak_slope__tail_2"
        valid["ledger_stage"] = "march_early_short_oof_extension"
        if not valid["training_label_resolved_max_utc"].lt(
            valid["validation_start_utc"]
        ).all():
            raise ExtensionError("early short purge failed")
        pieces.append(valid)
        audits.append(
            {
                "validation_start_utc": cut,
                "validation_end_utc": end,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "train_label_end_max_utc": train[END].max(),
                "score_min": float(np.min(score)),
                "score_max": float(np.max(score)),
                "score_mean": float(np.mean(score)),
                "p_positive_mean": float(np.mean(probability)),
                "conditional_favorable_mean": float(np.mean(favorable)),
                "conditional_adverse_mean": float(np.mean(adverse)),
            }
        )
    result = pd.concat(pieces, ignore_index=True)
    if len(result) != 8_064 or result.duplicated(list(IDENTITY)).any():
        raise ExtensionError("expected exactly 8,064 unique March 13-19 short rows")
    return result, pd.DataFrame(audits)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    v5_manifest = verified_manifest(
        args.v5,
        "short_winner_causal_recent_ev_mapping_v5",
    )
    winner_manifest = verified_manifest(
        args.winner,
        "bounded_short_conditional_payoff_ablation_v2",
    )
    seal_manifest = verified_manifest(
        args.final_seal,
        "bounded_short_conditional_payoff_final_seal_v1",
    )
    if not bool(seal_manifest.get("runner_and_tests_bound")):
        raise ExtensionError("final short winner seal is not bound")
    features, tail_weight = frozen_contract(winner_manifest)
    sealed_path = args.v5 / "march_inner_chronological_oof_score_ledger.parquet"
    if (
        v5_manifest.get("outputs_sha256", {}).get(sealed_path.name)
        != sha256(sealed_path)
    ):
        raise ExtensionError("sealed v5 March ledger hash mismatch")
    sealed = pd.read_parquet(sealed_path)
    for column in (
        "__ts__",
        TIME,
        END,
        "validation_start_utc",
        "validation_end_utc",
        "fold_train_cutoff_utc",
        "training_label_resolved_max_utc",
        "score_available_utc",
    ):
        sealed[column] = pd.to_datetime(sealed[column], utc=True, errors="raise")
    development = load_training_frame(args)
    early, fold_audit = early_short_rows(
        development,
        features=features,
        tail_weight=tail_weight,
    )
    missing_columns = sorted(set(sealed.columns) - set(early.columns))
    if missing_columns:
        raise ExtensionError(f"early score source lacks sealed columns: {missing_columns}")
    early = early.loc[:, sealed.columns]
    overlap = early.merge(
        sealed.loc[:, list(IDENTITY)],
        on=list(IDENTITY),
        how="inner",
    )
    if not overlap.empty:
        raise ExtensionError("early extension overlaps sealed v5 identities")
    extended = (
        pd.concat([early, sealed], ignore_index=True)
        .sort_values([TIME, "candidate_id"], kind="stable")
        .reset_index(drop=True)
    )
    if extended.duplicated(list(IDENTITY)).any():
        raise ExtensionError("extended score ledger contains duplicate identities")
    counts = extended.groupby("side_name").size().to_dict()
    if counts != {"long": 20_736, "short": 20_736}:
        raise ExtensionError(f"unexpected extended side counts: {counts}")
    sealed_check = extended.merge(
        sealed.loc[:, [*IDENTITY, "raw_score"]].rename(
            columns={"raw_score": "sealed_raw_score"}
        ),
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    if len(sealed_check) != len(sealed) or not np.array_equal(
        sealed_check["raw_score"].to_numpy(),
        sealed_check["sealed_raw_score"].to_numpy(),
    ):
        raise ExtensionError("sealed v5 score bytes changed during extension")
    if not extended["candidate_score_is_oof"].astype(bool).all():
        raise ExtensionError("extended ledger contains a non-OOF score")
    if not extended["upstream_scores_are_outer_oof"].astype(bool).all():
        raise ExtensionError("extended ledger contains a non-OOF upstream score")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.",
            dir=args.output_dir.parent,
        )
    )
    try:
        extended.to_parquet(
            stage / "march_extended_oof_score_ledger.parquet",
            index=False,
            compression="zstd",
        )
        fold_audit.to_csv(stage / "early_short_fold_audit.csv", index=False)
        parity = {
            "sealed_rows": len(sealed),
            "extended_rows": len(extended),
            "added_short_rows": len(early),
            "sealed_identity_overlap_rows": len(sealed_check),
            "sealed_raw_score_bit_identical": True,
            "extended_side_rows": counts,
            "first_extended_decision_utc": extended[TIME].min(),
            "last_extended_decision_utc": extended[TIME].max(),
        }
        write_json(stage / "sealed_v5_parity.json", parity)
        outputs = {
            path.name: sha256(path)
            for path in stage.iterdir()
            if path.is_file()
        }
        manifest = {
            "schema": "v5_early_short_oof_extension_v1",
            "status": "STRICT_OOF_HISTORY_EXTENSION_NO_SELECTION_NO_PROMOTION",
            "promotion_eligible": False,
            "purpose": (
                "provide same-architecture March 13-19 short OOF mapping "
                "history without changing any sealed v5 score"
            ),
            "contract": {
                "architecture": "frozen B_peak_slope short conditional payoff",
                "tail_weight": tail_weight,
                "features": list(features),
                "new_cutoffs_utc": [str(value) for value in CUTS],
                "new_score_use": "[2025-03-13, 2025-03-20)",
                "sealed_score_use": "[2025-03-20, 2025-03-31)",
                "purge": "training execution_label_end_utc < validation_start_utc",
                "selection": "none; feature arm and tail weight inherited from sealed v5 winner",
                "mapping": "not performed by this artifact",
            },
            "counts": parity,
            "input_sha256": {
                "v5_manifest": sha256(args.v5 / "manifest.json"),
                "v5_march_ledger": sha256(sealed_path),
                "winner_manifest": sha256(args.winner / "manifest.json"),
                "final_seal_manifest": sha256(args.final_seal / "manifest.json"),
                "source": sha256(args.source),
                "peak": sha256(args.peak),
                "slope": sha256(args.slope),
                "mae": sha256(args.mae / "oof_predictions.parquet"),
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "This is research OOF mapping support, not new promotion evidence.",
                "January and February canonical residual/auxiliary gaps remain blocked.",
                "A downstream conversion learner must still generate its own pre-selection OOF scores before fitting a score-specific causal map.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            f"{sha256(stage / 'manifest.json')}  manifest.json\n"
        )
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--v5", type=Path, default=V5)
    command.add_argument("--winner", type=Path, default=WINNER)
    command.add_argument("--final-seal", type=Path, default=FINAL_SEAL)
    command.add_argument("--source", type=Path, default=base.SRC)
    command.add_argument("--peak", type=Path, default=base.PEAK)
    command.add_argument("--slope", type=Path, default=base.SLOPE)
    command.add_argument("--mae", type=Path, default=short.MAE)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parser().parse_args(argv))
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
