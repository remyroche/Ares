#!/usr/bin/env python3
"""Attach canonical parent-policy labels to native BCF MC1 features.

Only target-free BCF score fields enter the feature derivation.  Policy labels
are joined afterwards and invalid paths remain explicit rather than becoming
zero-valued pseudo observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_bcf_mc1_mapper import derive_bcf_mc1_features


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_columns(
    path: Path, *, rank_fields_json: Path | None,
) -> tuple[list[str], list[str]]:
    """Load the immutable BCF head-field declaration before reading rows.

    The historical long scorer retains its fixed ten-head convention.  Short
    BCF is allowed only through a frozen declaration produced by the short
    consensus HPO; it must never discover fields from the scored population.
    """
    base = [
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "base_rank42", "upstream", "consensus_rank",
    ]
    if rank_fields_json is None:
        ranks = [
            f"residual_head__cap{cap}_{mode}__rank"
            for cap in (40, 60, 80, 100, 120) for mode in ("ordinary", "equal_month")
        ]
        return base + ranks, []
    payload = json.loads(rank_fields_json.read_text())
    ranks = [str(value) for value in payload.get("rank_fields", ())]
    ordinary = [str(value) for value in payload.get("ordinary_rank_fields", ())]
    if len(ranks) < 2 or len(ranks) != len(set(ranks)):
        raise ValueError("frozen BCF rank-fields contract needs two or more unique fields")
    if not ordinary or len(ordinary) != len(set(ordinary)) or not set(ordinary).issubset(ranks):
        raise ValueError("frozen BCF ordinary fields must be a non-empty unique subset")
    return base + ranks, ordinary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-scores", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument(
        "--side", choices=("long", "short"), default="long",
        help="Require both score and policy sources to be side-local.",
    )
    parser.add_argument(
        "--rank-fields-json", type=Path, default=None,
        help=("Frozen promoted-head declaration for a side-local BCF family. "
              "Omit only for the legacy ten-head long contract."),
    )
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()
    if args.out_path.exists():
        raise FileExistsError(args.out_path)
    score_columns, ordinary_fields = _score_columns(
        args.bcf_scores, rank_fields_json=args.rank_fields_json,
    )
    scores = pd.read_parquet(args.bcf_scores, columns=score_columns)
    scores["__decision_ts__"] = pd.to_datetime(scores["__decision_ts__"], utc=True)
    if scores["candidate_id"].duplicated().any():
        raise ValueError("BCF score source has duplicate candidate IDs")
    # Score producers standardise on a stable candidate identity rather than a
    # duplicated symbol column.  Reconstructing the symbol from that identity
    # keeps the ledger compatible with both old and current score schemas
    # without discovering anything from the scored population.
    symbol = scores["candidate_id"].astype(str).str.split("|", n=1).str[0]
    if (
        symbol.isna().any()
        or symbol.eq("").any()
        or not scores["candidate_id"].astype(str).str.contains("|", regex=False).all()
    ):
        raise ValueError("BCF candidate IDs must encode a non-empty symbol as the first pipe-delimited field")
    scores["__symbol__"] = symbol
    score_side = scores["side_name"].astype(str).str.strip().str.lower()
    if scores.empty or not score_side.eq(args.side).all():
        raise ValueError(
            "BCF score source must be side-local to --side; "
            f"expected={args.side}, observed={score_side.value_counts(dropna=False).to_dict()}"
        )
    scores["side_name"] = args.side
    native = derive_bcf_mc1_features(
        scores,
        rank_fields=None if args.rank_fields_json is None else [
            value for value in score_columns if value.endswith("__rank")
        ],
        ordinary_rank_fields=None if args.rank_fields_json is None else ordinary_fields,
    )
    policy_columns = [
        "candidate_id", "side_name", "policy_path_valid", "policy_net_bps",
        "policy_label_available_ts",
    ]
    policy = pd.read_parquet(args.policy_labels, columns=policy_columns)
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True)
    if policy["candidate_id"].duplicated().any():
        raise ValueError("parent-policy source has duplicate candidate IDs")
    policy_side = policy["side_name"].astype(str).str.strip().str.lower()
    if policy.empty or not policy_side.eq(args.side).all():
        raise ValueError(
            "parent-policy source must be side-local to --side; "
            f"expected={args.side}, observed={policy_side.value_counts(dropna=False).to_dict()}"
        )
    policy["side_name"] = args.side
    output = scores.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]].merge(
        native, on="candidate_id", how="inner", validate="one_to_one",
    ).merge(
        policy.drop(columns="side_name"), on="candidate_id", how="left",
        validate="one_to_one",
    )
    if len(output) != len(scores):
        raise AssertionError("BCF native feature attachment changed candidate identities")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.out_path, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_bcf_mc1_native_ledger_v1",
        "side": args.side,
        "rows": int(len(output)),
        "valid_policy_rows": int(output["policy_path_valid"].fillna(False).sum()),
        "bcf_scores": {"path": str(args.bcf_scores), "sha256": _sha(args.bcf_scores)},
        "policy_labels": {"path": str(args.policy_labels), "sha256": _sha(args.policy_labels)},
        "feature_contract": (
            "native_bcf_ten_head_agreement_v1" if args.rank_fields_json is None
            else "native_bcf_frozen_promoted_head_agreement_v2"
        ),
        "rank_fields_contract": (
            None if args.rank_fields_json is None else {
                "path": str(args.rank_fields_json), "sha256": _sha(args.rank_fields_json),
            }
        ),
    }
    args.out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
