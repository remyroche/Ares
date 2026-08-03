#!/usr/bin/env python3
"""Compare nearby exact contracts against the selected TP6/SL4/H12 contract.

This is deliberately an outcome-contract audit, rather than an assertion that
the R3 robust-clear target itself has been regenerated for every neighbour.
Every comparison is on identical candidate IDs with a complete path for both
contracts.  Invalid/missing paths never enter ranks or oracle economics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SELECTED = "tp6_sl4_h12"


def _read_selected(path: Path) -> pd.DataFrame:
    x = pd.read_parquet(path, columns=["candidate_id", "side_name", "t2_tp6_sl4_event", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"])
    return x.rename(columns={"t2_tp6_sl4_event": "event", "t4_tp6_sl4_gross_bps": "gross_bps", "t4_tp6_sl4_net_bps": "net_bps"})


def _read_neighbor(path: Path) -> pd.DataFrame:
    x = pd.read_parquet(path, columns=["candidate_id", "side_name", "label_valid", "event", "gross_bps", "net_bps"])
    return x.loc[x.label_valid, ["candidate_id", "side_name", "event", "gross_bps", "net_bps"]]


def _top_ids(frame: pd.DataFrame, col: str, fraction: float = .10) -> set[str]:
    n = max(1, int(np.ceil(len(frame) * fraction)))
    return set(frame.nlargest(n, col).candidate_id)


def _oracle(frame: pd.DataFrame, score_col: str, gross_col: str, net_col: str, fraction: float) -> tuple[int, float, float]:
    n = max(1, int(np.ceil(len(frame) * fraction)))
    tail = frame.nlargest(n, score_col)
    return n, float(tail[gross_col].mean()), float(tail[net_col].mean())


def _metrics(joined: pd.DataFrame, alt: str, cohort: str) -> dict[str, object]:
    if not len(joined):
        return {"contract": alt, "cohort": cohort, "rows": 0}
    a = joined.net_bps_selected.rank(method="average")
    b = joined.net_bps_alternative.rank(method="average")
    selected_top = _top_ids(joined.rename(columns={"net_bps_selected": "score"}), "score")
    alternative_top = _top_ids(joined.rename(columns={"net_bps_alternative": "score"}), "score")
    row: dict[str, object] = {
        "contract": alt,
        "reference_contract": SELECTED,
        "cohort": cohort,
        "rows": int(len(joined)),
        "event_agreement": float((joined.event_selected == joined.event_alternative).mean()),
        "net_sign_agreement": float(((joined.net_bps_selected > 0) == (joined.net_bps_alternative > 0)).mean()),
        "net_spearman": float(a.corr(b)),
        "top10_net_jaccard": float(len(selected_top & alternative_top) / len(selected_top | alternative_top)),
        "selected_net_mean_bps": float(joined.net_bps_selected.mean()),
        "alternative_net_mean_bps": float(joined.net_bps_alternative.mean()),
    }
    for key, fraction in (("top1", .01), ("top5", .05), ("top10", .10)):
        for prefix, gross, net in (("selected", "gross_bps_selected", "net_bps_selected"), ("alternative", "gross_bps_alternative", "net_bps_alternative")):
            n, g, e = _oracle(joined, net, gross, net, fraction)
            row[f"{prefix}_{key}_rows"] = n
            row[f"{prefix}_{key}_oracle_gross_bps"] = g
            row[f"{prefix}_{key}_oracle_net_bps"] = e
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--alternative", action="append", nargs=2, metavar=("NAME", "ROOT"), required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for name, raw_root in a.alternative:
        root = Path(raw_root); pieces=[]
        for selected_part in sorted((a.selected / "parts").glob("*.parquet")):
            alt_part = root / "parts" / selected_part.name
            if not alt_part.exists():
                raise FileNotFoundError(alt_part)
            left = _read_selected(selected_part).dropna(subset=["net_bps"])
            right = _read_neighbor(alt_part)
            pieces.append(left.merge(right, on=["candidate_id", "side_name"], suffixes=("_selected", "_alternative"), validate="one_to_one"))
        joined = pd.concat(pieces, ignore_index=True)
        for side in ("all", "long", "short"):
            cohort = joined if side == "all" else joined.loc[joined.side_name == side]
            all_rows.append(_metrics(cohort, name, side))
    result = pd.DataFrame(all_rows)
    result.to_parquet(a.out / "target_contract_stability.parquet", index=False)
    (a.out / "run_manifest.json").write_text(json.dumps({
        "selected_contract": {"tp_atr": 6, "sl_atr": 4, "horizon_hours": 12, "cost_bps": 100},
        "alternatives": [{"name": n, "root": str(r)} for n, r in a.alternative],
        "comparison": "complete-path intersection; outcome net rank and event stability",
    }, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
