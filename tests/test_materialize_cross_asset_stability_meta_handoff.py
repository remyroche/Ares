from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_cross_asset_stability_meta_handoff import materialize  # noqa: E402


def test_stability_handoff_uses_only_prior_month_cells(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    ledger = []
    for month in ("2026-05", "2026-06"):
        for idx in range(5):
            row = {
                "__ts__": f"{month}-{idx + 1:02d}T00:00:00Z",
                "__symbol__": f"SYM{idx}",
                "side_name": "short",
                "month": month,
                "source_semantic_family": "quiet_continuation",
                "score": float(idx),
                "selected_top10": True,
            }
            rows.append(row)
            ledger.append(
                {
                    "__ts__": row["__ts__"],
                    "__symbol__": row["__symbol__"],
                    "side_name": row["side_name"],
                    "frontier": "s52",
                    "exec_margin": 0.01,
                }
            )
    pd.DataFrame(rows).to_parquet(source / "train_meta_regime_handoff.parquet", index=False)
    pd.DataFrame(ledger).to_parquet(source / "s52_trailing_regime_scored_ledger.parquet", index=False)
    (source / "train_meta_regime_handoff_contract.json").write_text(json.dumps({"source": "unit"}))

    flip = tmp_path / "flip"
    flip.mkdir()
    cell_rows = []
    for month, value, ev in (("2026-05", 4.0, 0.010), ("2026-06", -9.0, -0.020)):
        for keep in (0.10, 0.20, 0.30):
            cell_rows.append(
                {
                    "month": month,
                    "keep_frac": keep,
                    "side_name": "short",
                    "source_semantic_family": "quiet_continuation",
                    "rows": 100,
                    "clean_rows": 40,
                    "positive_exec_rows": 45,
                    "effect_value_score": value,
                    "delta_ev_after_1pct": ev,
                    "delta_exec_margin": ev + 0.001,
                    "delta_clean_exec_precision": 0.20 if value > 0 else -0.20,
                    "delta_full_path_bad_mae": -0.10 if value > 0 else 0.20,
                    "delta_timeout": 0.0,
                    "delta_mfe_before_mae": 0.10 if value > 0 else -0.10,
                    "delta_mae_before_mfe": -0.10 if value > 0 else 0.10,
                    "delta_cell_oracle_overlap": 0.05 if value > 0 else -0.05,
                    "promoted_beneficial": value > 0,
                    "promoted_damaged": value < 0,
                }
            )
    pd.DataFrame(cell_rows).to_csv(flip / "promoted_cross_asset_month_cell_effects.csv", index=False)

    manifest = materialize(
        source_handoff_dir=source,
        baseline_smoke_dir=tmp_path / "unused_baseline",
        promoted_smoke_dir=tmp_path / "unused_promoted",
        flip_audit_dir=flip,
        out_dir=tmp_path / "out",
    )

    out = pd.read_parquet(manifest["handoff_path"])
    may = out[out["month"].astype(str).eq("2026-05")]
    june = out[out["month"].astype(str).eq("2026-06")]
    value_col = "xastab_k010_effect_value_score_prior_mean"
    ev_col = "xastab_k010_delta_ev_after_1pct_prior_mean"
    months_col = "xastab_k010_history_months"
    assert may[value_col].isna().all()
    assert may[months_col].fillna(0.0).eq(0.0).all()
    assert june[value_col].eq(4.0).all()
    assert june[ev_col].eq(0.010).all()
    assert june[months_col].eq(1.0).all()
    assert manifest["month_has_no_prior_history"]["2026-05"] is True
    assert manifest["month_has_no_prior_history"]["2026-06"] is False
    contract = json.loads((Path(manifest["contract_path"])).read_text())
    assert contract["cross_asset_stability_priors"]["leakage_contract"].startswith("For each row")
