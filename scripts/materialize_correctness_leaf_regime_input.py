#!/usr/bin/env python3
"""Materialise the common all-meta-feature input for correctness leaf regimes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.transport_supervised_archetypes import configured_available_meta_features  # noqa: E402


LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet"
TRUST = ROOT / "data_perp/artifacts/tp6_ordinal_residual_meta_input_20260803_v2/prequential_trust.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/correctness_leaf_regime_input_20260803_v1"
REGIME = (
    "soft_regime_prior_residual_bps", "soft_regime_prior_residual_scale_bps",
    "regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition",
    "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
)


def run(output: Path, *, proxy_modulus: int = 5) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if proxy_modulus < 1:
        raise ValueError("proxy_modulus must be positive")
    # The all-configured feature contract is intentionally wide.  DuckDB
    # streams the deterministic row proxy directly to Parquet; disable order
    # preservation so the join can spill rather than retain unnecessary state.
    con = duckdb.connect(config={"threads": "2", "memory_limit": "2GB", "temp_directory": "/tmp", "preserve_insertion_order": "false"})
    try:
        panel_columns = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PANEL.as_posix()}') LIMIT 1").fetchdf().column_name.tolist()
        meta = configured_available_meta_features(CFG, panel_columns)
        # The selector sees every configured and physically present meta field;
        # the deterministic row proxy controls compute, never the feature
        # universe.  Existing target/path fields are rejected by the helper.
        ledger_fields = ("candidate_id", "__ts__", "side_name", "era", "gross_bps", "net_bps", "p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps", *REGIME)
        select_l = ", ".join(f'l."{name}"' for name in ledger_fields)
        select_p = ", ".join(f'p."{name}"' for name in meta)
        destination = output.with_name(output.name + ".partial")
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "input.parquet"
        sql = f'''COPY (
          SELECT hash(l.candidate_id) AS candidate_key, {select_l}, {select_p},
                 t.trust_relationship_break_mean_abs,t.trust_relationship_break_max_abs,
                 t.trust_score_ood_mean_abs_z,t.trust_score_ood_max_abs_z,
                 t.trust_active_failure_probability,t.trust_active_failure_support_weeks
          FROM read_parquet('{LEDGER.as_posix()}') l
          JOIN read_parquet('{PANEL.as_posix()}') p USING(candidate_id)
          LEFT JOIN read_parquet('{TRUST.as_posix()}') t ON hash(l.candidate_id)=t.candidate_key
          WHERE l.shared_regime_contract_complete
            AND l.prequential_base_expected_net_bps IS NOT NULL
            AND abs(hash(l.candidate_id)) % {int(proxy_modulus)} = 0
        ) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)'''
        if not path.exists():
            con.execute(sql)
        coverage = con.execute(f"SELECT * FROM (DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}'))").fetchdf()
        feature_columns = [name for name in coverage.column_name.astype(str) if name not in {"candidate_id", "candidate_key", "__ts__", "side_name", "era", "gross_bps", "net_bps"}]
        coverage_rows = []
        for start in range(0, len(feature_columns), 80):
            fields = feature_columns[start:start + 80]
            terms = ", ".join(
                f"avg(CASE WHEN \"{field}\" IS NOT NULL THEN 1.0 ELSE 0.0 END) AS \"{field}__coverage\", approx_count_distinct(\"{field}\") AS \"{field}__unique\""
                for field in fields
            )
            summary = con.execute(f"SELECT {terms} FROM read_parquet('{path.as_posix()}')").fetchdf().iloc[0]
            coverage_rows.extend({"feature": field, "coverage": float(summary[f"{field}__coverage"]), "approx_unique": int(summary[f"{field}__unique"]), "usable_90pct_nonconstant": bool(summary[f"{field}__coverage"] >= .90 and summary[f"{field}__unique"] > 1)} for field in fields)
        data = pd.read_parquet(path, columns=["candidate_id", "__ts__", "side_name", "net_bps", "prequential_base_expected_net_bps"])
        data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
        timestamps = pd.Index(data["__ts__"].drop_duplicates().sort_values())
        history = []
        for fold in range(1, 5):
            start_pos = int(np.floor(len(timestamps) * fold / 5))
            start_ts = timestamps[start_pos]
            train = data.loc[data["__ts__"].lt(start_ts - pd.Timedelta(hours=13))]
            history.append({"outer_fold": fold, "evaluation_start": start_ts, "resolved_training_rows": len(train), "training_start": train["__ts__"].min(), "training_end": train["__ts__"].max(), "training_days": float((train["__ts__"].max() - train["__ts__"].min()).total_seconds() / 86400.) if len(train) else 0.})
        usable = int(sum(row["usable_90pct_nonconstant"] for row in coverage_rows))
        manifest = {"schema": "correctness_leaf_regime_input_v1", "status": "COMPLETED", "proxy": f"deterministic candidate_id hash 1/{proxy_modulus}", "feature_universe": "all configured, physically present meta fields", "configured_meta_features": len(meta), "eligible_90pct_nonconstant_features": usable, "rows": len(data), "time_start": str(data["__ts__"].min()), "time_end": str(data["__ts__"].max()), "label": "exact net bps minus prequential R3 TP6/SL4 expected net bps", "entry_and_label_availability": "signal close +1h entry + H12, assigned by downstream target builder"}
        coverage.to_parquet(destination / "schema.parquet", index=False)
        pd.DataFrame(coverage_rows).to_parquet(destination / "feature_availability.parquet", index=False)
        pd.DataFrame(history).to_parquet(destination / "chronological_history_audit.parquet", index=False)
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        destination.replace(output)
        return output
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--proxy-modulus", type=int, default=5)
    args = parser.parse_args()
    print(run(args.out, proxy_modulus=args.proxy_modulus))
