#!/usr/bin/env python3
"""Build the canonical 2023--24 M6 ledger and run no-imputation MDA.

This is a diagnostic, not a feature-selection result.  It uses only strict
same-side base OOF predictions, joins context exactly once by candidate_id,
and excludes (rather than fills) any row incomplete on the declared M6
contract.  The 2022 inverse-PI population is intentionally not joined: its
candidate/product/context schema is not demonstrably identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_m6_enriched_mda_20260809_v1"

# Explicitly omit market_state_transition_entropy_5d and breakout_retention_4h:
# their historical OOF coverage was 4.7%, so they cannot be zero-imputed.
CONTEXT = [
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h", "mkt_oi_chg_z_24h",
    "mkt_funding_dispersion", "cross_asset_corr_4h", "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score", "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market", "deleveraging_without_followthrough",
    "short_signal_recovery_conflict",
]
BASE = ["p_adverse", "p_weak", "p_clear", "base_raw"]
FEATURES = BASE + CONTEXT
ERAS = (
    ("2023-07_08", "2023-07-01", "2023-09-01", "oof23_f0"),
    ("2023-09_10", "2023-09-01", "2023-11-01", "oof23_f1"),
    ("2023-11_12", "2023-11-01", "2024-01-01", "oof23_f2"),
    ("2024-01_02", "2024-01-01", "2024-03-01", "oof23_f3"),
    ("2024-05_06", "2024-05-01", "2024-07-01", "ledger24"),
    ("2024-07_08", "2024-07-01", "2024-09-01", "ledger24"),
    ("2024-09_10", "2024-09-01", "2024-11-01", "ledger24"),
    ("2024-11", "2024-11-01", "2024-12-01", "ledger24"),
)
TOPS = (.01, .05, .10)
SEED = 20260809


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=.04, num_leaves=24,
        min_child_samples=400, colsample_bytree=.8, subsample=.8, reg_lambda=12.,
        random_state=SEED, n_jobs=1, verbosity=-1,
    )


def matrix(x: pd.DataFrame) -> np.ndarray:
    # The caller asserts finite complete contract rows.  No fill/zero imputation.
    a = x[FEATURES].to_numpy(np.float32)
    if not np.isfinite(a).all():
        raise ValueError("non-finite field reached M6 matrix")
    return a


def read_base() -> pd.DataFrame:
    all_parts = []
    for side in ("long", "short"):
        for fold in range(4):
            p = ROOT / f"data_perp/artifacts/tp6_r3_r5_{side}_baseoof_fold{fold}_20260802_v1/base_oof_predictions.parquet"
            x = pd.read_parquet(p)
            if len(x) == 0 or set(x.side_name.unique()) != {side} or x.candidate_id.duplicated().any():
                raise ValueError(f"strict same-side unique OOF failure: {p}")
            x = x.rename(columns={"prob_adverse":"p_adverse", "prob_weak":"p_weak", "prob_clear":"p_clear"})
            x["source"] = f"oof23_f{fold}"
            all_parts.append(x[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *BASE, "source"]])
    ledger_p = ROOT / "data_perp/artifacts/tp6_sl4_b10_bw4_base_oof_20260802_v1/base_oof_ledger.parquet"
    y = pd.read_parquet(ledger_p).rename(columns={
        "t4_tp6_sl4_net_bps":"net_bps", "base_expected_net_bps":"base_raw",
        "base_p_lower":"p_adverse", "base_p_timeout":"p_weak", "base_p_upper":"p_clear",
    })
    if y.candidate_id.duplicated().any():
        raise ValueError("2024 OOF ledger candidate ids are not unique")
    if not (pd.to_datetime(y.base_fit_resolved_before, utc=True) <= pd.to_datetime(y.__ts__, utc=True)).all():
        raise ValueError("2024 base output is not chronological OOF")
    y["gross_bps"] = y.net_bps + 100.
    y["source"] = "ledger24"
    all_parts.append(y[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *BASE, "source"]])
    out = pd.concat(all_parts, ignore_index=True)
    if out.candidate_id.duplicated().any():
        raise ValueError("base sources overlap candidate identities")
    out["__ts__"] = pd.to_datetime(out.__ts__, utc=True)
    if not np.allclose(out.gross_bps - out.net_bps, 100., atol=.01):
        raise ValueError("exact fixed 100-bps cost mismatch")
    return out


def read_context(ids: set[str], cache_dir: Path | None = None) -> pd.DataFrame:
    pieces = []
    sources = sorted(cache_dir.glob("*.parquet")) if cache_dir and cache_dir.exists() else sorted((PANEL / "parts").glob("*.parquet"))
    if not sources:
        raise ValueError("no context sources")
    for part in sources:
        x = pd.read_parquet(part, columns=["candidate_id", *CONTEXT])
        x = x[x.candidate_id.isin(ids)]
        if not x.empty:
            pieces.append(x)
    out = pd.concat(pieces, ignore_index=True)
    if out.candidate_id.duplicated().any() or set(out.candidate_id) != ids:
        raise ValueError("context candidate identity is missing or non-unique")
    return out


def era_frame(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for name, start, end, source in ERAS:
        x = data[(data.__ts__ >= pd.Timestamp(start, tz="UTC")) & (data.__ts__ < pd.Timestamp(end, tz="UTC")) & data.source.eq(source)].copy()
        if x.empty:
            raise ValueError(f"empty era: {name}")
        result[name] = x
    return result


def metrics(x: pd.DataFrame, score: np.ndarray, top: float) -> dict[str, float]:
    order = np.lexsort((x.candidate_id.to_numpy(), -score))
    take = x.iloc[order[:max(1, int(np.ceil(len(x) * top)))]]
    loss = np.maximum(-take.net_bps.to_numpy(float), 0.)
    return {"n":len(take), "net_bps":float(take.net_bps.mean()), "gross_bps":float(take.gross_bps.mean()),
            "fp_rate":float((take.net_bps <= 50).mean()), "false_positive_loss_bps":float(loss.mean()),
            "long_fraction":float(take.side_name.eq("long").mean())}


def score_pair(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    result = np.empty(len(test), dtype=np.float64)
    for side in ("long", "short"):
        tr, te = train[train.side_name.eq(side)], test[test.side_name.eq(side)]
        if tr.event.nunique() < 2:
            raise ValueError(f"one-class M6 train data for {side}")
        result[test.side_name.eq(side).to_numpy()] = model().fit(matrix(tr), tr.event).predict_proba(matrix(te))[:, 1]
    return result


def mda(test: pd.DataFrame, baseline: np.ndarray, train: pd.DataFrame, common: dict[str, object], rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    for top in TOPS:
        for metric, value in metrics(test, baseline, top).items():
            if metric != "n":
                rows.append({**common, "scope":"baseline", "feature":"__baseline__", "top_fraction":top, "metric":metric, "value":value, "delta":0.})
    # fixed fitted models; permute a test column then re-score.  This measures
    # out-of-sample reliance, not a retrained feature selection illusion.
    fitted = {}
    for side in ("long", "short"):
        tr = train[train.side_name.eq(side)]
        fitted[side] = model().fit(matrix(tr), tr.event)
    for feature in FEATURES:
        z = test.copy()
        z[feature] = rng.permutation(z[feature].to_numpy())
        perm = np.empty(len(z), dtype=np.float64)
        for side in ("long", "short"):
            mask = z.side_name.eq(side).to_numpy()
            perm[mask] = fitted[side].predict_proba(matrix(z.loc[mask]))[:, 1]
        for top in TOPS:
            base, changed = metrics(test, baseline, top), metrics(test, perm, top)
            for metric in ("net_bps", "gross_bps", "fp_rate", "false_positive_loss_bps", "long_fraction"):
                # positive mda is damage: lower economics or higher false-positive burden.
                delta = (base[metric] - changed[metric]) if metric in ("net_bps", "gross_bps") else (changed[metric] - base[metric])
                rows.append({**common, "scope":"permutation", "feature":feature, "top_fraction":top,
                             "metric":metric, "value":changed[metric], "delta":float(delta)})
    return rows


def classify_scope(train_name: str, test_name: str, era_names: list[str]) -> str:
    return "within_era_rolling" if era_names.index(test_name) == era_names.index(train_name) + 1 else "cross_era"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--materialise-only", action="store_true", help="write the validated ledger and stop before M6/MDA")
    ap.add_argument("--extract-context-shard", help="extract i/n candidate context shard, e.g. 0/12; supports bounded staged materialisation")
    ap.add_argument("--resume", action="store_true", help="continue a validated materialised ledger into M6/MDA")
    ap.add_argument("--mda-sample", type=int, default=0, help="deterministic max rows per MDA test cell; 0 means all")
    ap.add_argument("--mda-max-cells", type=int, default=0, help="bounded number of MDA transport cells; 0 means all")
    ap.add_argument("--mda-start-cell", type=int, default=0, help="zero-based transport-cell offset for resumable bounded MDA")
    ap.add_argument("--skip-expanding", action="store_true", help="skip full expanding-score rewrite when executing a bounded MDA cell")
    ap.add_argument("--aggregate-cells", action="store_true", help="aggregate previously completed bounded MDA cell artifacts")
    args = ap.parse_args()
    if args.aggregate_cells:
        parts = sorted(args.out.glob("mda_results_cells_*.parquet"))
        if not parts:
            raise FileNotFoundError("no bounded MDA cell artifacts")
        pm = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        pm.to_parquet(args.out / "mda_results.parquet", index=False)
        imp = pm[(pm["scope"].eq("permutation")) & (pm["metric"].isin(["net_bps", "false_positive_loss_bps"]))].groupby(
            ["relation", "feature", "top_fraction", "metric"], as_index=False
        ).agg(median_damage=("delta", "median"), mean_damage=("delta", "mean"),
              mad_damage=("delta", lambda x: float(np.median(np.abs(x-np.median(x))))), cells=("delta", "size"),
              positive_damage_share=("delta", lambda x: float((x>0).mean())))
        imp.to_parquet(args.out / "mda_stability_summary.parquet", index=False)
        print(json.dumps({"cells":len(parts), "mda_rows":len(pm), "out":str(args.out)}, indent=2))
        return
    if args.out.exists() and (args.out / "manifest.json").exists() and not (args.extract_context_shard or args.resume):
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    base = read_base()
    if args.extract_context_shard:
        try:
            i_text, n_text = args.extract_context_shard.split("/")
            i, n = int(i_text), int(n_text)
        except ValueError as exc:
            raise ValueError("--extract-context-shard must be i/n") from exc
        if not (0 <= i < n):
            raise ValueError("invalid context shard")
        target = args.out / "context_shards"
        target.mkdir(exist_ok=True)
        pieces = []
        for part in sorted((PANEL / "parts").glob("*.parquet"))[i::n]:
            x = pd.read_parquet(part, columns=["candidate_id", *CONTEXT])
            x = x[x.candidate_id.isin(set(base.candidate_id))]
            if not x.empty:
                pieces.append(x)
        x = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["candidate_id", *CONTEXT])
        if x.candidate_id.duplicated().any():
            raise ValueError("duplicate candidate in extracted context shard")
        x.to_parquet(target / f"context_{i:02d}_of_{n:02d}.parquet", index=False)
        print(json.dumps({"shard":f"{i}/{n}", "rows":len(x), "out":str(target)}, indent=2))
        return
    raw_rows = len(base)
    cache_dir = args.out / "context_shards"
    data = base.merge(read_context(set(base.candidate_id), cache_dir if cache_dir.exists() else None), on="candidate_id", how="inner", validate="one_to_one")
    if len(data) != raw_rows:
        raise ValueError("exact context join lost OOF candidates")
    data["event"] = data.net_bps.gt(50).astype("int8")
    finite = np.isfinite(data[FEATURES].to_numpy(float)).all(axis=1)
    coverage = 1 - data[FEATURES].replace([np.inf, -np.inf], np.nan).isna().mean()
    data["m6_contract_complete"] = finite
    # Retain every source row for coverage audit; models use only complete rows.
    complete = data.loc[finite].copy()
    eras_all, eras = era_frame(data), era_frame(complete)
    if len(complete) == 0:
        raise ValueError("no complete M6 rows")
    names = list(eras)
    data["era"] = pd.NA
    for name, x in eras_all.items():
        data.loc[data.candidate_id.isin(x.candidate_id), "era"] = name
    data["m6_probability_expanding"] = np.nan
    # Persist the validated substrate before expensive model diagnostics.  The
    # ledger deliberately includes incomplete rows rather than fabricating a
    # value for a missing causal field.
    data.sort_values("candidate_id").to_parquet(args.out / "canonical_enriched_ledger.parquet", index=False)
    validity = pd.DataFrame({"field":FEATURES, "coverage":coverage, "n_missing":data[FEATURES].isna().sum(), "finite_coverage": [float(np.isfinite(data[c].to_numpy(float)).mean()) for c in FEATURES]})
    validity.to_parquet(args.out / "ledger_feature_coverage.parquet", index=False)
    join = {"base_oof_rows":raw_rows, "joined_rows":len(data), "unique_candidate_ids":int(data.candidate_id.nunique()), "complete_contract_rows":int(finite.sum()), "incomplete_contract_rows":int((~finite).sum()), "exact_candidate_join":bool(len(data)==raw_rows==data.candidate_id.nunique())}
    (args.out / "join_audit.json").write_text(json.dumps(join, indent=2)+"\n")
    if args.materialise_only:
        (args.out / "manifest.json").write_text(json.dumps({"schema":"tp6_m6_enriched_ledger_mda_v1", "status":"MATERIALISED_PENDING_MDA", "no_imputation":True, "join_audit":join, "coverage":coverage.to_dict()}, indent=2)+"\n")
        print(json.dumps({"out":str(args.out), **join}, indent=2))
        return
    rolling_rows, mda_rows, prediction_rows = [], [], []
    rng = np.random.default_rng(SEED)
    # Canonical M6 OOF score: all prior compatible eras, later era only.
    if not args.skip_expanding:
        for i, test_name in enumerate(names[1:], 1):
            train = pd.concat([eras[n] for n in names[:i]], ignore_index=True)
            test = eras[test_name]
            score = score_pair(train, test)
            data.loc[data.candidate_id.isin(test.candidate_id), "m6_probability_expanding"] = score
            common = {"train_era":names[i-1], "train_eras":",".join(names[:i]), "test_era":test_name,
                      "train_rows":len(train), "test_rows":len(test), "relation":"expanding_prior"}
            for top in TOPS:
                for key, value in metrics(test, score, top).items():
                    rolling_rows.append({**common, "top_fraction":top, "metric":key, "value":value})
            prediction_rows.append(test[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "event"]].assign(m6_probability=score, **common))
    # MDA: single-era train -> every later era. Adjacent is the within-era
    # temporal transport view; non-adjacent is explicit cross-era transport.
    cell_index = 0
    for i, train_name in enumerate(names[:-1]):
        train = eras[train_name]
        for j, test_name in enumerate(names[i+1:], i+1):
            if cell_index < args.mda_start_cell:
                cell_index += 1
                continue
            if args.mda_max_cells and cell_index >= args.mda_start_cell + args.mda_max_cells:
                break
            test = eras[test_name]
            if args.mda_sample and len(test) > args.mda_sample:
                # Candidate-id hash ordering makes the diagnostic slice stable
                # across reruns and prevents any outcome-based sampling.
                key = pd.util.hash_pandas_object(test.candidate_id, index=False).to_numpy()
                test = test.iloc[np.argsort(key)[:args.mda_sample]].copy()
            score = score_pair(train, test)
            relation = classify_scope(train_name, test_name, names)
            common = {"train_era":train_name, "test_era":test_name, "train_rows":len(train), "test_rows":len(test), "relation":relation}
            mda_rows.extend(mda(test, score, train, common, rng))
            cell_index += 1
        if args.mda_max_cells and cell_index >= args.mda_start_cell + args.mda_max_cells:
            break
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_parquet(args.out / "expanding_m6_oof_predictions.parquet", index=False)
    if rolling_rows:
        pd.DataFrame(rolling_rows).to_parquet(args.out / "expanding_m6_metrics.parquet", index=False)
    pd.DataFrame(mda_rows).to_parquet(args.out / f"mda_results_cells_{args.mda_start_cell:02d}.parquet", index=False)
    pm = pd.DataFrame(mda_rows)
    importance = pm[(pm["scope"].eq("permutation")) & (pm["metric"].isin(["net_bps", "false_positive_loss_bps"]))].groupby(
        ["relation", "feature", "top_fraction", "metric"], as_index=False
    ).agg(
        median_damage=("delta", "median"), mean_damage=("delta", "mean"),
        mad_damage=("delta", lambda x: float(np.median(np.abs(x - np.median(x))))),
        cells=("delta", "size"), positive_damage_share=("delta", lambda x: float((x > 0).mean())),
    )
    importance.to_parquet(args.out / "mda_stability_summary.parquet", index=False)
    manifest = {"schema":"tp6_m6_enriched_ledger_mda_v1", "status":"COMPLETED_DIAGNOSTIC", "geometry":"TP6/SL4/H12", "cost_bps":100,
                "m6_target":"exact net > +50 bps", "base_lineage":"strict same-side chronological R3 OOF", "context":CONTEXT, "base_outputs":BASE,
                "no_imputation":True, "sparse_fields_excluded":["market_state_transition_entropy_5d", "breakout_retention_4h"],
                "historical_2022":"EXCLUDED_SCHEMA_NONPARITY", "eras":ERAS, "join_audit":join, "coverage":coverage.to_dict(),
                "inputs_sha256":{"script":sha(Path(__file__)), "ledger24":sha(ROOT / "data_perp/artifacts/tp6_sl4_b10_bw4_base_oof_20260802_v1/base_oof_ledger.parquet")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    lines = ["# Enriched 2023–24 TP6 M6 ledger and MDA", "", "## Contract", "", "- Candidate-keyed one-to-one join from strict same-side base OOF predictions to 14 high-coverage causal context fields.", "- No imputation: incomplete rows remain in the ledger with `m6_contract_complete=false` and are excluded from M6/MDA.", "- M6 outcome is exact H12 net > +50 bps; fixed cost is verified as 100 bps once.", "- 2022 is not pooled: schema parity is not established.", "", "## Result artifacts", "", "- `canonical_enriched_ledger.parquet`: all source rows, context, outcome and expanding-prior M6 score.", "- `mda_results.parquet`: train/test-cell permutation damage. Positive net damage means permutation lowered tail net; positive loss damage means it raised selected false-positive loss.", "- `mda_stability_summary.parquet`: median, MAD and positive-damage share by within-era versus cross-era cells.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"out":str(args.out), **join, "mda_rows":len(mda_rows)}, indent=2))


if __name__ == "__main__":
    main()
