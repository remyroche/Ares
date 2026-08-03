#!/usr/bin/env python3
"""Immutable identical-row audit of legacy-24h and native-12h base labels.

This diagnostic compares two *already frozen OOF* base score streams on their
common Feb--Apr 2025 candidate population.  It does not fit, calibrate, map,
or select a model.  In particular, every economic selection is a single
pooled-global book (candidate-ID ties), never a timestamp, side, or asset
quota.  Side figures are descriptive slices of that already selected book.

The native first-touch target and exact execution replay both resolve at
decision+12h.  The archived base target resolves at decision+24h.  Therefore
the legacy score's native-target IC is not a like-for-like target comparison
with exact-12h execution economics; this report makes that horizon mismatch
explicit instead of interpreting it as calibration evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/oof_predictions.parquet"
NATIVE_FEB = ROOT / "data_perp/artifacts/feb2025_native12h_base_oof_20260729_v1/oof_predictions.parquet"
NATIVE_MARAPR = ROOT / "data_perp/artifacts/febapr2025_native12h_partial_marapr_base_oof_20260729_v1/oof_predictions.parquet"
EXACT = ROOT / "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/labels.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_exact12h_legacy24h_base_label_parity_20260730_v1"

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
SCORES = ("legacy24h_oof_score", "native12h_oof_score")
TARGETS = (
    "legacy24h_native_soft_target",
    "native12h_native_soft_target",
    "exact12h_gross_return",
    "exact12h_cost_return",
    "exact12h_net_return",
)


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rank_ic(left: pd.Series, right: pd.Series) -> float:
    local = pd.DataFrame({"left": left, "right": right}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(local) < 3 or local.left.nunique() < 2 or local.right.nunique() < 2:
        return np.nan
    value = spearmanr(local.left, local.right).statistic
    return float(value) if np.isfinite(value) else np.nan


def stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    """Choose one deterministic pooled-global book; no quota is applied."""

    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    values = pd.to_numeric(frame[score], errors="raise").to_numpy(float)
    order = np.lexsort((frame.candidate_id.astype(str).to_numpy(), -values))
    return frame.iloc[order[:count]].copy()


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks identity fields: {missing}")
    if frame.duplicated(list(IDENTITY)).any() or frame.candidate_id.astype(str).duplicated().any():
        raise ValueError(f"{name} does not have unique candidate identities")


def _as_utc(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        if column not in frame:
            raise ValueError(f"{name} lacks {column}")
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")


def _load_legacy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _require_unique(frame, "legacy base OOF")
    _as_utc(frame, ("__ts__", "__decision_ts__", "base_label_resolution_utc"), "legacy base OOF")
    frame = frame.rename(columns={
        "__decision_ts__": "decision_utc",
        "base_label_resolution_utc": "legacy24h_label_available_utc",
        "__first_touch_target_soft__": "legacy24h_native_soft_target",
        "base_oof_score": "legacy24h_oof_score",
    })
    return frame


def _load_legacy(path: Path) -> pd.DataFrame:
    wanted = [*IDENTITY, "__decision_ts__", "base_label_resolution_utc", "__first_touch_target_soft__", "base_oof_score"]
    return _load_legacy_frame(pd.read_parquet(path, columns=wanted))


def _load_native_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _require_unique(frame, "native 12h base OOF")
    _as_utc(frame, ("__ts__", "__decision_ts__", "base_label_resolution_utc"), "native 12h base OOF")
    frame = frame.rename(columns={
        "__decision_ts__": "native_decision_utc",
        "base_label_resolution_utc": "native12h_label_available_utc",
        "target_12h": "native12h_native_soft_target",
        "base_oof_score": "native12h_oof_score",
    })
    return frame


def _load_native(feb_path: Path, marapr_path: Path) -> pd.DataFrame:
    wanted = [*IDENTITY, "__decision_ts__", "base_label_resolution_utc", "target_12h", "base_oof_score"]
    return _load_native_frame(pd.concat([pd.read_parquet(path, columns=wanted) for path in (feb_path, marapr_path)], ignore_index=True))


def _load_exact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _require_unique(frame, "exact 12h execution labels")
    _as_utc(frame, ("__ts__", "execution_decision_utc", "execution_label_end_utc", "execution_label_available_at"), "exact 12h execution labels")
    frame = frame.rename(columns={
        "execution_decision_utc": "exact12h_decision_utc",
        "execution_label_end_utc": "exact12h_label_end_utc",
        "execution_label_available_at": "exact12h_label_available_utc",
        "execution_gross_ev_12h": "exact12h_gross_return",
        "execution_cost_return": "exact12h_cost_return",
        "execution_net_ev_12h": "exact12h_net_return",
    })
    return frame


def _load_exact(path: Path) -> pd.DataFrame:
    wanted = [
        *IDENTITY, "execution_decision_utc", "execution_label_end_utc", "execution_label_available_at",
        "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h",
    ]
    return _load_exact_frame(pd.read_parquet(path, columns=wanted))


def build_identical_panel(legacy: pd.DataFrame, native: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    """Join and prove the matched population and all label-resolution contracts."""

    ids = {name: set(frame.candidate_id.astype(str)) for name, frame in {"legacy": legacy, "native": native, "exact": exact}.items()}
    if not (ids["legacy"] == ids["native"] == ids["exact"]):
        counts = {name: len(values) for name, values in ids.items()}
        raise ValueError(f"candidate IDs are not identical across frozen sources: {counts}")
    # The frozen base OOF artifacts render symbols with underscores while the
    # exact execution source retains exchange-style slashes. Candidate IDs,
    # side and timestamp are the authoritative shared identity; only this
    # reversible display normalization is allowed for the symbol assertion.
    def normalized_symbol(values: pd.Series) -> pd.Series:
        return values.astype(str).str.replace("_", "/", regex=False)

    source_index = legacy.set_index("candidate_id")
    for name, source in (("native", native), ("exact", exact)):
        other = source.set_index("candidate_id").loc[source_index.index]
        if not source_index.side_name.eq(other.side_name).all() or not source_index.__ts__.eq(other.__ts__).all():
            raise ValueError(f"{name} identity side/timestamp differs from legacy base OOF")
        if not normalized_symbol(source_index.__symbol__).eq(normalized_symbol(other.__symbol__)).all():
            raise ValueError(f"{name} identity symbol differs from legacy base OOF after canonical normalization")
    joined = legacy.merge(native.drop(columns=["side_name", "__symbol__", "__ts__"]), on="candidate_id", how="inner", validate="one_to_one")
    joined = joined.merge(exact.drop(columns=["side_name", "__symbol__", "__ts__"]), on="candidate_id", how="inner", validate="one_to_one")
    if len(joined) != len(legacy):
        raise ValueError("one-to-one join lost candidates")
    if not joined.decision_utc.eq(joined.native_decision_utc).all() or not joined.decision_utc.eq(joined.exact12h_decision_utc).all():
        raise ValueError("decision timestamps disagree across frozen sources")
    decision = joined.decision_utc
    if not joined.native12h_label_available_utc.eq(decision + pd.Timedelta(hours=12)).all():
        raise ValueError("native target is not decision+12h")
    if not joined.exact12h_label_available_utc.eq(decision + pd.Timedelta(hours=12)).all():
        raise ValueError("exact execution target is not decision+12h")
    if not joined.exact12h_label_end_utc.eq(joined.exact12h_label_available_utc).all():
        raise ValueError("exact execution label end/availability disagree")
    if not joined.legacy24h_label_available_utc.eq(decision + pd.Timedelta(hours=24)).all():
        raise ValueError("legacy target is not decision+24h")
    numeric = [*SCORES, *TARGETS[0:2], "exact12h_gross_return", "exact12h_cost_return", "exact12h_net_return"]
    if not np.isfinite(joined.loc[:, numeric].to_numpy(float)).all():
        raise ValueError("non-finite score or target")
    if not np.allclose(joined.exact12h_gross_return - joined.exact12h_cost_return, joined.exact12h_net_return, rtol=0.0, atol=1e-10):
        raise ValueError("exact economics violates gross - explicit cost = net")
    joined["candidate_month"] = joined.__ts__.dt.strftime("%Y-%m")
    joined["exact12h_opportunity"] = joined.exact12h_gross_return.gt(joined.exact12h_cost_return)
    return joined.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _scopes(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "all_months", "all", frame
    for month, local in frame.groupby("candidate_month", sort=True, observed=True):
        yield "monthly", str(month), local


def ic_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, month, local in _scopes(frame):
        for score in SCORES:
            for target in TARGETS:
                rows.append({"scope": scope, "candidate_month": month, "score": score, "target": target, "rows": int(len(local)), "rank_ic": rank_ic(local[score], local[target])})
            # A side slice is attribution only: neither score nor candidate population is reselected.
            for side, side_rows in local.groupby("side_name", sort=True, observed=True):
                for target in TARGETS:
                    rows.append({"scope": f"{scope}_side_attribution", "candidate_month": month, "side_name": str(side), "score": score, "target": target, "rows": int(len(side_rows)), "rank_ic": rank_ic(side_rows[score], side_rows[target])})
    return pd.DataFrame(rows)


def _economics(rows: pd.DataFrame, all_rows: pd.DataFrame, fraction: float) -> dict[str, Any]:
    gross = rows.exact12h_gross_return
    cost = rows.exact12h_cost_return
    net = rows.exact12h_net_return
    opportunity = rows.exact12h_opportunity
    denominator_opportunity = int(all_rows.exact12h_opportunity.sum())
    # Same-size exact-outcome oracles are a recall diagnostic, never an admission rule.
    oracle_gross = set(stable_top(all_rows, "exact12h_gross_return", fraction).candidate_id.astype(str))
    oracle_net = set(stable_top(all_rows, "exact12h_net_return", fraction).candidate_id.astype(str))
    selected_ids = set(rows.candidate_id.astype(str))
    favorable = rows.loc[opportunity]
    adverse = rows.loc[~opportunity]
    return {
        "candidate_rows": int(len(all_rows)), "selected_rows": int(len(rows)),
        "mean_gross_bps": float(gross.mean() * 1e4), "mean_cost_bps": float(cost.mean() * 1e4), "mean_net_bps": float(net.mean() * 1e4),
        "positive_net_rate": float(net.gt(0).mean()), "opportunity_precision": float(opportunity.mean()),
        "opportunity_recall": float(opportunity.sum() / denominator_opportunity) if denominator_opportunity else np.nan,
        "gross_oracle_same_k_recall": float(len(selected_ids.intersection(oracle_gross)) / len(oracle_gross)) if oracle_gross else np.nan,
        "net_oracle_same_k_recall": float(len(selected_ids.intersection(oracle_net)) / len(oracle_net)) if oracle_net else np.nan,
        "conditional_favorable_gross_bps": float(favorable.exact12h_gross_return.mean() * 1e4) if len(favorable) else np.nan,
        "conditional_favorable_net_bps": float(favorable.exact12h_net_return.mean() * 1e4) if len(favorable) else np.nan,
        "conditional_adverse_gross_bps": float(adverse.exact12h_gross_return.mean() * 1e4) if len(adverse) else np.nan,
        "conditional_adverse_net_bps": float(adverse.exact12h_net_return.mean() * 1e4) if len(adverse) else np.nan,
    }


def tail_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, month, local in _scopes(frame):
        for score in SCORES:
            for fraction in FRACTIONS:
                selected = stable_top(local, score, fraction)
                base = {"scope": scope, "candidate_month": month, "score": score, "fraction": fraction, "selection": "one_pooled_global_book"}
                rows.append({**base, "side_name": "all", **_economics(selected, local, fraction)})
                # These side rows only partition the parent global selection.  They never re-rank by side.
                for side, side_selected in selected.groupby("side_name", sort=True, observed=True):
                    side_population = local.loc[local.side_name.eq(side)]
                    rows.append({**base, "side_name": str(side), "selection": "side_attribution_of_pooled_global_book", **_economics(side_selected, side_population, fraction)})
    return pd.DataFrame(rows)


def ventile_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, month, local in _scopes(frame):
        for score in SCORES:
            work = local.copy()
            order = np.lexsort((work.candidate_id.astype(str).to_numpy(), pd.to_numeric(work[score], errors="raise").to_numpy(float)))
            ranks = np.empty(len(work), dtype=int)
            ranks[order] = np.arange(len(work))
            work["score_ventile"] = np.minimum((ranks * 20) // len(work), 19)
            denominator_opportunity = int(work.exact12h_opportunity.sum())
            for ventile, cell in work.groupby("score_ventile", sort=True, observed=True):
                for side, part in [("all", cell), *[(str(name), value) for name, value in cell.groupby("side_name", sort=True, observed=True)]]:
                    favorable = part.loc[part.exact12h_opportunity]
                    adverse = part.loc[~part.exact12h_opportunity]
                    rows.append({
                        "scope": scope, "candidate_month": month, "score": score, "score_ventile": int(ventile), "side_name": side,
                        "rows": int(len(part)), "score_mean": float(part[score].mean()), "mean_cost_bps": float(part.exact12h_cost_return.mean() * 1e4),
                        "mean_gross_bps": float(part.exact12h_gross_return.mean() * 1e4), "mean_net_bps": float(part.exact12h_net_return.mean() * 1e4),
                        "opportunity_rate": float(part.exact12h_opportunity.mean()),
                        "opportunity_recall": float(part.exact12h_opportunity.sum() / denominator_opportunity) if denominator_opportunity else np.nan,
                        "conditional_favorable_gross_bps": float(favorable.exact12h_gross_return.mean() * 1e4) if len(favorable) else np.nan,
                        "conditional_favorable_net_bps": float(favorable.exact12h_net_return.mean() * 1e4) if len(favorable) else np.nan,
                        "conditional_adverse_gross_bps": float(adverse.exact12h_gross_return.mean() * 1e4) if len(adverse) else np.nan,
                        "conditional_adverse_net_bps": float(adverse.exact12h_net_return.mean() * 1e4) if len(adverse) else np.nan,
                    })
    return pd.DataFrame(rows)


def run(*, legacy_path: Path, native_feb_path: Path, native_marapr_path: Path, exact_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    sources = {"legacy_base_oof": legacy_path, "native12h_feb_oof": native_feb_path, "native12h_marapr_oof": native_marapr_path, "exact12h_execution_labels": exact_path}
    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
    panel = build_identical_panel(_load_legacy(legacy_path), _load_native(native_feb_path, native_marapr_path), _load_exact(exact_path))
    output_dir.mkdir(parents=True, exist_ok=False)
    tables = {"paired_scores": panel, "rank_ic": ic_table(panel), "pooled_global_tails": tail_table(panel), "score_ventile_attribution": ventile_table(panel)}
    output: dict[str, dict[str, Any]] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False, compression="zstd")
        output[name] = {"path": str(path), "rows": int(len(table)), "sha256": sha256(path)}
    manifest = {
        "schema": "exact12h_vs_legacy24h_base_label_parity_v1",
        "status": "DIAGNOSTIC_ONLY_IDENTICAL_IDS_NO_PROMOTION",
        "promotion_eligible": False,
        "contracts": {
            "identity": list(IDENTITY), "eligible_candidate_ids": "identical set across legacy24h base OOF, native12h base OOF and exact12h execution labels; asserted before joining",
            "decision_time_safety": "legacy target resolves decision+24h; native first-touch and exact execution labels resolve decision+12h; decision timestamps agree on every candidate",
            "selection": "top 1/5/10/20% only, one pooled-global score book per all-month or monthly population with candidate-ID ties; never per timestamp, side or asset",
            "side_attribution": "side rows partition an already selected pooled-global book or an already formed pooled-global score ventile; no side-local re-ranking",
            "economics": "exact 12h gross - explicit realized cost = exact 12h net, asserted rowwise; costs are reported separately and only once",
            "opportunity": "exact gross > exact explicit cost; recall is selected opportunities divided by all opportunities in the matching all-month/month and, for side attribution, side population; same-K gross/net oracle recall is diagnostic only",
            "horizon_mismatch_caveat": "legacy base score was trained against a decision+24h native target. Its IC against legacy24h target is not same-target comparable with native12h target or exact12h gross/net. This audit assesses conversion, not calibration or a production winner.",
        },
        "source_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()},
        "population": {"rows": int(len(panel)), "candidate_id_sha256": hashlib.sha256("\n".join(panel.candidate_id.astype(str)).encode("utf-8")).hexdigest(), "months": sorted(panel.candidate_month.unique()), "side_rows": {str(name): int(count) for name, count in panel.side_name.value_counts().sort_index().items()}},
        "outputs": output,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "manifest.sha256").write_text(sha256(output_dir / "manifest.json") + "\n", encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--legacy", type=Path, default=LEGACY)
    result.add_argument("--native-feb", type=Path, default=NATIVE_FEB)
    result.add_argument("--native-marapr", type=Path, default=NATIVE_MARAPR)
    result.add_argument("--exact", type=Path, default=EXACT)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(legacy_path=args.legacy, native_feb_path=args.native_feb, native_marapr_path=args.native_marapr, exact_path=args.exact, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
