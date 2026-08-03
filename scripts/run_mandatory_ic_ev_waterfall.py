#!/usr/bin/env python3
"""Audit the identical-row base-IC to exact-execution-EV conversion.

This is a read-only attribution runner.  It consumes three already frozen
score/economic panels and deliberately keeps their populations separate:

* Feb--Apr canonical base OOF (base alpha only);
* Mar--Apr exact four-field base/residual/direct intersection;
* May--Jul exact four-field base/residual/direct intersection.

It never reads a mapped score, re-fits a model, changes a policy, or treats a
legacy 24h residual score as calibrated to the 12h execution label.  Every
top-k is pooled globally within month (with candidate-ID ties), not top-k per
timestamp or per side.  Side-local rows are attribution slices only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_PANEL = ROOT / (
    "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
)
DEFAULT_BASE_MANIFEST = DEFAULT_BASE_PANEL.with_name("manifest.json")
DEFAULT_MARAPR = ROOT / (
    "data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet"
)
DEFAULT_MARAPR_MANIFEST = DEFAULT_MARAPR.with_name("manifest.json")
DEFAULT_MAYJUL = ROOT / (
    "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/allscore_waterfall.parquet"
)
DEFAULT_MAYJUL_MANIFEST = DEFAULT_MAYJUL.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/mandatory_ic_ev_waterfall_20260730_v1"

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
TARGETS = {
    "native_24h_alpha": "__first_touch_target_soft__",
    "exact_12h_mfe": "execution_mfe_return_12h",
    "exact_12h_gross": "execution_gross_ev_12h",
    "exact_12h_cost": "execution_cost_return",
    "exact_12h_net": "execution_net_ev_12h",
}
EXIT_NAMES = ("trailing", "timeout", "full_stop", "adverse_exit")


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
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def rank_ic(left: pd.Series, right: pd.Series) -> float:
    local = pd.DataFrame({"left": left, "right": right}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(local) < 3 or local.left.nunique() < 2 or local.right.nunique() < 2:
        return np.nan
    value = spearmanr(local.left, local.right).statistic
    return float(value) if np.isfinite(value) else np.nan


def stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    """One global book, score descending and candidate-ID ascending ties."""

    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    values = pd.to_numeric(frame[score], errors="raise").to_numpy(float)
    order = np.lexsort((frame.candidate_id.astype(str).to_numpy(), -values))
    return frame.iloc[order[:count]].copy()


def _scopes(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "pooled_global", frame
    for side, local in frame.groupby("side_name", sort=True, observed=True):
        yield f"side_{side}", local


def _exit_class(frame: pd.DataFrame) -> pd.Series:
    raw = frame.get("execution_exit_class", frame.get("execution_exit_reason", pd.Series("unknown", index=frame.index)))
    value = raw.astype(str).str.lower()
    return value.where(value.isin(EXIT_NAMES), "adverse_exit")


def _cvar05(values: pd.Series) -> float:
    ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if ordered.empty:
        return np.nan
    return float(ordered.iloc[: max(1, int(math.ceil(len(ordered) * .05)))].mean())


def _capture_metrics(frame: pd.DataFrame) -> tuple[float, float, int]:
    mfe = pd.to_numeric(frame.execution_mfe_return_12h, errors="coerce")
    gross = pd.to_numeric(frame.execution_gross_ev_12h, errors="coerce")
    valid = mfe.gt(0.0) & np.isfinite(gross)
    if not valid.any():
        return np.nan, np.nan, 0
    row_capture = gross.loc[valid] / mfe.loc[valid]
    ratio = gross.loc[valid].mean() / mfe.loc[valid].mean()
    return float(ratio), float(row_capture.mean()), int(valid.sum())


def tail_metrics(frame: pd.DataFrame, family: str, score: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True, observed=True):
        for scope, local in _scopes(month_rows):
            for fraction in TOP_FRACTIONS:
                selected = stable_top(local, score, fraction)
                net = pd.to_numeric(selected.execution_net_ev_12h, errors="raise")
                gross = pd.to_numeric(selected.execution_gross_ev_12h, errors="raise")
                mfe = pd.to_numeric(selected.execution_mfe_return_12h, errors="raise")
                cost = pd.to_numeric(selected.execution_cost_return, errors="raise")
                capture_ratio, mean_row_capture, capture_rows = _capture_metrics(selected)
                exits = _exit_class(selected)
                row: dict[str, Any] = {
                    "source_family": family, "score": score, "candidate_month": str(month),
                    "scope": scope, "fraction": float(fraction), "candidate_rows": int(len(local)),
                    "selected_rows": int(len(selected)), "full_net_rank_ic": rank_ic(local[score], local.execution_net_ev_12h),
                    "tail_net_rank_ic": rank_ic(selected[score], net), "mean_mfe_bps": float(mfe.mean() * 1e4),
                    "mean_gross_bps": float(gross.mean() * 1e4), "mean_cost_bps": float(cost.mean() * 1e4),
                    "mean_net_bps": float(net.mean() * 1e4), "precision_positive_net": float(net.gt(0).mean()),
                    "loss_rate": float(net.lt(0).mean()), "cvar05_net_bps": float(_cvar05(net) * 1e4),
                    "mfe_to_gross_capture_ratio": capture_ratio, "mean_row_capture_ratio": mean_row_capture,
                    "capture_rows": capture_rows,
                }
                for name in EXIT_NAMES:
                    mask = exits.eq(name)
                    row[f"exit_{name}_rate"] = float(mask.mean())
                    row[f"exit_{name}_gross_bps"] = float(gross.loc[mask].mean() * 1e4) if mask.any() else np.nan
                    row[f"exit_{name}_net_bps"] = float(net.loc[mask].mean() * 1e4) if mask.any() else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def ic_metrics(frame: pd.DataFrame, family: str, score: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True, observed=True):
        for scope, local in _scopes(month_rows):
            for target_name, target in TARGETS.items():
                rows.append({"source_family": family, "score": score, "candidate_month": str(month),
                             "scope": scope, "selection": "full_sample", "fraction": np.nan,
                             "target": target_name, "rows": int(len(local)), "rank_ic": rank_ic(local[score], local[target])})
            for fraction in TOP_FRACTIONS:
                selected = stable_top(local, score, fraction)
                for target_name, target in TARGETS.items():
                    rows.append({"source_family": family, "score": score, "candidate_month": str(month),
                                 "scope": scope, "selection": "score_global_tail", "fraction": float(fraction),
                                 "target": target_name, "rows": int(len(selected)), "rank_ic": rank_ic(selected[score], selected[target])})
    return pd.DataFrame(rows)


def response_cells(frame: pd.DataFrame, family: str, score: str, bins: int = 20) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True, observed=True):
        for scope, local in _scopes(month_rows):
            work = local.copy()
            order = np.lexsort((work.candidate_id.astype(str).to_numpy(), pd.to_numeric(work[score], errors="raise").to_numpy(float)))
            ranks = np.empty(len(work), dtype=int); ranks[order] = np.arange(len(work))
            work["score_ventile"] = np.minimum((ranks * bins) // len(work), bins - 1)
            exits = _exit_class(work)
            for ventile, cell in work.groupby("score_ventile", sort=True, observed=True):
                net = pd.to_numeric(cell.execution_net_ev_12h, errors="raise")
                gross = pd.to_numeric(cell.execution_gross_ev_12h, errors="raise")
                cost = pd.to_numeric(cell.execution_cost_return, errors="raise")
                mfe = pd.to_numeric(cell.execution_mfe_return_12h, errors="raise")
                mae = pd.to_numeric(cell.execution_mae_return_12h, errors="raise")
                capture_ratio, mean_row_capture, capture_rows = _capture_metrics(cell)
                row: dict[str, Any] = {
                    "source_family": family, "score": score, "candidate_month": str(month), "scope": scope,
                    "score_ventile": int(ventile), "rows": int(len(cell)), "score_mean": float(cell[score].mean()),
                    "net_response_intercept_bps": float(net.mean() * 1e4), "gross_response_intercept_bps": float(gross.mean() * 1e4),
                    "mfe_mean_bps": float(mfe.mean() * 1e4), "mae_mean_bps": float(mae.mean() * 1e4),
                    "cost_bps": float(cost.mean() * 1e4), "positive_net_rate": float(net.gt(0).mean()),
                    "conditional_positive_gross_bps": float(gross.loc[net.gt(0)].mean() * 1e4) if net.gt(0).any() else np.nan,
                    "mfe_to_gross_capture_ratio": capture_ratio, "mean_row_capture_ratio": mean_row_capture,
                    "capture_rows": capture_rows,
                }
                local_exits = exits.loc[cell.index]
                for name in EXIT_NAMES:
                    mask = local_exits.eq(name)
                    row[f"exit_{name}_rate"] = float(mask.mean())
                    row[f"exit_{name}_gross_bps"] = float(gross.loc[mask].mean() * 1e4) if mask.any() else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _state_rows(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    net = pd.to_numeric(work.execution_net_ev_12h, errors="raise")
    exits = _exit_class(work)
    work["state"] = np.where(net.gt(0), "positive_net", np.where(exits.eq("timeout"), "timeout_nonpositive", np.where(exits.eq("adverse_exit"), "adverse_nonpositive", np.where(exits.eq("full_stop"), "full_stop_nonpositive", "trailing_nonpositive"))))
    work["day"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise").dt.floor("D")
    return work


def _state_decomposition(source: pd.DataFrame, target: pd.DataFrame) -> list[dict[str, float | str]]:
    """Exact symmetric probability/value waterfall on mutually exclusive states."""

    states = ("positive_net", "timeout_nonpositive", "adverse_nonpositive", "full_stop_nonpositive", "trailing_nonpositive")
    result: list[dict[str, float | str]] = []
    for state in states:
        a, b = source.loc[source.state.eq(state)], target.loc[target.state.eq(state)]
        pa, pb = len(a) / len(source), len(b) / len(target)
        va = float(a.execution_gross_ev_12h.mean() * 1e4) if len(a) else 0.0
        vb = float(b.execution_gross_ev_12h.mean() * 1e4) if len(b) else 0.0
        result.extend((
            {"component": f"{state}_prevalence", "contribution_bps": .5 * (pb - pa) * (va + vb)},
            {"component": f"{state}_gross_payoff", "contribution_bps": .5 * (pa + pb) * (vb - va)},
        ))
    return result


def _rank_cell(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    work = frame.copy()
    work["score_ventile"] = -1
    for _, local in work.groupby(["candidate_month", "side_name"], sort=False, observed=True):
        order = np.lexsort((local.candidate_id.astype(str).to_numpy(), pd.to_numeric(local[score], errors="raise").to_numpy(float)))
        ranks = np.empty(len(local), dtype=int); ranks[order] = np.arange(len(local))
        work.loc[local.index, "score_ventile"] = np.minimum((ranks * 20) // len(local), 19)
    if work.score_ventile.lt(0).any():
        raise AssertionError("missing deterministic rank cells")
    work["rank_cell"] = work.side_name.astype(str) + "|v" + work.score_ventile.astype(str)
    return work


def rank_level_waterfall(frame: pd.DataFrame, family: str, score: str, fraction: float = .10) -> pd.DataFrame:
    """Adjacent-month global-book change decomposition with exact reconciliation."""

    selected = {
        str(month): _state_rows(stable_top(local, score, fraction))
        for month, local in frame.groupby("candidate_month", sort=True, observed=True)
    }
    rows: list[dict[str, Any]] = []
    for first, second in zip(sorted(selected)[:-1], sorted(selected)[1:]):
        a, b = _rank_cell(selected[first], score), _rank_cell(selected[second], score)
        ac, bc = a.groupby("rank_cell", observed=True), b.groupby("rank_cell", observed=True)
        cells = sorted(set(ac.groups).intersection(bc.groups))
        if not cells:
            continue
        # Side x rank-ventile cells are an attribution composition, not an admission quota.
        wa = np.asarray([len(ac.get_group(key)) / len(a) for key in cells])
        wb = np.asarray([len(bc.get_group(key)) / len(b) for key in cells])
        gross_a = np.asarray([ac.get_group(key).execution_gross_ev_12h.mean() * 1e4 for key in cells])
        cost_a = np.asarray([ac.get_group(key).execution_cost_return.mean() * 1e4 for key in cells])
        gross_b = np.asarray([bc.get_group(key).execution_gross_ev_12h.mean() * 1e4 for key in cells])
        cost_b = np.asarray([bc.get_group(key).execution_cost_return.mean() * 1e4 for key in cells])
        components: list[dict[str, Any]] = [
            {"component": "rank_cell_composition_gross", "contribution_bps": float(np.dot(wb - wa, gross_a))},
            {"component": "rank_cell_composition_cost", "contribution_bps": float(-np.dot(wb - wa, cost_a))},
            {"component": "within_cell_cost", "contribution_bps": float(-np.dot(wb, cost_b - cost_a))},
        ]
        for weight, key in zip(wb, cells):
            for entry in _state_decomposition(ac.get_group(key), bc.get_group(key)):
                entry = dict(entry)
                entry["contribution_bps"] = float(weight * float(entry["contribution_bps"]))
                components.append(entry)
        aggregate = pd.DataFrame(components).groupby("component", as_index=False).contribution_bps.sum()
        actual = float(b.execution_net_ev_12h.mean() * 1e4 - a.execution_net_ev_12h.mean() * 1e4)
        for entry in aggregate.itertuples(index=False):
            rows.append({"source_family": family, "score": score, "fraction": fraction, "from_month": first, "to_month": second,
                         "component": entry.component, "contribution_bps": float(entry.contribution_bps), "actual_net_delta_bps": actual,
                         "reconciliation_error_bps": float(aggregate.contribution_bps.sum() - actual), "common_rank_cell_mass_source": float(wa.sum()), "common_rank_cell_mass_target": float(wb.sum())})
    return pd.DataFrame(rows)


def bootstrap_tail_ci(frame: pd.DataFrame, family: str, score: str, *, reps: int, seed: int, fraction: float = .10) -> pd.DataFrame:
    """Day-block CIs hold the frozen monthly global selected identities fixed."""

    rng = np.random.default_rng(seed)
    selected = {str(month): _state_rows(stable_top(local, score, fraction)) for month, local in frame.groupby("candidate_month", sort=True)}
    rows: list[dict[str, Any]] = []
    samples: dict[str, np.ndarray] = {}
    for month, local in selected.items():
        daily = local.groupby("day", observed=True).agg(net_sum=("execution_net_ev_12h", "sum"), rows=("candidate_id", "size"))
        values = daily.to_numpy(float)
        draws = np.empty(reps, dtype=float)
        for index in range(reps):
            chosen = rng.integers(0, len(values), len(values))
            draws[index] = values[chosen, 0].sum() / values[chosen, 1].sum() * 1e4
        samples[month] = draws
        rows.append({"source_family": family, "score": score, "fraction": fraction, "kind": "monthly_tail_net", "from_month": None, "to_month": month,
                     "estimate_bps": float(local.execution_net_ev_12h.mean() * 1e4), "ci95_low_bps": float(np.quantile(draws, .025)), "ci95_high_bps": float(np.quantile(draws, .975)), "bootstrap_reps": reps, "day_blocks": int(len(values))})
    for first, second in zip(sorted(samples)[:-1], sorted(samples)[1:]):
        draws = samples[second] - samples[first]
        observed = float(selected[second].execution_net_ev_12h.mean() * 1e4 - selected[first].execution_net_ev_12h.mean() * 1e4)
        rows.append({"source_family": family, "score": score, "fraction": fraction, "kind": "adjacent_month_delta", "from_month": first, "to_month": second,
                     "estimate_bps": observed, "ci95_low_bps": float(np.quantile(draws, .025)), "ci95_high_bps": float(np.quantile(draws, .975)), "bootstrap_reps": reps, "day_blocks": np.nan})
    return pd.DataFrame(rows)


def _require_economics(frame: pd.DataFrame, name: str) -> None:
    required = {*IDENTITY, "candidate_month", *TARGETS.values(), "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_mfe_return_12h", "execution_mae_return_12h"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks required fields: {missing}")
    if frame.duplicated(list(IDENTITY)).any() or frame.candidate_id.astype(str).duplicated().any():
        raise ValueError(f"{name} lacks unique four-field/candidate identities")
    gross = pd.to_numeric(frame.execution_gross_ev_12h, errors="raise").to_numpy(float)
    cost = pd.to_numeric(frame.execution_cost_return, errors="raise").to_numpy(float)
    net = pd.to_numeric(frame.execution_net_ev_12h, errors="raise").to_numpy(float)
    if not np.allclose(gross - cost, net, rtol=0., atol=1e-7):
        raise ValueError(f"{name} violates gross - explicit cost = net")


def prepare_inputs(base: pd.DataFrame, marapr: pd.DataFrame, mayjul: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Select only raw score streams; mapped score fields are never copied."""

    base = base.loc[base.candidate_month.astype(str).isin(("2025-02", "2025-03", "2025-04"))].copy()
    base = base.rename(columns={"base_oof_score": "score_base_alpha"})
    base["source_family"] = "febapr2025_canonical_base_oof"
    marapr = marapr.copy(); marapr["source_family"] = "marapr2025_exact_base_residual_direct_intersection"
    mayjul = mayjul.copy(); mayjul["source_family"] = "mayjul2026_exact_base_residual_direct_intersection"
    result = {"febapr2025_canonical_base_oof": base, "marapr2025_exact_base_residual_direct_intersection": marapr, "mayjul2026_exact_base_residual_direct_intersection": mayjul}
    for name, frame in result.items():
        _require_economics(frame, name)
        if any("mapped" in value.lower() for value in frame.columns if value.startswith("score_")):
            raise ValueError("mapped score fields are forbidden")
    return result


def declared_scores(frame: pd.DataFrame) -> list[str]:
    values = sorted(column for column in frame if column.startswith("score_") and np.issubdtype(frame[column].dtype, np.number))
    if not values:
        raise ValueError("no raw numeric score streams")
    return values


def run(*, base_panel: Path, base_manifest: Path, marapr: Path, marapr_manifest: Path, mayjul: Path, mayjul_manifest: Path, output_dir: Path, bootstrap_reps: int = 300) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    inputs = {"base_panel": (base_panel, base_manifest), "marapr": (marapr, marapr_manifest), "mayjul": (mayjul, mayjul_manifest)}
    for name, (path, manifest) in inputs.items():
        if not path.exists() or not manifest.exists():
            raise FileNotFoundError(f"missing frozen input for {name}")
    frames = prepare_inputs(pd.read_parquet(base_panel), pd.read_parquet(marapr), pd.read_parquet(mayjul))
    output_parts: dict[str, list[pd.DataFrame]] = {key: [] for key in ("score_registry", "ic", "tails", "response_ventiles", "rank_level_waterfall", "day_block_ci")}
    for family, frame in frames.items():
        scores = declared_scores(frame)
        for score in scores:
            output_parts["score_registry"].append(pd.DataFrame([{"source_family": family, "score": score, "rows": len(frame), "score_role": "base_alpha" if score == "score_base_alpha" else ("residual_component" if score == "score_residual_delta_ev" else ("residual_or_base_expected_ev" if "expected_ev" in score else "raw_direct_or_transfer_score"))}]))
            output_parts["ic"].append(ic_metrics(frame, family, score))
            output_parts["tails"].append(tail_metrics(frame, family, score))
            output_parts["response_ventiles"].append(response_cells(frame, family, score))
            output_parts["rank_level_waterfall"].append(rank_level_waterfall(frame, family, score))
            seed = int.from_bytes(
                hashlib.sha256(f"{family}|{score}".encode("utf-8")).digest()[:4],
                "little",
            )
            output_parts["day_block_ci"].append(
                bootstrap_tail_ci(
                    frame, family, score, reps=bootstrap_reps, seed=seed
                )
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, dict[str, Any]] = {}
    for name, pieces in output_parts.items():
        path = output_dir / f"{name}.parquet"
        frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        frame.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {"path": str(path), "rows": int(len(frame)), "sha256": sha256(path)}
    report = {
        "schema": "mandatory_identical_row_ic_ev_waterfall_v1",
        "status": "DIAGNOSTIC_ONLY_NO_MAPPING_NO_PROMOTION",
        "contracts": {
            "identity": list(IDENTITY), "selection": "monthly pooled-global top 1/5/10/20; score descending, candidate-ID ascending ties; never per timestamp top-k", "score_streams": "base alpha, residual components/expected EV, and direct scores remain separately named; score availability differs by frozen source family", "economics": "gross - explicit cost = exact 12h net; no additional cost is applied", "legacy_target": "native alpha target is a legacy 24h target; exact execution targets are 12h; this diagnostic does not claim same-target calibration", "bootstrap": "day-block bootstrap resamples frozen selected daily blocks, and so quantifies economics uncertainty without reselecting or tuning on labels", "rank_level": "rank-cell composition uses side x raw-score ventile as attribution cells, never admission quotas", "mapped_scores": "forbidden", "promotion": "forbidden",
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path), "manifest_path": str(manifest), "manifest_sha256": sha256(manifest)} for name, (path, manifest) in inputs.items()},
        "families": {family: {"rows": len(frame), "months": sorted(frame.candidate_month.astype(str).unique()), "scores": declared_scores(frame)} for family, frame in frames.items()},
        "outputs": outputs,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "promotion_eligible": False,
    }
    manifest = output_dir / "manifest.json"; write_json(manifest, report)
    (output_dir / "manifest.sha256").write_text(sha256(manifest) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-panel", type=Path, default=DEFAULT_BASE_PANEL)
    result.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    result.add_argument("--marapr", type=Path, default=DEFAULT_MARAPR)
    result.add_argument("--marapr-manifest", type=Path, default=DEFAULT_MARAPR_MANIFEST)
    result.add_argument("--mayjul", type=Path, default=DEFAULT_MAYJUL)
    result.add_argument("--mayjul-manifest", type=Path, default=DEFAULT_MAYJUL_MANIFEST)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--bootstrap-reps", type=int, default=300)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(base_panel=args.base_panel, base_manifest=args.base_manifest, marapr=args.marapr, marapr_manifest=args.marapr_manifest, mayjul=args.mayjul, mayjul_manifest=args.mayjul_manifest, output_dir=args.output_dir, bootstrap_reps=args.bootstrap_reps)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
