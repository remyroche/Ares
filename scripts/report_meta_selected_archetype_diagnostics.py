#!/usr/bin/env python3
"""Report side/archetype diagnostics for exported train_meta selected rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _num(series: pd.Series | Any, default: float = np.nan) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.Series(default)


def _safe_mean(series: pd.Series | Any) -> float:
    values = _num(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if len(values) else float("nan")


def _safe_quantile(series: pd.Series | Any, q: float) -> float:
    values = _num(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.quantile(q)) if len(values) else float("nan")


def _rate(mask: pd.Series | Any) -> float:
    if not isinstance(mask, pd.Series):
        return float("nan")
    values = mask.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if len(values) else float("nan")


def _row_uid(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").astype(str)
    symbol = frame["symbol"].astype(str)
    side = _num(frame["side"]).fillna(0.0).map(lambda v: "long" if v > 0.0 else "short")
    return timestamp + "|" + symbol + "|" + side


def _posterior_cols(frame: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in frame.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[-1]) if c.rsplit("_", 1)[-1].isdigit() else 999)


def _argmax_bucket(frame: pd.DataFrame, cols: list[str], name: str) -> pd.Series:
    if not cols:
        return pd.Series("missing", index=frame.index, dtype="object")
    values = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(values).any(axis=1)
    filled = np.where(np.isfinite(values), values, -np.inf)
    idx = np.argmax(filled, axis=1)
    out = pd.Series([f"{name}_{int(i)}" for i in idx], index=frame.index, dtype="object")
    out.loc[missing] = "missing"
    return out


def _add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["row_uid"] = _row_uid(out)
    side_num = _num(out["side"]).fillna(0.0)
    out["side_name"] = np.where(side_num > 0.0, "long", "short")
    out["date"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.date.astype(str)
    out["global_archetype"] = _argmax_bucket(
        out,
        _posterior_cols(out, "ctx_gmm_cluster_posterior_"),
        "global",
    )
    long_bucket = _argmax_bucket(
        out,
        _posterior_cols(out, "ctx_long_gmm_cluster_posterior_"),
        "long",
    )
    short_bucket = _argmax_bucket(
        out,
        _posterior_cols(out, "ctx_short_gmm_cluster_posterior_"),
        "short",
    )
    out["side_archetype"] = np.where(side_num > 0.0, long_bucket, short_bucket)
    spread = _num(out.get("ctx_median_spread_bps", pd.Series(np.nan, index=out.index)))
    try:
        out["spread_bucket"] = pd.qcut(
            spread.rank(method="first"),
            q=min(5, max(1, int(spread.notna().sum()))),
            labels=["spread_q1_low", "spread_q2", "spread_q3", "spread_q4", "spread_q5_high"][
                : min(5, max(1, int(spread.notna().sum())))
            ],
            duplicates="drop",
        ).astype("object")
    except Exception:
        out["spread_bucket"] = "missing"
    out["spread_bucket"] = out["spread_bucket"].fillna("missing").astype(str)
    return out


def _summarize(group: pd.DataFrame, *, oracle_denominator: int | None = None) -> dict[str, Any]:
    rows = int(len(group))
    side = group.get("side_name", pd.Series(index=group.index, dtype="object")).astype(str)
    u = _num(group.get("u_policy_net", pd.Series(np.nan, index=group.index)))
    gain = u[u > 0.0].sum()
    loss = -u[u < 0.0].sum()
    oracle_hits = int(group.get("oracle_top", pd.Series(False, index=group.index)).astype(bool).sum())
    clean_oracle_hits = int(
        group.get("clean_oracle_top", pd.Series(False, index=group.index)).astype(bool).sum()
    )
    return {
        "rows": rows,
        "symbols": int(group.get("symbol", pd.Series(dtype=str)).astype(str).nunique()) if rows else 0,
        "days": int(group.get("date", pd.Series(dtype=str)).astype(str).nunique()) if rows else 0,
        "long_share": float((side == "long").mean()) if rows else float("nan"),
        "short_share": float((side == "short").mean()) if rows else float("nan"),
        "mean_u": _safe_mean(u),
        "median_u": _safe_quantile(u, 0.50),
        "p10_u": _safe_quantile(u, 0.10),
        "profit_factor": float(gain / loss) if loss > 0 else float("nan"),
        "bad_mae_1r_rate": _rate(group.get("bad_mae_1r", pd.Series(dtype=float)).astype(bool)),
        "timeout_rate": _rate(_num(group.get("is_timeout", pd.Series(dtype=float))) > 0.5),
        "clean_positive_rate": _rate(group.get("clean_positive", pd.Series(dtype=float)).astype(bool)),
        "dirty_positive_rate": _rate(group.get("dirty_positive", pd.Series(dtype=float)).astype(bool)),
        "oracle_hits": oracle_hits,
        "clean_oracle_hits": clean_oracle_hits,
        "oracle_recall": float(oracle_hits / oracle_denominator)
        if oracle_denominator
        else float("nan"),
    }


def _group_summary(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    variant_denominators: dict[str, int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        base = dict(zip(keys, key_values, strict=False))
        variant = str(base.get("meta_variant", ""))
        base.update(_summarize(group, oracle_denominator=(variant_denominators or {}).get(variant)))
        rows.append(base)
    return pd.DataFrame(rows)


def _schema_audit(frame: pd.DataFrame) -> dict[str, Any]:
    cols = list(frame.columns)
    derived_bucket_cols = {"global_archetype", "side_archetype"}
    hard_cluster_cols = [
        c
        for c in cols
        if c not in derived_bucket_cols
        and (
            "cluster_id" in c.lower()
            or c.lower().endswith("_cluster")
            or c.lower().endswith("_archetype")
        )
    ]
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(len(cols)),
        "n_ctx_columns": int(sum(c.startswith("ctx_") for c in cols)),
        "n_global_ae_gmm": int(sum(c.startswith("ctx_gmm_") or c.startswith("ctx_cluster_") for c in cols)),
        "n_long_ae_gmm": int(sum(c.startswith("ctx_long_") for c in cols)),
        "n_short_ae_gmm": int(sum(c.startswith("ctx_short_") for c in cols)),
        "n_soft_prob_features": int(sum("posterior_" in c or "gmm_prob_" in c for c in cols)),
        "n_distance_features": int(sum("dist_center" in c or "mahal" in c for c in cols)),
        "n_transition_features": int(sum(token in c for c in cols for token in ("delta_", "accel", "speed", "stability", "flip_count"))),
        "n_entropy_features": int(sum("entropy" in c for c in cols)),
        "n_reconstruction_features": int(sum("reconstruction" in c for c in cols)),
        "n_hard_cluster_id_features": int(len(hard_cluster_cols)),
        "hard_cluster_id_features": hard_cluster_cols[:50],
    }


def build_report(
    *,
    candidate_ledger_path: Path,
    selected_rows_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(candidate_ledger_path)
    selected = pd.read_parquet(selected_rows_path)
    ledger = _add_derived_columns(ledger)
    selected = _add_derived_columns(selected)

    periods_by_variant = {
        str(variant): set(group["period"].astype(str).unique())
        for variant, group in selected.groupby("meta_variant", sort=False)
    }
    oracle_denominators: dict[str, int] = {}
    for variant, periods in periods_by_variant.items():
        scoped = ledger[ledger["period"].astype(str).isin(periods)]
        oracle_denominators[variant] = int(scoped["oracle_top"].astype(bool).sum())

    overview = _group_summary(
        selected,
        ["meta_variant"],
        variant_denominators=oracle_denominators,
    )
    period_side = _group_summary(
        selected,
        ["meta_variant", "period", "side_name"],
        variant_denominators=oracle_denominators,
    )
    side_archetype = _group_summary(
        selected,
        ["meta_variant", "side_name", "side_archetype"],
        variant_denominators=oracle_denominators,
    )
    global_archetype = _group_summary(
        selected,
        ["meta_variant", "side_name", "global_archetype"],
        variant_denominators=oracle_denominators,
    )
    spread_side = _group_summary(
        selected,
        ["meta_variant", "spread_bucket", "side_name"],
        variant_denominators=oracle_denominators,
    )

    missed_rows: list[dict[str, Any]] = []
    selected_uids_by_variant = {
        str(variant): set(group["row_uid"].astype(str))
        for variant, group in selected.groupby("meta_variant", sort=False)
    }
    for variant, selected_uids in selected_uids_by_variant.items():
        scoped = ledger[ledger["period"].astype(str).isin(periods_by_variant[variant])].copy()
        scoped["selected_by_variant"] = scoped["row_uid"].astype(str).isin(selected_uids)
        oracle = scoped[scoped["oracle_top"].astype(bool)].copy()
        for keys, name in (
            (["side_name"], "side"),
            (["side_name", "side_archetype"], "side_archetype"),
            (["side_name", "global_archetype"], "global_archetype"),
            (["spread_bucket", "side_name"], "spread_side"),
        ):
            for key_values, group in oracle.groupby(keys, dropna=False, sort=True):
                if not isinstance(key_values, tuple):
                    key_values = (key_values,)
                row = {
                    "meta_variant": variant,
                    "slice_name": name,
                    **dict(zip(keys, key_values, strict=False)),
                    "oracle_rows": int(len(group)),
                    "selected_oracle_rows": int(group["selected_by_variant"].sum()),
                    "missed_oracle_rows": int((~group["selected_by_variant"]).sum()),
                    "selected_oracle_rate": float(group["selected_by_variant"].mean())
                    if len(group)
                    else float("nan"),
                    "mean_u_oracle": _safe_mean(group["u_policy_net"]),
                    "bad_mae_oracle": _rate(group["bad_mae_1r"].astype(bool)),
                    "timeout_oracle": _rate(_num(group["is_timeout"]) > 0.5),
                }
                missed_rows.append(row)
    missed_oracle = pd.DataFrame(missed_rows)

    paths = {
        "overview": output_dir / "meta_selected_overview.csv",
        "period_side": output_dir / "meta_selected_period_side.csv",
        "side_archetype": output_dir / "meta_selected_side_archetype.csv",
        "global_archetype": output_dir / "meta_selected_global_archetype.csv",
        "spread_side": output_dir / "meta_selected_spread_side.csv",
        "missed_oracle": output_dir / "meta_selected_missed_oracle.csv",
        "schema": output_dir / "meta_selected_schema_audit.json",
        "markdown": output_dir / "meta_selected_archetype_diagnostics.md",
    }
    overview.to_csv(paths["overview"], index=False)
    period_side.to_csv(paths["period_side"], index=False)
    side_archetype.to_csv(paths["side_archetype"], index=False)
    global_archetype.to_csv(paths["global_archetype"], index=False)
    spread_side.to_csv(paths["spread_side"], index=False)
    missed_oracle.to_csv(paths["missed_oracle"], index=False)
    schema = {
        "candidate_ledger": _schema_audit(ledger),
        "selected_rows": _schema_audit(selected),
        "oracle_denominators": oracle_denominators,
    }
    paths["schema"].write_text(json.dumps(schema, indent=2), encoding="utf-8")

    best = overview.sort_values(["bad_mae_1r_rate", "timeout_rate"], ascending=[True, True])
    recall = overview.sort_values("oracle_recall", ascending=False)
    md = [
        "# Meta Selected Archetype Diagnostics",
        "",
        "## Schema",
        "",
        f"- selected rows: `{schema['selected_rows']['n_rows']}`",
        f"- ctx columns: `{schema['selected_rows']['n_ctx_columns']}`",
        f"- long AE/GMM columns: `{schema['selected_rows']['n_long_ae_gmm']}`",
        f"- short AE/GMM columns: `{schema['selected_rows']['n_short_ae_gmm']}`",
        f"- hard cluster ID columns: `{schema['selected_rows']['n_hard_cluster_id_features']}`",
        "",
        "## Best Risk Rows",
        "",
        best.head(8).to_markdown(index=False),
        "",
        "## Best Recall Rows",
        "",
        recall.head(8).to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for name, path in paths.items():
        if name != "markdown":
            md.append(f"- {name}: `{path}`")
    paths["markdown"].write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"outputs": {k: str(v) for k, v in paths.items()}, "schema": schema}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ledger-path", type=Path, required=True)
    parser.add_argument("--selected-rows-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_report(
        candidate_ledger_path=args.candidate_ledger_path,
        selected_rows_path=args.selected_rows_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
