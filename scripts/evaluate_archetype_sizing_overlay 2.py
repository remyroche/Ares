"""Evaluate causal archetype sizing overlays on a frozen admitted-trade set.

The overlay is deliberately post-admission: no score, threshold, candidate, or
portfolio decision is changed.  Archetype expected net returns are estimated
only from the historical fit summary, shrunk to the side mean, converted to
bounded multipliers, and then applied to the already accepted parent trades.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def normal_key(value: object) -> str:
    text = str(value)
    return text[len("policy_archetype_") :] if text.startswith("policy_archetype_") else text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--accepted", required=True)
    p.add_argument("--fit-summary", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    accepted = pd.read_parquet(args.accepted).copy()
    fit = pd.read_csv(args.fit_summary).copy()
    fit["arch_key"] = fit["policy_archetype"].map(normal_key)
    fit["side_key"] = np.where(fit["side"].astype(str).str.lower().eq("short"), -1.0, 1.0)
    mu = pd.to_numeric(fit["stage_a_cumulative_mean_net_trade"], errors="coerce").fillna(0.0)
    n = pd.to_numeric(fit["rows"], errors="coerce").fillna(0.0).clip(lower=0.0)
    fit["raw_train_net"] = mu
    fit["train_rows"] = n
    # 1,000 rows is a fixed prior strength, specified before inspecting July.
    side_prior = fit.groupby("side_key").apply(
        lambda g: np.average(g["raw_train_net"], weights=np.maximum(g["train_rows"], 1.0)),
        include_groups=False,
    )
    fit["side_prior"] = fit["side_key"].map(side_prior)
    fit["shrunk_train_net"] = (
        fit["train_rows"] * fit["raw_train_net"] + 1000.0 * fit["side_prior"]
    ) / (fit["train_rows"] + 1000.0)
    # Relative per-side quality, bounded to prevent a sparse archetype from
    # becoming a de facto admission decision.
    fit["weighted_center"] = fit["side_key"].map(
        fit.groupby("side_key").apply(
            lambda g: np.average(g["shrunk_train_net"], weights=np.maximum(g["train_rows"], 1.0)),
            include_groups=False,
        )
    )
    fit["weighted_scale"] = fit["side_key"].map(
        fit.groupby("side_key").apply(
            lambda g: max(
                float(np.sqrt(np.average(
                    (g["shrunk_train_net"] - np.average(g["shrunk_train_net"], weights=np.maximum(g["train_rows"], 1.0))) ** 2,
                    weights=np.maximum(g["train_rows"], 1.0),
                ))),
                1e-6,
            ),
            include_groups=False,
        )
    )
    fit["quality_z"] = (fit["shrunk_train_net"] - fit["weighted_center"]) / fit["weighted_scale"]
    for strength in (0.10, 0.20):
        fit[f"size_multiplier_{int(strength * 100):02d}"] = np.clip(
            1.0 + strength * fit["quality_z"], 0.75, 1.25
        )

    accepted["arch_key"] = accepted["policy_archetype"].map(normal_key)
    accepted["side_key"] = np.where(
        accepted["side"].astype(str).str.lower().eq("short"), -1.0, 1.0
    )
    merged = accepted.merge(
        fit[["strategy_id", "arch_key", "side_key", "raw_train_net", "shrunk_train_net", "quality_z", "size_multiplier_10", "size_multiplier_20"]],
        on=["strategy_id", "arch_key", "side_key"], how="left", validate="many_to_one",
    )
    if merged["quality_z"].isna().any():
        raise ValueError("Frozen accepted trades contain an archetype absent from the fit summary")

    rows: list[dict[str, object]] = []
    for arm, multiplier in [("parent_fixed_size", np.ones(len(merged)))] + [
        ("archetype_eb_shrink_10pct", merged["size_multiplier_10"].to_numpy()),
        ("archetype_eb_shrink_20pct", merged["size_multiplier_20"].to_numpy()),
    ]:
        work = merged.copy()
        work["sizing_multiplier"] = multiplier
        work["sized_net_pnl"] = work["net_pnl"] * work["sizing_multiplier"]
        work["sized_gross_pnl"] = work["gross_pnl"] * work["sizing_multiplier"]
        work["sized_notional"] = work["position_size"] * work["sizing_multiplier"]
        for label, group in [("pooled", work), *[(str(k), g) for k, g in work.groupby("side", sort=True)]]:
            rows.append({
                "arm": arm,
                "scope": label,
                "trades": int(len(group)),
                "net_pnl": float(group["sized_net_pnl"].sum()),
                "gross_pnl": float(group["sized_gross_pnl"].sum()),
                "notional": float(group["sized_notional"].sum()),
                "net_bps_per_notional": float(1e4 * group["sized_net_pnl"].sum() / max(group["sized_notional"].sum(), 1e-12)),
                "gross_bps_per_notional": float(1e4 * group["sized_gross_pnl"].sum() / max(group["sized_notional"].sum(), 1e-12)),
                "mean_multiplier": float(group["sizing_multiplier"].mean()),
            })
        work["arm"] = arm
        work.to_parquet(args.output.rsplit(".", 1)[0] + f"_{arm}_trades.parquet", index=False)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    fit.to_csv(args.output.rsplit(".", 1)[0] + "_train_mapping.csv", index=False)


if __name__ == "__main__":
    main()
