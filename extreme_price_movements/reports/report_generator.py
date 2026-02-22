"""
Pipeline Report Generator — produces detailed .md reports for each pipeline step.

Reports are saved to extreme_price_movements/reports/<run_id>/
  - training_report.md
  - risk_optimization_report.md
  - backtest_report.md
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


DEFAULT_REPORTS_DIR = Path(__file__).parent


def _ensure_dir(run_id: str) -> Path:
    reports_dir = Path(os.environ.get("EPM_REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))
    d = reports_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _pct(v, decimals=2):
    return f"{v * 100:.{decimals}f}%"


# ──────────────────────────────────────────────
# TRAINING REPORT
# ──────────────────────────────────────────────
def generate_training_report(
    run_id: str,
    cfg: Dict[str, Any],
    bundle: Dict[str, Any],
    datasets: Dict[str, Any],
    specialist_models: Optional[Dict] = None,
    extra_info: Optional[Dict] = None,
) -> str:
    out = _ensure_dir(run_id)
    lines = []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# Training Report — {run_id}")
    lines.append(f"Generated: {ts}\n")

    # ── Config summary ──
    lines.append("## Configuration")
    lines.append(f"- **Train lookback**: {cfg.get('train_lookback_hours', '?')} hours")
    lines.append(f"- **Label horizons**: {cfg.get('label_horizons_hours', '?')}")
    lines.append(f"- **Label method**: triple_barrier")
    lines.append(f"- **Label quantiles**: lo={cfg.get('label_quantile_lo', 0.3)}, hi={cfg.get('label_quantile_hi', 0.7)}")
    lines.append(f"- **OOS holdout**: {cfg.get('oos_holdout_days', 0)} days")
    lines.append(f"- **Min train samples**: {cfg.get('min_train_samples', '?')}")
    lines.append(f"- **Feature selection**: MDI (min={cfg.get('mdi_min_features', 30)}, cap={cfg.get('mdi_cumulative_cap', 0.995)})")
    lines.append(f"- **15m precision**: {cfg.get('use_15m_precision', False)}")
    lines.append("")

    # ── Dataset sizes ──
    lines.append("## Dataset Sizes")
    lines.append("| Dataset | Rows | Features |")
    lines.append("|---------|------|----------|")
    for key in sorted(datasets.keys()):
        df = datasets[key]
        if hasattr(df, 'shape'):
            n_rows = df.shape[0]
            n_cols = df.shape[1] if len(df.shape) > 1 else 1
            lines.append(f"| {key} | {n_rows:,} | {n_cols} |")
    lines.append("")

    # ── Alpha models ──
    alpha_models = bundle.get("alpha_models", {})
    lines.append("## Alpha Models")

    # Summary table
    lines.append("\n### Performance Summary")
    lines.append("| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |")
    lines.append("|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|")

    for side in ["long", "short"]:
        side_bundle = alpha_models.get(side, {})
        for kind in ["mr", "tf"]:
            model_info = side_bundle.get(kind, {})
            if not model_info:
                continue

            model_name = f"{side.upper()}_{kind.upper()}"
            feat_cols = model_info.get("feat_cols", [])
            model = model_info.get("model")
            best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else "N/A"

            # Extract metrics from extra_info
            oof_key = f"{side}_{kind}"
            oof = extra_info.get(f"oof_{oof_key}", {}) if extra_info else {}
            if not oof:
                oof = model_info.get("alpha_diag", {})

            auc = _fmt(oof.get('auc', 0.0)) if oof.get('auc') else "N/A"
            ic = _fmt(oof.get('ic', 0.0)) if oof.get('ic') else "N/A"
            sharpe = _fmt(oof.get('sharpe', 0.0)) if oof.get('sharpe') else "N/A"
            win_rate = _pct(oof.get('win_rate', 0.0)) if oof.get('win_rate') else "N/A"
            p10 = _fmt(oof.get('prec10', 0.0)) if oof.get('prec10') is not None else "N/A"
            p40 = _fmt(oof.get('prec40', 0.0)) if oof.get('prec40') is not None else "N/A"
            at10 = _fmt(oof.get('avg_trades_day_10', 0.0)) if oof.get('avg_trades_day_10') is not None else "N/A"
            at30 = _fmt(oof.get('avg_trades_day_30', 0.0)) if oof.get('avg_trades_day_30') is not None else "N/A"
            avg_ret = _fmt(oof.get('avg_return', 0.0), 6) if oof.get('avg_return') else "N/A"
            n_trades = oof.get('n_trades', 0) if oof.get('n_trades') else "N/A"

            lines.append(
                f"| {model_name} | {len(feat_cols)} | {auc} | {ic} | {sharpe} | "
                f"{win_rate} | {p10} | {p40} | {at10} | {at30} | {avg_ret} | {n_trades} | {best_iter} |"
            )
    lines.append("")

    # Detailed per-model breakdown
    lines.append("### Detailed Model Performance")
    for side in ["long", "short"]:
        side_bundle = alpha_models.get(side, {})
        for kind in ["mr", "tf"]:
            model_info = side_bundle.get(kind, {})
            if not model_info:
                lines.append(f"\n#### {side.upper()}_{kind.upper()}: **NOT TRAINED**\n")
                continue

            model = model_info.get("model")
            feat_cols = model_info.get("feat_cols", [])
            model_name = f"{side.upper()}_{kind.upper()}"

            lines.append(f"\n#### {model_name}")
            lines.append(f"- **Features**: {len(feat_cols)}")

            if hasattr(model, 'best_iteration_'):
                lines.append(f"- **Best iteration**: {model.best_iteration_}")

            # OOF metrics from extra_info
            if extra_info:
                oof_key = f"{side}_{kind}"
                oof = extra_info.get(f"oof_{oof_key}", {})
                if not oof:
                    oof = model_info.get("alpha_diag", {})
                if oof:
                    lines.append(f"- **OOF AUC**: {_fmt(oof.get('auc', 0.0))}")
                    lines.append(f"- **OOF IC**: {_fmt(oof.get('ic', 0.0))}")
                    lines.append(f"- **OOF Rank IC**: {_fmt(oof.get('rank_ic', 0.0))}")
                    lines.append(f"- **OOF Sharpe**: {_fmt(oof.get('sharpe', 0.0))}")
                    lines.append(f"- **OOF Win Rate**: {_pct(oof.get('win_rate', 0.0))}")
                    lines.append(f"- **OOF Avg Return**: {_fmt(oof.get('avg_return', 0.0), 6)}")
                    lines.append(f"- **OOF Max Drawdown**: {_fmt(oof.get('max_dd', 0.0), 6)}")
                    lines.append(f"- **OOF Sortino**: {_fmt(oof.get('sortino', 0.0))}")
                    lines.append(f"- **OOF Calmar**: {_fmt(oof.get('calmar', 0.0))}")
                    lines.append(f"- **OOF Trades**: {oof.get('n_trades', 0)}")
                    lines.append(f"- **OOF Prec@10**: {_fmt(oof.get('prec10', 0.0))}")
                    lines.append(f"- **OOF Prec@40**: {_fmt(oof.get('prec40', 0.0))}")
                    lines.append(f"- **OOF Avg Trades/Day @10%**: {_fmt(oof.get('avg_trades_day_10', 0.0))}")
                    lines.append(f"- **OOF Avg Trades/Day @30%**: {_fmt(oof.get('avg_trades_day_30', 0.0))}")
                    lines.append(f"- **OOF ECE@10**: {_fmt(oof.get('ece_top10', 0.0))}")
                    lines.append(f"- **OOF Calibration Profile**: {oof.get('calibration_profile', 'N/A')}")

                    # Per-regime breakdown if available
                    if 'per_regime' in oof:
                        lines.append(f"- **Per-Regime Performance**:")
                        for regime, metrics in oof['per_regime'].items():
                            lines.append(f"  - {regime}: AUC={_fmt(metrics.get('auc', 0.0))}, IC={_fmt(metrics.get('ic', 0.0))}, WR={_pct(metrics.get('win_rate', 0.0))}")

            # Per-regime BSS/AUC from model_info
            per_regime = model_info.get("per_regime", {})
            if per_regime:
                lines.append(f"\n##### Per-Regime BSS, Brier & AUC")
                lines.append("| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |")
                lines.append("|--------|----------------------------|----------------------------|------------------------------|")
                for rname, rbuckets in per_regime.items():
                    low = rbuckets.get("low", {})
                    mid = rbuckets.get("mid", {})
                    high = rbuckets.get("high", {})
                    lines.append(
                        f"| {rname} | "
                        f"{_fmt(low.get('bss', 0.0))} / {_fmt(low.get('brier', 0.0))} / {_fmt(low.get('auc', 0.5))} / {low.get('n', 0)} | "
                        f"{_fmt(mid.get('bss', 0.0))} / {_fmt(mid.get('brier', 0.0))} / {_fmt(mid.get('auc', 0.5))} / {mid.get('n', 0)} | "
                        f"{_fmt(high.get('bss', 0.0))} / {_fmt(high.get('brier', 0.0))} / {_fmt(high.get('auc', 0.5))} / {high.get('n', 0)} |"
                    )
                lines.append("")

            lines.append(f"- **Top features**: {', '.join(feat_cols[:10])}")
            lines.append("")

    # ── Meta models ──
    meta_models = bundle.get("meta_models", {})
    if meta_models:
        lines.append("## Meta Models")
        lines.append("\n### Performance Summary")
        lines.append("| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |")
        lines.append("|-------|----------|-----|----|---------| ---------|-------------|")

        for key, meta in meta_models.items():
            if meta is None:
                continue

            n_feats = len(meta.selected_features) if hasattr(meta, 'selected_features') and meta.selected_features else 0

            # Extract meta model metrics from extra_info
            meta_metrics = extra_info.get(f"meta_{key}", {}) if extra_info else {}

            auc = _fmt(meta_metrics.get('auc', 0.0)) if meta_metrics.get('auc') else "N/A"
            ic = _fmt(meta_metrics.get('ic', 0.0)) if meta_metrics.get('ic') else "N/A"
            sharpe = _fmt(meta_metrics.get('sharpe', 0.0)) if meta_metrics.get('sharpe') else "N/A"
            win_rate = _pct(meta_metrics.get('win_rate', 0.0)) if meta_metrics.get('win_rate') else "N/A"
            calib = _fmt(meta_metrics.get('calibration_error', 0.0)) if meta_metrics.get('calibration_error') else "N/A"

            lines.append(f"| {key} | {n_feats} | {auc} | {ic} | {sharpe} | {win_rate} | {calib} |")

        lines.append("\n### Detailed Meta Model Performance")
        for key, meta in meta_models.items():
            if meta is None:
                lines.append(f"\n#### {key}: **DISABLED**")
                continue

            n_feats = len(meta.selected_features) if hasattr(meta, 'selected_features') and meta.selected_features else 0
            lines.append(f"\n#### {key}")
            lines.append(f"- **Features**: {n_feats}")

            # Meta model metrics
            if extra_info:
                meta_metrics = extra_info.get(f"meta_{key}", {})
                if meta_metrics:
                    lines.append(f"- **AUC**: {_fmt(meta_metrics.get('auc', 0.0))}")
                    lines.append(f"- **IC**: {_fmt(meta_metrics.get('ic', 0.0))}")
                    lines.append(f"- **Sharpe**: {_fmt(meta_metrics.get('sharpe', 0.0))}")
                    lines.append(f"- **Win Rate**: {_pct(meta_metrics.get('win_rate', 0.0))}")
                    lines.append(f"- **Calibration Error**: {_fmt(meta_metrics.get('calibration_error', 0.0))}")
                    lines.append(f"- **Brier Score**: {_fmt(meta_metrics.get('brier_score', 0.0))}")
                    lines.append(f"- **Log Loss**: {_fmt(meta_metrics.get('log_loss', 0.0))}")

                    if hasattr(meta, 'selected_features') and meta.selected_features:
                        lines.append(f"- **Selected features**: {', '.join(meta.selected_features[:10])}")
        lines.append("")

    # ── Specialist models ──
    if specialist_models:
        lines.append("## Specialist Models")
        trap = specialist_models.get("trap_model")
        if trap:
            lines.append(f"- **Trap (GMM)**: {len(trap.get('columns', []))} features, clusters={trap.get('gmm', {})}")
        gamma = specialist_models.get("gamma_model")
        if gamma and hasattr(gamma, 'selected_features_'):
            lines.append(f"- **Gamma (ExtraTrees)**: {len(gamma.selected_features_)} features")
        lines.append("")

    # ── Write ──
    report_path = out / "training_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


# ──────────────────────────────────────────────
# RISK OPTIMIZATION REPORT
# ──────────────────────────────────────────────
def generate_risk_report(
    run_id: str,
    cfg: Dict[str, Any],
    granular_risk: Dict[str, Any],
    optimization_details: Optional[Dict[str, Any]] = None,
) -> str:
    out = _ensure_dir(run_id)
    lines = []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# Risk Optimization Report — {run_id}")
    lines.append(f"Generated: {ts}\n")

    lines.append("## Configuration")
    lines.append(f"- **15m precision**: {cfg.get('use_15m_precision', False)}")
    lines.append(f"- **Fee BPS**: {cfg.get('fee_bps', 25)}")
    lines.append(f"- **Max concurrent trades**: {cfg.get('max_concurrent_trades', 5)}")
    lines.append(f"- **Max portfolio weight**: {cfg.get('max_portfolio_weight', 0.25)}")
    lines.append(f"- **Daily cap per specialist**: {cfg.get('max_daily_per_specialist', 8)}")
    lines.append(f"- **Daily cap total**: {cfg.get('max_daily_total', 25)}")
    lines.append("")

    # ── Per-bucket risk params ──
    lines.append("## Optimized Risk Parameters")
    lines.append("| Bucket | TP | SL | Trail | BE% | Lock% | LockAmt% | MaxLoss% | Vol Lo | Vol Hi | Z Max | Hold (h) |")
    lines.append("|--------|----|----|-------|-----|-------|----------|----------|--------|--------|-------|----------|")
    for key in sorted(granular_risk.keys()):
        if not key.startswith("risk_"):
            continue
        rp = granular_risk[key]
        lines.append(
            f"| {key} | {rp.get('tp_mult', 0):.2f} | {rp.get('sl_mult', 0):.2f} | "
            f"{rp.get('trail_mult', 0):.2f} | {rp.get('be_threshold_pct', 0)*100:.1f} | "
            f"{rp.get('profit_lock_pct', 0)*100:.1f} | {rp.get('profit_lock_amount', 0)*100:.1f} | "
            f"{rp.get('max_loss_pct', 0)*100:.1f} | "
            f"{rp.get('vol_lo', 0):.3f} | {rp.get('vol_hi', 0):.3f} | "
            f"{rp.get('vol_z_max', 0):.1f} | {rp.get('max_hold_hours', '?')} |"
        )
    lines.append("")

    # ── Optimization details (outer fold results) ──
    if optimization_details:
        lines.append("## Outer Fold Results")
        for bucket_key, details in optimization_details.items():
            lines.append(f"\n### {bucket_key}")
            if "outer_results" in details:
                lines.append("| Fold | TP | SL | Lo | Hi | Z | AUC | IC | PnL |")
                lines.append("|------|----|----|----|----|---|-----|----|----|")
                for r in details["outer_results"]:
                    lines.append(
                        f"| {r.get('fold', '?')} | {r.get('tp', '?'):.2f} | {r.get('sl', '?'):.2f} | "
                        f"{r.get('lo', '?'):.2f} | {r.get('hi', '?'):.2f} | {r.get('z', '?'):.1f} | "
                        f"{r.get('auc', '?'):.4f} | {r.get('ic', '?'):.4f} | {r.get('pnl', '?'):.4f} |"
                    )
        lines.append("")

    report_path = out / "risk_optimization_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


# ──────────────────────────────────────────────
# BACKTEST REPORT
# ──────────────────────────────────────────────
def generate_backtest_report(
    run_id: str,
    cfg: Dict[str, Any],
    trades: List[Dict],
    signal_params: Dict[str, Any],
    fee_rate: float,
) -> str:
    out = _ensure_dir(run_id)
    lines = []
    ts_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if not trades:
        lines.append(f"# Backtest Report — {run_id}")
        lines.append(f"Generated: {ts_now}\n")
        lines.append("**No trades generated.**")
        report_path = out / "backtest_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    df = pd.DataFrame(trades)
    df["_entry"] = pd.to_datetime(df["entry_ts"])
    df["_exit"] = pd.to_datetime(df["exit_ts"])
    df["_hold_h"] = (df["_exit"] - df["_entry"]).dt.total_seconds() / 3600.0
    if "bucket" not in df.columns:
        df["bucket"] = df["side"].str.upper() + "_" + df["dom"].str.upper()

    n = len(df)
    ts_min = df["_entry"].min()
    ts_max = df["_entry"].max()
    n_days = max(1, (ts_max - ts_min).total_seconds() / 86400)

    rets = df["pnl"].values
    total_pnl = float(rets.sum())
    neg = rets[rets < 0]
    sortino = float(np.mean(rets) / (np.std(neg) + 1e-12)) if neg.size > 0 else 0.0
    eq = np.cumsum(rets)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(dd.min()) if dd.size else 0.0
    win_rate = float((rets > 0).mean())
    avg_win = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0
    avg_loss = float(rets[rets <= 0].mean()) if (rets <= 0).any() else 0.0
    payoff = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-9 else 0.0
    gross_profit = float(rets[rets > 0].sum())
    gross_loss = float(rets[rets <= 0].sum())
    pf = gross_profit / abs(gross_loss) if abs(gross_loss) > 1e-9 else float("inf")

    lines.append(f"# Backtest Report — {run_id}")
    lines.append(f"Generated: {ts_now}\n")

    # ── Summary ──
    lines.append("## Summary")
    lines.append(f"- **Period**: {ts_min.date()} to {ts_max.date()} ({n_days:.0f} days)")
    lines.append(f"- **Total trades**: {n} ({n/n_days:.1f}/day)")
    lines.append(f"- **Net PnL**: {total_pnl:+.6f}")
    lines.append(f"- **Sortino**: {sortino:.4f}")
    lines.append(f"- **Max Drawdown**: {max_dd:.4f}")
    lines.append(f"- **Win Rate**: {_pct(win_rate)}")
    lines.append(f"- **Profit Factor**: {pf:.3f}")
    lines.append(f"- **Payoff Ratio**: {payoff:.2f}")
    lines.append(f"- **Avg Win**: {avg_win:+.6f} | **Avg Loss**: {avg_loss:+.6f}")
    lines.append(f"- **Fee rate**: {fee_rate*10000:.0f} bps")
    lines.append(f"- **15m precision**: {cfg.get('use_15m_precision', False)}")
    lines.append("")

    # ── Signal params ──
    lines.append("## Signal Parameters")
    for k, v in sorted(signal_params.items()):
        if isinstance(v, dict):
            continue
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # ── Per-bucket breakdown ──
    lines.append("## Per-Bucket Performance")
    lines.append("| Bucket | N | N/day | PnL | WR | Sortino | MaxDD | PF | AvgWin | AvgLoss | Payoff |")
    lines.append("|--------|---|-------|-----|----|---------|----- -|----|--------|---------|--------|")
    for bkt in sorted(df["bucket"].unique()):
        db = df[df["bucket"] == bkt]
        br = db["pnl"].values
        b_pnl = float(br.sum())
        b_wr = float((br > 0).mean())
        b_neg = br[br < 0]
        b_sort = float(np.mean(br) / (np.std(b_neg) + 1e-12)) if b_neg.size > 0 else 0.0
        b_eq = np.cumsum(br)
        b_pk = np.maximum.accumulate(b_eq)
        b_dd = float((b_eq - b_pk).min()) if b_eq.size else 0.0
        b_gp = float(br[br > 0].sum())
        b_gl = float(br[br <= 0].sum())
        b_pf = b_gp / abs(b_gl) if abs(b_gl) > 1e-9 else float("inf")
        b_aw = float(br[br > 0].mean()) if (br > 0).any() else 0.0
        b_al = float(br[br <= 0].mean()) if (br <= 0).any() else 0.0
        b_po = abs(b_aw / b_al) if abs(b_al) > 1e-9 else 0.0
        lines.append(
            f"| {bkt} | {len(db)} | {len(db)/n_days:.1f} | {b_pnl:+.4f} | {_pct(b_wr)} | "
            f"{b_sort:.3f} | {b_dd:.4f} | {b_pf:.2f} | {b_aw:+.6f} | {b_al:+.6f} | {b_po:.2f} |"
        )
    lines.append("")

    # ── Exit reasons ──
    lines.append("## Exit Reasons")
    lines.append("| Reason | N | % | PnL | WR | Avg Hold (h) |")
    lines.append("|--------|---|---|-----|----|--------------|")
    for reason in sorted(df["reason"].dropna().unique()):
        dr = df[df["reason"] == reason]
        r_pnl = dr["pnl"].sum()
        r_wr = (dr["pnl"] > 0).mean()
        r_hold = dr["_hold_h"].mean()
        lines.append(
            f"| {reason} | {len(dr)} | {_pct(len(dr)/n)} | {r_pnl:+.4f} | {_pct(r_wr)} | {r_hold:.1f} |"
        )
    lines.append("")

    # ── MAE/MFE ──
    if "mae_pct" in df.columns and "mfe_pct" in df.columns:
        lines.append("## MAE / MFE Analysis")
        lines.append(f"- **Global MAE**: mean={_pct(df['mae_pct'].mean())}, med={_pct(df['mae_pct'].median())}, q90={_pct(df['mae_pct'].quantile(0.9))}")
        lines.append(f"- **Global MFE**: mean={_pct(df['mfe_pct'].mean())}, med={_pct(df['mfe_pct'].median())}, q90={_pct(df['mfe_pct'].quantile(0.9))}")
        ratio = df['mfe_pct'].mean() / max(df['mae_pct'].mean(), 1e-9)
        lines.append(f"- **MFE/MAE ratio**: {ratio:.2f}")
        lines.append("")

        lines.append("### Per-Bucket MAE/MFE")
        lines.append("| Bucket | MAE mean | MFE mean | MFE/MAE | Losers w/ MFE>0.5% | Winner Capture |")
        lines.append("|--------|----------|----------|---------|---------------------|----------------|")
        for bkt in sorted(df["bucket"].unique()):
            db = df[df["bucket"] == bkt]
            if len(db) < 3:
                continue
            b_mae = db['mae_pct'].mean()
            b_mfe = db['mfe_pct'].mean()
            b_ratio = b_mfe / max(b_mae, 1e-9)
            losers = db[db["pnl"] <= 0]
            losers_mfe = losers[losers["mfe_pct"] > 0.005] if len(losers) > 0 else pd.DataFrame()
            pct_l_mfe = len(losers_mfe) / max(len(losers), 1)
            winners = db[db["pnl"] > 0]
            if len(winners) > 0 and "gross_ret" in winners.columns:
                capture = winners["gross_ret"].mean() / max(winners["mfe_pct"].mean(), 1e-9)
            else:
                capture = 0.0
            lines.append(
                f"| {bkt} | {_pct(b_mae)} | {_pct(b_mfe)} | {b_ratio:.2f} | "
                f"{_pct(pct_l_mfe)} ({len(losers_mfe)}/{len(losers)}) | {capture:.2f} |"
            )
        lines.append("")

    # ── PnL Reconciliation ──
    lines.append("## PnL Reconciliation")
    has_weight = "weight" in df.columns
    has_gross = "gross_ret" in df.columns
    if has_weight and has_gross:
        total_fees = float((2.0 * fee_rate * df["weight"]).sum())
        gross_pnl_pre = float(df["gross_ret"].mul(df["weight"]).sum())
        lines.append(f"- **Gross PnL (pre-fee)**: {gross_pnl_pre:+.6f}")
        lines.append(f"- **Total fees**: {total_fees:+.6f}")
        lines.append(f"- **Net PnL (post-fee)**: {gross_pnl_pre - total_fees:+.6f}")
    lines.append(f"- **Gross Profit**: {gross_profit:+.6f}")
    lines.append(f"- **Gross Loss**: {gross_loss:+.6f}")
    lines.append("")

    lines.append("### Per-Bucket Contribution")
    lines.append("| Bucket | N | Gross Profit | Gross Loss | Net PnL | PF | WR |")
    lines.append("|--------|---|-------------|------------|---------|----|----|")
    for bkt in sorted(df["bucket"].unique()):
        db = df[df["bucket"] == bkt]
        b_gp = float(db.loc[db["pnl"] > 0, "pnl"].sum())
        b_gl = float(db.loc[db["pnl"] <= 0, "pnl"].sum())
        b_net = b_gp + b_gl
        b_pf = b_gp / abs(b_gl) if abs(b_gl) > 1e-9 else float("inf")
        b_wr = (db["pnl"] > 0).mean()
        lines.append(f"| {bkt} | {len(db)} | {b_gp:+.6f} | {b_gl:+.6f} | {b_net:+.6f} | {b_pf:.2f} | {_pct(b_wr)} |")
    lines.append("")

    # ── Daily concentration ──
    df["_date"] = df["_entry"].dt.date
    daily_counts = df.groupby("_date").size()
    lines.append("## Daily Concentration")
    lines.append(f"- **Max trades/day**: {daily_counts.max()}")
    lines.append(f"- **Mean trades/day**: {daily_counts.mean():.1f}")
    lines.append("")

    lines.append("### Per-Bucket Daily Max")
    lines.append("| Bucket | Max/day | Mean/day |")
    lines.append("|--------|---------|----------|")
    for bkt in sorted(df["bucket"].unique()):
        db = df[df["bucket"] == bkt]
        bkt_daily = db.groupby("_date").size()
        lines.append(f"| {bkt} | {bkt_daily.max()} | {bkt_daily.mean():.1f} |")
    lines.append("")

    # ── Exit Stage Analysis ──
    if "exit_stage" in df.columns:
        lines.append("## Exit Stage Analysis")
        lines.append("Exit stages: 0=initial SL, 1=break-even, 2=tight trail, 3=full trail")
        lines.append("")
        lines.append("### Global")
        lines.append("| Stage | N | % | PnL | WR | Avg Hold (h) |")
        lines.append("|-------|---|---|-----|----|--------------|")
        for stage in sorted(df["exit_stage"].unique()):
            ds = df[df["exit_stage"] == stage]
            s_pnl = ds["pnl"].sum()
            s_wr = (ds["pnl"] > 0).mean()
            s_hold = ds["_hold_h"].mean()
            lines.append(f"| Stage {stage} | {len(ds)} | {_pct(len(ds)/n)} | {s_pnl:+.4f} | {_pct(s_wr)} | {s_hold:.1f} |")
        lines.append("")

        lines.append("### Per-Bucket Exit Stages")
        lines.append("| Bucket | Stage | N | % | PnL | WR |")
        lines.append("|--------|-------|---|---|-----|----|")
        for bkt in sorted(df["bucket"].unique()):
            db = df[df["bucket"] == bkt]
            for stage in sorted(db["exit_stage"].unique()):
                ds = db[db["exit_stage"] == stage]
                s_pnl = ds["pnl"].sum()
                s_wr = (ds["pnl"] > 0).mean()
                lines.append(f"| {bkt} | {stage} | {len(ds)} | {_pct(len(ds)/len(db))} | {s_pnl:+.4f} | {_pct(s_wr)} |")
        lines.append("")

    # ── Per-Regime Metrics ──
    # Analyze outcomes by regime features (gate columns in trade records)
    regime_cols = [c for c in df.columns if c.startswith("G_") or c in ("vol_regime", "trend_regime", "mkt_regime")]
    # If no explicit regime columns, try to infer from score/dom patterns
    if not regime_cols and "dom" in df.columns:
        lines.append("## Per-Regime Metrics")
        lines.append("")

        # Analyze by side x dom (which is effectively the regime bucket)
        lines.append("### Performance by Side x Dominance")
        lines.append("| Side | Dom | N | PnL | WR | PF | Avg MFE | Avg MAE | SL% |")
        lines.append("|------|-----|---|-----|----|----|---------|---------|----|")
        for side_val in ["long", "short"]:
            for dom_val in ["mr", "tf"]:
                mask = (df["side"] == side_val) & (df["dom"] == dom_val)
                ds = df[mask]
                if len(ds) < 5:
                    continue
                s_pnl = ds["pnl"].sum()
                s_wr = (ds["pnl"] > 0).mean()
                s_gp = float(ds.loc[ds["pnl"] > 0, "pnl"].sum())
                s_gl = float(ds.loc[ds["pnl"] <= 0, "pnl"].sum())
                s_pf = s_gp / abs(s_gl) if abs(s_gl) > 1e-9 else float("inf")
                s_mfe = ds["mfe_pct"].mean() if "mfe_pct" in ds.columns else 0.0
                s_mae = ds["mae_pct"].mean() if "mae_pct" in ds.columns else 0.0
                s_sl = (ds["reason"] == "stop_loss").mean() if "reason" in ds.columns else 0.0
                lines.append(f"| {side_val} | {dom_val} | {len(ds)} | {s_pnl:+.4f} | {_pct(s_wr)} | {s_pf:.2f} | {_pct(s_mfe)} | {_pct(s_mae)} | {_pct(s_sl)} |")
        lines.append("")

        # Analyze by score magnitude quartiles per bucket
        if "score" in df.columns:
            lines.append("### Performance by Score Quartile (per bucket)")
            lines.append("| Bucket | Quartile | N | PnL | WR | Avg |score| | SL% |")
            lines.append("|--------|----------|---|-----|----|-----------|-----|")
            for bkt in sorted(df["bucket"].unique()):
                db = df[df["bucket"] == bkt].copy()
                if len(db) < 20:
                    continue
                try:
                    db["_sq"] = pd.qcut(db["score"].abs(), q=4, labels=["Q1_Low", "Q2", "Q3", "Q4_High"], duplicates="drop")
                    for q in ["Q1_Low", "Q2", "Q3", "Q4_High"]:
                        dq = db[db["_sq"] == q]
                        if len(dq) == 0:
                            continue
                        q_pnl = dq["pnl"].sum()
                        q_wr = (dq["pnl"] > 0).mean()
                        q_abs = dq["score"].abs().mean()
                        q_sl = (dq["reason"] == "stop_loss").mean() if "reason" in dq.columns else 0.0
                        lines.append(f"| {bkt} | {q} | {len(dq)} | {q_pnl:+.4f} | {_pct(q_wr)} | {q_abs:.4f} | {_pct(q_sl)} |")
                except Exception:
                    pass
            lines.append("")

        # Analyze by hold duration quartiles
        lines.append("### Performance by Hold Duration")
        lines.append("| Bucket | Duration | N | PnL | WR | Reason Breakdown |")
        lines.append("|--------|----------|---|-----|----|-----------------|")
        for bkt in sorted(df["bucket"].unique()):
            db = df[df["bucket"] == bkt].copy()
            if len(db) < 10:
                continue
            try:
                db["_hq"] = pd.cut(db["_hold_h"], bins=[0, 2, 6, 12, 100], labels=["0-2h", "2-6h", "6-12h", "12h+"])
                for hq in ["0-2h", "2-6h", "6-12h", "12h+"]:
                    dh = db[db["_hq"] == hq]
                    if len(dh) == 0:
                        continue
                    h_pnl = dh["pnl"].sum()
                    h_wr = (dh["pnl"] > 0).mean()
                    reasons = dh["reason"].value_counts(normalize=True)
                    reason_str = " | ".join([f"{r}:{_pct(p)}" for r, p in reasons.items()])
                    lines.append(f"| {bkt} | {hq} | {len(dh)} | {h_pnl:+.4f} | {_pct(h_wr)} | {reason_str} |")
            except Exception:
                pass
        lines.append("")

    elif regime_cols:
        lines.append("## Per-Regime Metrics")
        for rc in regime_cols:
            lines.append(f"\n### Regime: {rc}")
            lines.append("| Value | N | PnL | WR | PF |")
            lines.append("|-------|---|-----|----|----| ")
            for val in sorted(df[rc].dropna().unique()):
                dr = df[df[rc] == val]
                r_pnl = dr["pnl"].sum()
                r_wr = (dr["pnl"] > 0).mean()
                r_gp = float(dr.loc[dr["pnl"] > 0, "pnl"].sum())
                r_gl = float(dr.loc[dr["pnl"] <= 0, "pnl"].sum())
                r_pf = r_gp / abs(r_gl) if abs(r_gl) > 1e-9 else float("inf")
                lines.append(f"| {val} | {len(dr)} | {r_pnl:+.4f} | {_pct(r_wr)} | {r_pf:.2f} |")
        lines.append("")

    # ── Weekly PnL ──
    df["_week"] = df["_entry"].dt.isocalendar().week.astype(int)
    weekly = df.groupby("_week").agg(
        n=("pnl", "count"),
        pnl=("pnl", "sum"),
        wr=("pnl", lambda x: (x > 0).mean())
    )
    lines.append("## Weekly PnL")
    lines.append("| Week | N | PnL | WR |")
    lines.append("|------|---|-----|----|")
    for wk, row in weekly.iterrows():
        lines.append(f"| W{wk:02d} | {int(row['n'])} | {row['pnl']:+.4f} | {_pct(row['wr'])} |")
    lines.append("")

    # ── Write ──
    report_path = out / "backtest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)
