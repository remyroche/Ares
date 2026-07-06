import pandas as pd

from scripts.run_s23_local_abstention_diagnostic import (
    DEFAULT_THRESHOLDS,
    _active_bucket_policy,
    _apply_policy,
    _metric_row,
    _oracle_total,
)


def _row(period: str, side: float, spread: int, clean: bool, oracle: bool) -> dict:
    return {
        "period": period,
        "symbol": "AAA/USD:USD" if spread == 0 else "BBB/USD:USD",
        "side": side,
        "side_bucket": "long" if side > 0 else "short",
        "spread_bucket": spread,
        "liquidity_bucket": 0,
        "ctx_state_spectral_top3_reconstruction_error_bucket": 0,
        "ctx_q_iqr__bars_in_high_vol_state_log_norm_bucket": 0,
        "ctx_q_tail_width__bars_in_high_vol_state_log_norm_bucket": 0,
        "u_policy_net": 0.003 if clean else 0.001,
        "bad_mae_1r": 0 if clean else 1,
        "is_timeout": 0,
        "clean_positive": clean,
        "dirty_positive": not clean,
        "oracle_top": oracle,
        "oracle_rows_total": 2,
        "selector_score": 0.9 if clean else 0.2,
    }


def test_s23_policy_uses_prior_bucket_evidence_only() -> None:
    train = pd.DataFrame(
        [_row("2026-04", 1.0, 0, True, True) for _ in range(12)]
        + [_row("2026-04", 1.0, 0, False, False) for _ in range(8)]
    )
    valid = pd.DataFrame(
        [_row("2026-05", 1.0, 0, True, True)]
        + [_row("2026-05", -1.0, 1, True, True)]
    )
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds["min_selected_rows"] = 10
    diagnostics, policy = _active_bucket_policy(
        train,
        valid_period="2026-05",
        thresholds=thresholds,
    )
    accepted = _apply_policy(valid, policy)

    assert not diagnostics.empty
    assert int(policy["active"].sum()) == 1
    assert len(accepted) == 1
    assert accepted["spread_bucket"].iloc[0] == 0
    assert accepted["side_bucket"].iloc[0] == "long"


def test_s23_no_trade_fold_keeps_valid_oracle_denominator() -> None:
    valid = pd.DataFrame([_row("2026-05", 1.0, 0, True, True) for _ in range(3)])
    accepted = valid.iloc[:0]
    metrics = _metric_row(accepted)
    valid_oracle_total = _oracle_total(valid)
    metrics["oracle_rows_total"] = valid_oracle_total
    metrics["final_oracle_recall"] = (
        metrics["oracle_hit_rows"] / valid_oracle_total if valid_oracle_total else float("nan")
    )

    assert metrics["oracle_hit_rows"] == 0
    assert metrics["oracle_rows_total"] == 2
    assert metrics["final_oracle_recall"] == 0
