import json
from pathlib import Path

import pandas as pd

from extreme_price_movements.inference.daily_reporter import DailyDeploymentReporter
from extreme_price_movements.inference.deployment_checks import (
    CHECK_NAMES,
    DeploymentCheckContext,
    require_deployment_checks,
    run_deployment_checks,
    summarize_deployment_checks,
)
from extreme_price_movements.inference.trade_logger import TradeLogger
from extreme_price_movements.portfolio_manager import PortfolioManager


class _Sizer:
    model_names_ = ["meta_pred", "calibrated_score"]


class _FakeFetcher:
    def __init__(self):
        self.dead_letter_symbols = {}
        self.api_error_counts = {}

    def fetch_hourly_universe_once(self, *args, **kwargs):
        return {}

    def has_recent_gap(self, *args, **kwargs):
        return False


class _FakeExchange:
    def __init__(self):
        self.transfers = []

    def fetch_balance(self, params=None):
        self.balance_params = params
        return {
            "total": {"USDC": 1_100.0},
            "free": {"USDC": 1_000.0},
            "used": {"USDC": 100.0},
        }

    def fetch_positions(self, params=None):
        self.position_params = params
        return [{"symbol": "BTC/USDT", "contracts": 0.1}]

    def sapiPostAssetTransfer(self, payload):
        self.transfers.append(payload)
        return {"tranId": "deployment-check"}


class _FakeSMTP:
    sent_messages = []
    logins = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starttls(self):
        return None

    def login(self, user, password):
        self.logins.append((user, password))

    def send_message(self, message):
        self.sent_messages.append(message)


def _write_artifacts(tmp_path: Path, run_id: str) -> None:
    run_dir = tmp_path / "artifacts" / run_id
    (run_dir / "labels").mkdir(parents=True)
    (run_dir / "meta_oof").mkdir(parents=True)
    (run_dir / "policy_params").mkdir(parents=True)
    (run_dir / "ridge_sizer").mkdir(parents=True)
    (run_dir / "simple_position_sizer").mkdir(parents=True)
    (run_dir / "base_meta_contract.json").write_text(
        json.dumps({"schema_version": "v1"})
    )
    (run_dir / "meta_oof" / "meta_feature_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "meta_models": {
                    "long_mr": {
                        "feature_columns": ["ret24h", "base_probability_long_mr"],
                        "positional_feature_mapping": {
                            "f0": "ret24h",
                            "f1": "base_probability_long_mr",
                        },
                        "n_features": 2,
                    }
                },
            }
        )
    )
    (run_dir / "labels" / "labels_manifest.json").write_text(
        json.dumps({"datasets": {"long_mr": {"rows": 10}}})
    )
    (run_dir / "policy_params" / "best_policy_params.json").write_text(
        json.dumps({"strategies": [{"strategy_id": "long_mr", "sl_mult": 1.0}]})
    )
    (run_dir / "strategy_for_inference.json").write_text(
        json.dumps({"strategies": [{"strategy_id": "long_mr"}]})
    )
    (run_dir / "ridge_sizer" / "strategy_params.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_mr",
                        "allow_downstream": True,
                        "wallet_pnl": 1.0,
                        "net_pnl": 1.0,
                    }
                ]
            }
        )
    )
    (run_dir / "ridge_sizer" / "confidence_calibration.contract.json").write_text(
        json.dumps(
            {
                "required_strategy_fields": ["p75_threshold", "calibration_curve"],
                "rank_semantics": "empirical_oof_rank_percentile",
            }
        )
    )
    (
        run_dir / "simple_position_sizer" / "confidence_calibration.contract.json"
    ).write_text(
        json.dumps(
            {
                "required_strategy_fields": ["p75_threshold", "calibration_curve"],
                "rank_semantics": "empirical_oof_rank_percentile",
            }
        )
    )
    (run_dir / "ridge_sizer" / "confidence_calibration.json").write_text(
        json.dumps(
            {
                "strategies": {
                    "long_mr": {
                        "p75_threshold": 0.75,
                        "calibration_curve": [[0.0, 0.0], [1.0, 1.0]],
                    }
                }
            }
        )
    )
    (run_dir / "simple_position_sizer" / "confidence_calibration.json").write_text(
        json.dumps(
            {
                "strategies": {
                    "long_mr": {
                        "p75_threshold": 0.75,
                        "calibration_curve": [[0.0, 0.0], [1.0, 1.0]],
                    }
                }
            }
        )
    )


def test_step12_deployment_checks_cover_all_required_items(tmp_path, monkeypatch):
    run_id = "20260101_000000"
    _write_artifacts(tmp_path, run_id)
    _FakeSMTP.sent_messages = []
    _FakeSMTP.logins = []
    monkeypatch.setenv("GMAIL_USER", "sender@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_PORT", "2525")

    trade_logger = TradeLogger(
        output_path=str(tmp_path / "deployment_trades.csv"),
        run_id=run_id,
    )
    trade_logger.log_entry(
        symbol="BTC/USDT",
        side="long",
        size=10.0,
        price=100.0,
        predictions={"meta_pred": 0.9},
        features={"strategy_id": "long_mr", "net_pnl": 1.5},
        mode="shadow",
    )
    state_path = tmp_path / "daily_report_state.json"
    state_path.write_text(
        json.dumps(
            {
                "previous_best_balance_usdt": 1_000.0,
                "last_report_ts": "2026-01-01T00:00:00+00:00",
                "last_trade_report_ts": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    ctx = DeploymentCheckContext(
        data_root=str(tmp_path),
        run_id=run_id,
        model_bundle={
            "ridge_sizer": _Sizer(),
            "bucket_params": {"buckets": {"long_mr": {"sl_mult": 1.0}}},
            "bundle": {
                "alpha_models": {
                    "long_mr": {"model": object(), "feat_cols": ["ret24h"]}
                },
                "meta_models": {"long_mr": object()},
            },
        },
        calibration_data={
            "long_mr": {
                "p75_threshold": 0.75,
                "calibration_curve": [[0.0, 0.0], [1.0, 1.0]],
            }
        },
        accepted_strategies={"long_mr"},
        candidate_selection_probe=lambda accepted: {
            "passed": "long_mr" in (accepted or set()),
            "top_quartile_meta_gate": True,
        },
        data_fetcher=_FakeFetcher(),
        portfolio_mgr=PortfolioManager(portfolio_value=1_000.0),
        trade_logger=trade_logger,
        daily_reporter=DailyDeploymentReporter(
            state_path=str(state_path),
            smtp_factory=_FakeSMTP,
            env_file=str(tmp_path / ".env"),
        ),
        exchange=_FakeExchange(),
        now=pd.Timestamp("2026-01-02T00:01:00Z"),
        temp_dir=str(tmp_path),
    )

    results = require_deployment_checks(ctx)
    summary = summarize_deployment_checks(results)

    assert [result.name for result in results] == list(CHECK_NAMES)
    assert summary == {"total": 15, "passed": 15, "failed": 0, "failures": []}
    assert len(_FakeSMTP.sent_messages) == 1


def test_deployment_checks_report_missing_artifact_manifest(tmp_path):
    ctx = DeploymentCheckContext(
        data_root=str(tmp_path),
        run_id="20260101_000000",
    )

    results = run_deployment_checks(ctx, checks=["Artifact manifest verified"])

    assert len(results) == 1
    assert results[0].passed is False
    assert "labels_manifest" in results[0].error
