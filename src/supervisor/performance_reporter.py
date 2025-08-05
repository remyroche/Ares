import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class AdvancedReportingEngine:
    """Advanced reporting engine with real-time analytics and comprehensive analysis."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("AdvancedReportingEngine")
        self.report_templates: dict[str, Any] = {}
        self.real_time_metrics: dict[str, Any] = {}
        self.performance_trends: dict[str, list[float]] = {}

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def generate_real_time_report(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate real-time performance report with advanced analytics."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "real_time_metrics": await self._calculate_real_time_metrics(
                    performance_data,
                ),
                "performance_trends": await self._analyze_performance_trends(
                    performance_data,
                ),
                "risk_analysis": await self._perform_risk_analysis(performance_data),
                "attribution_analysis": await self._perform_attribution_analysis(
                    performance_data,
                ),
                "forecasting": await self._generate_performance_forecast(
                    performance_data,
                ),
            }

            # Cache the report
            self._cache_report("real_time", report)

            return report

        except Exception as e:
            self.logger.error(f"Error generating real-time report: {e}")
            return {}

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def _calculate_real_time_metrics(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate real-time performance metrics."""
        try:
            returns = performance_data.get("returns", [])
            if not returns:
                return {}

            metrics = {
                "current_return": returns[-1] if returns else 0,
                "rolling_1h_return": np.mean(returns[-60:])
                if len(returns) >= 60
                else np.mean(returns),
                "rolling_24h_return": np.mean(returns[-1440:])
                if len(returns) >= 1440
                else np.mean(returns),
                "volatility": np.std(returns[-100:])
                if len(returns) >= 100
                else np.std(returns),
                "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                "max_drawdown": self._calculate_max_drawdown(returns),
                "win_rate": self._calculate_win_rate(returns),
                "profit_factor": self._calculate_profit_factor(returns),
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating real-time metrics: {e}")
            return {}

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def _analyze_performance_trends(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            returns = performance_data.get("returns", [])
            if not returns:
                return {}

            # Calculate trend indicators
            short_trend = (
                np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
            )
            medium_trend = (
                np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
            )
            long_trend = (
                np.mean(returns[-500:]) if len(returns) >= 500 else np.mean(returns)
            )

            trends = {
                "short_term_trend": "bullish" if short_trend > 0 else "bearish",
                "medium_term_trend": "bullish" if medium_trend > 0 else "bearish",
                "long_term_trend": "bullish" if long_trend > 0 else "bearish",
                "trend_strength": abs(short_trend - long_trend),
                "trend_consistency": self._calculate_trend_consistency(returns),
            }

            return trends

        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {}

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def _perform_risk_analysis(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform comprehensive risk analysis."""
        try:
            returns = performance_data.get("returns", [])
            if not returns:
                return {}

            risk_analysis = {
                "var_95": self._calculate_var(returns, 0.95),
                "var_99": self._calculate_var(returns, 0.99),
                "expected_shortfall": self._calculate_expected_shortfall(returns),
                "downside_deviation": self._calculate_downside_deviation(returns),
                "tail_risk": self._calculate_tail_risk(returns),
                "risk_adjusted_return": self._calculate_risk_adjusted_return(returns),
            }

            return risk_analysis

        except Exception as e:
            self.logger.error(f"Error performing risk analysis: {e}")
            return {}

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def _perform_attribution_analysis(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform performance attribution analysis."""
        try:
            attribution = {
                "market_timing": self._calculate_market_timing_contribution(
                    performance_data,
                ),
                "stock_selection": self._calculate_stock_selection_contribution(
                    performance_data,
                ),
                "risk_management": self._calculate_risk_management_contribution(
                    performance_data,
                ),
                "leverage_usage": self._calculate_leverage_contribution(
                    performance_data,
                ),
            }

            return attribution

        except Exception as e:
            self.logger.error(f"Error performing attribution analysis: {e}")
            return {}

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def _generate_performance_forecast(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate performance forecast using simple models."""
        try:
            returns = performance_data.get("returns", [])
            if len(returns) < 10:
                return {}

            # Simple forecasting using moving averages
            short_ma = np.mean(returns[-10:])
            medium_ma = np.mean(returns[-30:]) if len(returns) >= 30 else short_ma
            long_ma = np.mean(returns[-100:]) if len(returns) >= 100 else medium_ma

            forecast = {
                "next_period_forecast": short_ma,
                "confidence_interval": [
                    short_ma - np.std(returns[-10:]),
                    short_ma + np.std(returns[-10:]),
                ],
                "trend_forecast": "bullish" if short_ma > long_ma else "bearish",
                "forecast_horizon": "1_period",
            }

            return forecast

        except Exception as e:
            self.logger.error(f"Error generating performance forecast: {e}")
            return {}

    def _calculate_sharpe_ratio(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio."""
        try:
            if not returns:
                return 0.0
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            return mean_return / std_return if std_return > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, returns: list[float]) -> float:
        """Calculate maximum drawdown."""
        try:
            if not returns:
                return 0.0
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except Exception:
            return 0.0

    def _calculate_win_rate(self, returns: list[float]) -> float:
        """Calculate win rate."""
        try:
            if not returns:
                return 0.0
            wins = sum(1 for r in returns if r > 0)
            return wins / len(returns)
        except Exception:
            return 0.0

    def _calculate_profit_factor(self, returns: list[float]) -> float:
        """Calculate profit factor."""
        try:
            if not returns:
                return 0.0
            gains = sum(r for r in returns if r > 0)
            losses = abs(sum(r for r in returns if r < 0))
            return gains / losses if losses > 0 else float("inf")
        except Exception:
            return 0.0

    def _calculate_trend_consistency(self, returns: list[float]) -> float:
        """Calculate trend consistency."""
        try:
            if len(returns) < 2:
                return 0.0
            # Calculate correlation between consecutive returns
            return np.corrcoef(returns[:-1], returns[1:])[0, 1]
        except Exception:
            return 0.0

    def _calculate_var(self, returns: list[float], confidence_level: float) -> float:
        """Calculate Value at Risk."""
        try:
            if not returns:
                return 0.0
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception:
            return 0.0

    def _calculate_expected_shortfall(self, returns: list[float]) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        try:
            if not returns:
                return 0.0
            var_95 = self._calculate_var(returns, 0.95)
            tail_returns = [r for r in returns if r <= var_95]
            return np.mean(tail_returns) if tail_returns else 0.0
        except Exception:
            return 0.0

    def _calculate_downside_deviation(self, returns: list[float]) -> float:
        """Calculate downside deviation."""
        try:
            if not returns:
                return 0.0
            negative_returns = [r for r in returns if r < 0]
            return np.std(negative_returns) if negative_returns else 0.0
        except Exception:
            return 0.0

    def _calculate_tail_risk(self, returns: list[float]) -> float:
        """Calculate tail risk."""
        try:
            if not returns:
                return 0.0
            # Calculate kurtosis as a measure of tail risk
            return np.mean((np.array(returns) - np.mean(returns)) ** 4) / (
                np.std(returns) ** 4
            )
        except Exception:
            return 0.0

    def _calculate_risk_adjusted_return(self, returns: list[float]) -> float:
        """Calculate risk-adjusted return."""
        try:
            if not returns:
                return 0.0
            mean_return = np.mean(returns)
            downside_dev = self._calculate_downside_deviation(returns)
            return mean_return / downside_dev if downside_dev > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_market_timing_contribution(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate market timing contribution."""
        try:
            # Mock calculation - replace with actual market timing analysis
            return {
                "contribution": 0.15,
                "method": "regression_analysis",
                "significance": "high",
            }
        except Exception:
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

    def _calculate_stock_selection_contribution(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate stock selection contribution."""
        try:
            # Mock calculation - replace with actual stock selection analysis
            return {
                "contribution": 0.25,
                "method": "factor_analysis",
                "significance": "high",
            }
        except Exception:
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

    def _calculate_risk_management_contribution(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate risk management contribution."""
        try:
            # Mock calculation - replace with actual risk management analysis
            return {
                "contribution": 0.10,
                "method": "risk_decomposition",
                "significance": "medium",
            }
        except Exception:
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

    def _calculate_leverage_contribution(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate leverage contribution."""
        try:
            # Mock calculation - replace with actual leverage analysis
            return {
                "contribution": 0.05,
                "method": "leverage_analysis",
                "significance": "low",
            }
        except Exception:
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

    def _cache_report(self, report_type: str, report_data: dict[str, Any]) -> None:
        """Cache report data."""
        try:
            self.report_templates[report_type] = {
                "data": report_data,
                "timestamp": time.time(),
            }
        except Exception as e:
            self.logger.error(f"Error caching report: {e}")


class PerformanceReporter:
    """
    Enhanced Performance Reporter component with DI, type hints, robust error handling,
    and advanced reporting capabilities.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PerformanceReporter")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.reporter_config: dict[str, Any] = self.config.get(
            "performance_reporter",
            {},
        )
        self.report_interval: int = self.reporter_config.get("report_interval", 3600)
        self.max_history: int = self.reporter_config.get("max_history", 100)
        self.reports: list[dict[str, Any]] = []
        self.report_templates: dict[str, Any] = {}
        self.max_reports: int = self.reporter_config.get("max_reports", 100)
        self.attribution_config: dict[str, Any] = self.reporter_config.get(
            "attribution",
            {},
        )
        self.attribution_factors: list[str] = self.attribution_config.get(
            "factors",
            ["market_timing", "stock_selection", "risk_management", "leverage_usage"],
        )

        # Advanced reporting engine
        self.advanced_engine = AdvancedReportingEngine(
            self.reporter_config.get("advanced_engine", {}),
        )

        # Real-time reporting
        self.enable_real_time_reporting: bool = self.reporter_config.get(
            "enable_real_time_reporting",
            True,
        )
        self.real_time_interval: int = self.reporter_config.get(
            "real_time_interval",
            60,
        )  # 1 minute

        # Report export settings
        self.export_formats: list[str] = self.reporter_config.get(
            "export_formats",
            ["json", "csv"],
        )
        self.export_directory: str = self.reporter_config.get(
            "export_directory",
            "reports",
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid performance reporter configuration"),
            AttributeError: (False, "Missing required performance reporter parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="performance reporter initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Performance Reporter...")
            await self._load_reporter_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for performance reporter")
                return False
            await self._setup_advanced_reporting()
            await self._setup_real_time_reporting()
            await self._setup_export_directory()
            self.logger.info(
                "✅ Performance Reporter initialization completed successfully",
            )
            return True
        except Exception as e:
            self.logger.error(f"❌ Performance Reporter initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="reporter configuration loading",
    )
    async def _load_reporter_configuration(self) -> None:
        try:
            self.reporter_config.setdefault("report_interval", 3600)
            self.reporter_config.setdefault("max_history", 100)
            self.reporter_config.setdefault("enable_real_time_reporting", True)
            self.reporter_config.setdefault("real_time_interval", 60)
            self.reporter_config.setdefault("export_formats", ["json", "csv"])
            self.reporter_config.setdefault("export_directory", "reports")
            self.report_interval = self.reporter_config["report_interval"]
            self.max_history = self.reporter_config["max_history"]
            self.enable_real_time_reporting = self.reporter_config[
                "enable_real_time_reporting"
            ]
            self.real_time_interval = self.reporter_config["real_time_interval"]
            self.export_formats = self.reporter_config["export_formats"]
            self.export_directory = self.reporter_config["export_directory"]
            self.logger.info("Performance reporter configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading reporter configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.report_interval <= 0:
                self.logger.error("Invalid report interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            if self.real_time_interval <= 0:
                self.logger.error("Invalid real-time interval")
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="advanced reporting setup",
    )
    async def _setup_advanced_reporting(self) -> None:
        """Setup advanced reporting capabilities."""
        try:
            # Initialize advanced reporting engine
            self.advanced_engine = AdvancedReportingEngine(
                self.reporter_config.get("advanced_engine", {}),
            )

            self.logger.info("Advanced reporting setup complete")
        except Exception as e:
            self.logger.error(f"Error setting up advanced reporting: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="real-time reporting setup",
    )
    async def _setup_real_time_reporting(self) -> None:
        """Setup real-time reporting capabilities."""
        try:
            if self.enable_real_time_reporting:
                # Start real-time reporting task
                asyncio.create_task(self._real_time_reporting_task())
                self.logger.info("Real-time reporting setup complete")
            else:
                self.logger.info("Real-time reporting disabled")
        except Exception as e:
            self.logger.error(f"Error setting up real-time reporting: {e}")

    async def _real_time_reporting_task(self) -> None:
        """Background task for real-time reporting."""
        while self.is_running:
            try:
                await self._generate_real_time_report()
                await asyncio.sleep(self.real_time_interval)
            except Exception as e:
                self.logger.error(f"Error in real-time reporting task: {e}")
                await asyncio.sleep(self.real_time_interval)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="export directory setup",
    )
    async def _setup_export_directory(self) -> None:
        """Setup export directory for reports."""
        try:
            os.makedirs(self.export_directory, exist_ok=True)
            self.logger.info(
                f"Export directory setup complete: {self.export_directory}",
            )
        except Exception as e:
            self.logger.error(f"Error setting up export directory: {e}")

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Performance reporter run failed"),
        },
        default_return=False,
        context="performance reporter run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Performance Reporter started.")
            while self.is_running:
                await self._generate_performance_report()
                await asyncio.sleep(self.report_interval)
            return True
        except Exception as e:
            self.logger.error(f"Error in performance reporter run: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance report generation",
    )
    async def _generate_performance_report(self) -> None:
        try:
            self.logger.info("📊 Generating performance report...")

            # Collect performance data
            performance_data = await self._collect_performance_data()

            # Generate comprehensive report
            report = await self._create_advanced_report(performance_data)

            # Export report
            await self._export_report(report)

            # Store report
            self.reports.append(report)
            if len(self.reports) > self.max_reports:
                self.reports.pop(0)

            self.logger.info("✅ Performance report generated successfully")

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="real-time report generation",
    )
    async def _generate_real_time_report(self) -> None:
        try:
            # Collect real-time performance data
            performance_data = await self._collect_performance_data()

            # Generate real-time report using advanced engine
            real_time_report = await self.advanced_engine.generate_real_time_report(
                performance_data,
            )

            # Store real-time report
            self.real_time_metrics = real_time_report

            self.logger.debug("Real-time report updated")

        except Exception as e:
            self.logger.error(f"Error generating real-time report: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance data collection",
    )
    async def _collect_performance_data(self) -> dict[str, Any]:
        try:
            # Mock performance data - replace with actual data collection
            performance_data = {
                "returns": [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003, 0.012],
                "positions": [
                    {"symbol": "ETHUSDT", "size": 0.1, "pnl": 0.01},
                    {"symbol": "BTCUSDT", "size": 0.05, "pnl": -0.005},
                ],
                "trades": [
                    {"symbol": "ETHUSDT", "side": "buy", "size": 0.1, "price": 2000},
                    {"symbol": "BTCUSDT", "side": "sell", "size": 0.05, "price": 50000},
                ],
                "metrics": {"total_pnl": 0.015, "win_rate": 0.75, "sharpe_ratio": 1.2},
            }

            return performance_data

        except Exception as e:
            self.logger.error(f"Error collecting performance data: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="advanced report creation",
    )
    async def _create_advanced_report(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            # Generate comprehensive report using advanced engine
            advanced_report = await self.advanced_engine.generate_real_time_report(
                performance_data,
            )

            # Add attribution analysis
            attribution_analysis = self.analyze_performance_attribution(
                performance_data,
            )
            advanced_report["attribution_analysis"] = attribution_analysis

            # Add timestamp and metadata
            advanced_report["metadata"] = {
                "report_type": "comprehensive",
                "generated_at": datetime.now().isoformat(),
                "data_points": len(performance_data.get("returns", [])),
                "report_version": "2.0",
            }

            return advanced_report

        except Exception as e:
            self.logger.error(f"Error creating advanced report: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="report export",
    )
    async def _export_report(self, report: dict[str, Any]) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for export_format in self.export_formats:
                if export_format == "json":
                    await self._export_json_report(report, timestamp)
                elif export_format == "csv":
                    await self._export_csv_report(report, timestamp)

        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="JSON report export",
    )
    async def _export_json_report(self, report: dict[str, Any], timestamp: str) -> None:
        try:
            filename = f"performance_report_{timestamp}.json"
            filepath = os.path.join(self.export_directory, filename)

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"JSON report exported: {filepath}")

        except Exception as e:
            self.logger.error(f"Error exporting JSON report: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="CSV report export",
    )
    async def _export_csv_report(self, report: dict[str, Any], timestamp: str) -> None:
        try:
            filename = f"performance_report_{timestamp}.csv"
            filepath = os.path.join(self.export_directory, filename)

            # Convert report to CSV format
            csv_data = self._convert_report_to_csv(report)

            with open(filepath, "w") as f:
                f.write(csv_data)

            self.logger.info(f"CSV report exported: {filepath}")

        except Exception as e:
            self.logger.error(f"Error exporting CSV report: {e}")

    def _convert_report_to_csv(self, report: dict[str, Any]) -> str:
        """Convert report to CSV format."""
        try:
            csv_lines = []

            # Add header
            csv_lines.append("Metric,Value")

            # Flatten report structure
            for section, data in report.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        csv_lines.append(f"{section}_{key},{value}")
                else:
                    csv_lines.append(f"{section},{data}")

            return "\n".join(csv_lines)

        except Exception as e:
            self.logger.error(f"Error converting report to CSV: {e}")
            return "Metric,Value\nError,Conversion failed"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance reporter stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Performance Reporter...")
        try:
            self.is_running = False
            self.logger.info("✅ Performance Reporter stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping performance reporter: {e}")

    def get_status(self) -> dict[str, Any]:
        return {
            "is_running": self.is_running,
            "report_interval": self.report_interval,
            "max_history": self.max_history,
            "enable_real_time_reporting": self.enable_real_time_reporting,
            "real_time_interval": self.real_time_interval,
            "export_formats": self.export_formats,
            "export_directory": self.export_directory,
        }

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_reports(self, limit: int | None = None) -> list[dict[str, Any]]:
        reports = self.reports.copy()
        if limit:
            reports = reports[-limit:]
        return reports

    def get_latest_report(self) -> dict[str, Any] | None:
        return self.reports[-1] if self.reports else None

    def get_real_time_metrics(self) -> dict[str, Any]:
        return self.real_time_metrics.copy()

    def analyze_performance_attribution(
        self,
        portfolio_data: dict[str, Any],
        benchmark_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Analyze performance attribution with enhanced factors."""
        try:
            attribution_results = {
                "timestamp": datetime.now().isoformat(),
                "factors": {},
            }

            # Calculate factor contributions
            for factor in self.attribution_factors:
                contribution = self._calculate_factor_contribution(
                    factor,
                    portfolio_data,
                    benchmark_data,
                )
                attribution_results["factors"][factor] = contribution

            # Calculate total attribution
            total_contribution = sum(
                contribution.get("contribution", 0)
                for contribution in attribution_results["factors"].values()
            )

            attribution_results["total_contribution"] = total_contribution
            attribution_results["unexplained"] = 1.0 - total_contribution

            return attribution_results

        except Exception as e:
            self.logger.error(f"Error analyzing performance attribution: {e}")
            return {"error": str(e)}

    def _calculate_factor_contribution(
        self,
        factor: str,
        portfolio_data: dict,
        benchmark_data: dict = None,
    ) -> dict[str, Any]:
        """Calculate contribution of a specific factor."""
        try:
            if factor == "market_timing":
                return self._calculate_market_timing_contribution(
                    portfolio_data,
                    benchmark_data,
                )
            if factor == "stock_selection":
                return self._calculate_stock_selection_contribution(
                    portfolio_data,
                    benchmark_data,
                )
            if factor == "risk_management":
                return self._calculate_risk_management_contribution(portfolio_data)
            if factor == "leverage_usage":
                return self._calculate_leverage_contribution(portfolio_data)
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

        except Exception as e:
            self.logger.error(
                f"Error calculating factor contribution for {factor}: {e}",
            )
            return {"contribution": 0.0, "method": "error", "significance": "low"}

    def _calculate_market_timing_contribution(
        self,
        portfolio_data: dict,
        benchmark_data: dict = None,
    ) -> dict[str, Any]:
        """Calculate market timing contribution."""
        try:
            # Mock calculation - replace with actual market timing analysis
            return {
                "contribution": 0.15,
                "method": "regression_analysis",
                "significance": "high",
                "details": {"timing_score": 0.75, "timing_accuracy": 0.68},
            }
        except Exception:
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

    def _calculate_stock_selection_contribution(
        self,
        portfolio_data: dict,
        benchmark_data: dict = None,
    ) -> dict[str, Any]:
        """Calculate stock selection contribution."""
        try:
            # Mock calculation - replace with actual stock selection analysis
            return {
                "contribution": 0.25,
                "method": "factor_analysis",
                "significance": "high",
                "details": {"selection_score": 0.82, "selection_accuracy": 0.71},
            }
        except Exception:
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

    def _calculate_risk_management_contribution(
        self,
        portfolio_data: dict,
    ) -> dict[str, Any]:
        """Calculate risk management contribution."""
        try:
            # Mock calculation - replace with actual risk management analysis
            return {
                "contribution": 0.10,
                "method": "risk_decomposition",
                "significance": "medium",
                "details": {"risk_score": 0.65, "risk_efficiency": 0.73},
            }
        except Exception:
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

    def _calculate_leverage_contribution(self, portfolio_data: dict) -> dict[str, Any]:
        """Calculate leverage contribution."""
        try:
            # Mock calculation - replace with actual leverage analysis
            return {
                "contribution": 0.05,
                "method": "leverage_analysis",
                "significance": "low",
                "details": {"leverage_score": 0.45, "leverage_efficiency": 0.58},
            }
        except Exception:
            return {"contribution": 0.0, "method": "unknown", "significance": "low"}

    def _decompose_risk(self, portfolio_data: dict) -> dict[str, Any]:
        """Decompose risk into various components."""
        try:
            returns = portfolio_data.get("returns", [])
            if not returns:
                return {}

            risk_decomposition = {
                "total_risk": np.std(returns),
                "systematic_risk": np.std(returns) * 0.7,  # Mock calculation
                "idiosyncratic_risk": np.std(returns) * 0.3,  # Mock calculation
                "downside_risk": self._calculate_downside_deviation(returns),
                "tail_risk": self._calculate_tail_risk(returns),
            }

            return risk_decomposition

        except Exception as e:
            self.logger.error(f"Error decomposing risk: {e}")
            return {}

    def _analyze_timing(self, portfolio_data: dict) -> dict[str, Any]:
        """Analyze market timing effectiveness."""
        try:
            returns = portfolio_data.get("returns", [])
            if not returns:
                return {}

            timing_analysis = {
                "timing_accuracy": 0.68,  # Mock calculation
                "timing_score": 0.75,  # Mock calculation
                "timing_contribution": 0.15,  # Mock calculation
                "timing_consistency": 0.72,  # Mock calculation
            }

            return timing_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing timing: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: list) -> float:
        """Calculate maximum drawdown."""
        try:
            if not returns:
                return 0.0
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except Exception:
            return 0.0

    def _calculate_downside_deviation(self, returns: list) -> float:
        """Calculate downside deviation."""
        try:
            if not returns:
                return 0.0
            negative_returns = [r for r in returns if r < 0]
            return np.std(negative_returns) if negative_returns else 0.0
        except Exception:
            return 0.0

    def _calculate_tail_risk(self, returns: list) -> float:
        """Calculate tail risk."""
        try:
            if not returns:
                return 0.0
            # Calculate kurtosis as a measure of tail risk
            return np.mean((np.array(returns) - np.mean(returns)) ** 4) / (
                np.std(returns) ** 4
            )
        except Exception:
            return 0.0


performance_reporter: PerformanceReporter | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="performance reporter setup",
)
async def setup_performance_reporter(
    config: dict[str, Any] | None = None,
) -> PerformanceReporter | None:
    try:
        global performance_reporter
        if config is None:
            config = {
                "performance_reporter": {"report_interval": 3600, "max_history": 100},
            }
        performance_reporter = PerformanceReporter(config)
        success = await performance_reporter.initialize()
        if success:
            return performance_reporter
        return None
    except Exception as e:
        print(f"Error setting up performance reporter: {e}")
        return None
