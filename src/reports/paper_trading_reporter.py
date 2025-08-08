#!/usr/bin/env python3
"""
Comprehensive Paper Trading Reporter

This module provides detailed reporting for paper trading with comprehensive trade tracking,
including PnL analysis, trade types, position sizing, indicators, market health metrics,
and ML ensemble confidence scores.
"""

import json
import os
import sys
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from dataclasses_json import dataclass_json

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


@dataclass_json
@dataclass
class TradeType:
    """Trade type classification."""
    
    side: str  # "long", "short"
    leverage: float
    duration: str  # "scalping", "day_trading", "swing", "position"
    strategy: str  # "breakout", "mean_reversion", "momentum", etc.
    order_type: str  # "market", "limit", "stop", "stop_limit"


@dataclass_json
@dataclass
class PositionSizing:
    """Position sizing information."""
    
    absolute_size: float
    portfolio_percentage: float
    risk_percentage: float
    max_position_size: float
    position_ranking: int  # Position size ranking in portfolio


@dataclass_json
@dataclass
class PnLAnalysis:
    """PnL analysis for trades."""
    
    absolute_pnl: float
    percentage_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_cost: float
    total_proceeds: float
    commission_paid: float
    slippage_paid: float
    net_pnl: float


@dataclass_json
@dataclass
class MarketIndicators:
    """Market indicators at trade time."""
    
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    atr: float
    volume_sma: float
    price_sma_20: float
    price_sma_50: float
    price_sma_200: float
    volatility: float
    momentum: float
    support_level: float
    resistance_level: float


@dataclass_json
@dataclass
class MarketHealth:
    """Market health indicators."""
    
    overall_health_score: float
    volatility_regime: str
    liquidity_score: float
    stress_score: float
    market_strength: float
    volume_health: str
    price_trend: str
    market_regime: str


@dataclass_json
@dataclass
class MLConfidenceScores:
    """ML ensemble confidence scores."""
    
    analyst_confidence: float
    tactician_confidence: float
    ensemble_confidence: float
    meta_learner_confidence: float
    individual_model_confidences: Dict[str, float]
    ensemble_agreement: float
    model_diversity: float
    prediction_consistency: float


@dataclass_json
@dataclass
class DetailedTradeRecord:
    """Comprehensive trade record with all required information."""
    # Basic trade information
    trade_id: str
    symbol: str
    exchange: str
    timestamp: datetime
    trade_type: TradeType
    position_sizing: PositionSizing
    pnl_analysis: PnLAnalysis
    market_indicators: MarketIndicators
    market_health: MarketHealth
    ml_confidence_scores: MLConfidenceScores
    trade_status: str  # "open", "closed", "cancelled"
    # Fields with defaults must come after all non-default fields
    close_timestamp: Optional[datetime] = None
    close_reason: Optional[str] = None
    execution_quality: float = 0.0
    risk_metrics: Dict[str, float] = None
    notes: Optional[str] = None


class PaperTradingReporter:
    """
    Comprehensive paper trading reporter that tracks detailed trade information
    including PnL, trade types, position sizing, indicators, market health,
    and ML confidence scores.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize paper trading reporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PaperTradingReporter")
        
        # Trade records storage
        self.trade_records: List[DetailedTradeRecord] = []
        self.trade_counter = 0
        
        # Reporting configuration
        self.report_config = config.get("paper_trading_reporter", {})
        self.enable_detailed_reporting = self.report_config.get("enable_detailed_reporting", True)
        # Patch: Always use absolute path for report directory
        report_dir = self.report_config.get("report_directory", "reports/paper_trading")
        if not os.path.isabs(report_dir):
            # Use workspace root as base
            workspace_root = os.environ.get("WORKSPACE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
            report_dir = os.path.abspath(os.path.join(workspace_root, report_dir))
        self.report_directory = report_dir
        self.export_formats = self.report_config.get("export_formats", ["json", "csv", "html"])
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.portfolio_summary: Dict[str, Any] = {}
        
        # Initialize report directory
        os.makedirs(self.report_directory, exist_ok=True)
        # Log current working directory and report directory
        cwd = os.getcwd()
        self.logger.debug(f"CWD: {cwd}")
        self.logger.debug(f"Report directory (abs): {self.report_directory}")
    
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid trade data"),
            KeyError: (False, "Missing required trade fields"),
        },
        default_return=False,
        context="trade recording",
    )
    async def record_trade(
        self,
        trade_data: Dict[str, Any],
        market_indicators: Dict[str, Any],
        market_health: Dict[str, Any],
        ml_confidence: Dict[str, Any],
    ) -> bool:
        """
        Record a detailed trade with all required information.
        
        Args:
            trade_data: Basic trade information
            market_indicators: Market indicators at trade time
            market_health: Market health metrics
            ml_confidence: ML ensemble confidence scores
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate trade ID
            trade_id = f"trade_{self.trade_counter:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trade_counter += 1
            
            # Create trade type
            trade_type = TradeType(
                side=trade_data.get("side", "unknown"),
                leverage=trade_data.get("leverage", 1.0),
                duration=trade_data.get("duration", "unknown"),
                strategy=trade_data.get("strategy", "unknown"),
                order_type=trade_data.get("order_type", "market"),
            )
            
            # Create position sizing
            position_sizing = PositionSizing(
                absolute_size=trade_data.get("quantity", 0.0),
                portfolio_percentage=trade_data.get("portfolio_percentage", 0.0),
                risk_percentage=trade_data.get("risk_percentage", 0.0),
                max_position_size=trade_data.get("max_position_size", 0.0),
                position_ranking=trade_data.get("position_ranking", 0),
            )
            
            # Create PnL analysis
            pnl_analysis = PnLAnalysis(
                absolute_pnl=trade_data.get("absolute_pnl", 0.0),
                percentage_pnl=trade_data.get("percentage_pnl", 0.0),
                unrealized_pnl=trade_data.get("unrealized_pnl", 0.0),
                realized_pnl=trade_data.get("realized_pnl", 0.0),
                total_cost=trade_data.get("total_cost", 0.0),
                total_proceeds=trade_data.get("total_proceeds", 0.0),
                commission_paid=trade_data.get("commission", 0.0),
                slippage_paid=trade_data.get("slippage", 0.0),
                net_pnl=trade_data.get("net_pnl", 0.0),
            )
            
            # Create market indicators
            market_indicators_obj = MarketIndicators(
                rsi=market_indicators.get("rsi", 0.0),
                macd=market_indicators.get("macd", 0.0),
                macd_signal=market_indicators.get("macd_signal", 0.0),
                bollinger_upper=market_indicators.get("bollinger_upper", 0.0),
                bollinger_lower=market_indicators.get("bollinger_lower", 0.0),
                bollinger_middle=market_indicators.get("bollinger_middle", 0.0),
                atr=market_indicators.get("atr", 0.0),
                volume_sma=market_indicators.get("volume_sma", 0.0),
                price_sma_20=market_indicators.get("price_sma_20", 0.0),
                price_sma_50=market_indicators.get("price_sma_50", 0.0),
                price_sma_200=market_indicators.get("price_sma_200", 0.0),
                volatility=market_indicators.get("volatility", 0.0),
                momentum=market_indicators.get("momentum", 0.0),
                support_level=market_indicators.get("support_level", 0.0),
                resistance_level=market_indicators.get("resistance_level", 0.0),
            )
            
            # Create market health
            market_health_obj = MarketHealth(
                overall_health_score=market_health.get("overall_health_score", 0.0),
                volatility_regime=market_health.get("volatility_regime", "unknown"),
                liquidity_score=market_health.get("liquidity_score", 0.0),
                stress_score=market_health.get("stress_score", 0.0),
                market_strength=market_health.get("market_strength", 0.0),
                volume_health=market_health.get("volume_health", "unknown"),
                price_trend=market_health.get("price_trend", "unknown"),
                market_regime=market_health.get("market_regime", "unknown"),
            )
            
            # Create ML confidence scores
            ml_confidence_obj = MLConfidenceScores(
                analyst_confidence=ml_confidence.get("analyst_confidence", 0.0),
                tactician_confidence=ml_confidence.get("tactician_confidence", 0.0),
                ensemble_confidence=ml_confidence.get("ensemble_confidence", 0.0),
                meta_learner_confidence=ml_confidence.get("meta_learner_confidence", 0.0),
                individual_model_confidences=ml_confidence.get("individual_model_confidences", {}),
                ensemble_agreement=ml_confidence.get("ensemble_agreement", 0.0),
                model_diversity=ml_confidence.get("model_diversity", 0.0),
                prediction_consistency=ml_confidence.get("prediction_consistency", 0.0),
            )
            
            # Create detailed trade record
            trade_record = DetailedTradeRecord(
                trade_id=trade_id,
                symbol=trade_data.get("symbol", "unknown"),
                exchange=trade_data.get("exchange", "unknown"),
                timestamp=datetime.fromisoformat(trade_data.get("timestamp", datetime.now().isoformat())),
                trade_type=trade_type,
                position_sizing=position_sizing,
                pnl_analysis=pnl_analysis,
                market_indicators=market_indicators_obj,
                market_health=market_health_obj,
                ml_confidence_scores=ml_confidence_obj,
                trade_status=trade_data.get("status", "open"),
                execution_quality=trade_data.get("execution_quality", 0.0),
                risk_metrics=trade_data.get("risk_metrics", {}),
                notes=trade_data.get("notes"),
            )
            
            # Add to records
            self.trade_records.append(trade_record)
            
            # Update performance metrics
            await self._update_performance_metrics(trade_record)
            
            self.logger.info(f"✅ Recorded detailed trade: {trade_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance metrics update",
    )
    async def _update_performance_metrics(self, trade_record: DetailedTradeRecord) -> None:
        """Update performance metrics with new trade."""
        try:
            # Calculate basic metrics
            total_trades = len(self.trade_records)
            closed_trades = [t for t in self.trade_records if t.trade_status == "closed"]
            
            if closed_trades:
                # PnL metrics
                total_pnl = sum(t.pnl_analysis.net_pnl for t in closed_trades)
                total_cost = sum(t.pnl_analysis.total_cost for t in closed_trades)
                total_proceeds = sum(t.pnl_analysis.total_proceeds for t in closed_trades)
                
                # Win rate
                profitable_trades = [t for t in closed_trades if t.pnl_analysis.net_pnl > 0]
                win_rate = len(profitable_trades) / len(closed_trades)
                
                # Average PnL
                avg_pnl = total_pnl / len(closed_trades)
                avg_pnl_percentage = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
                
                # Risk metrics
                max_drawdown = self._calculate_max_drawdown()
                sharpe_ratio = self._calculate_sharpe_ratio()
                
                self.performance_metrics = {
                    "total_trades": total_trades,
                    "closed_trades": len(closed_trades),
                    "total_pnl": total_pnl,
                    "total_cost": total_cost,
                    "total_proceeds": total_proceeds,
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "avg_pnl_percentage": avg_pnl_percentage,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "last_updated": datetime.now().isoformat(),
                }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            if not self.trade_records:
                return 0.0
            
            # Calculate cumulative PnL
            cumulative_pnl = []
            current_pnl = 0.0
            
            for trade in self.trade_records:
                current_pnl += trade.pnl_analysis.net_pnl
                cumulative_pnl.append(current_pnl)
            
            # Calculate drawdown
            peak = cumulative_pnl[0]
            max_drawdown = 0.0
            
            for pnl in cumulative_pnl:
                peak = max(peak, pnl)
                drawdown = (peak - pnl) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(self.trade_records) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(self.trade_records)):
                prev_pnl = sum(t.pnl_analysis.net_pnl for t in self.trade_records[:i])
                curr_pnl = sum(t.pnl_analysis.net_pnl for t in self.trade_records[:i+1])
                ret = (curr_pnl - prev_pnl) / max(abs(prev_pnl), 1.0)
                returns.append(ret)
            
            if not returns:
                return 0.0
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            return avg_return / std_return if std_return > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trade update",
    )
    async def update_trade(
        self,
        trade_id: str,
        update_data: Dict[str, Any],
    ) -> bool:
        """
        Update an existing trade record.
        
        Args:
            trade_id: Trade ID to update
            update_data: Data to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find trade record
            trade_record = next((t for t in self.trade_records if t.trade_id == trade_id), None)
            
            if not trade_record:
                self.logger.warning(f"Trade record not found: {trade_id}")
                return False
            
            # Update fields
            if "close_timestamp" in update_data:
                trade_record.close_timestamp = datetime.fromisoformat(update_data["close_timestamp"])
            
            if "trade_status" in update_data:
                trade_record.trade_status = update_data["trade_status"]
            
            if "close_reason" in update_data:
                trade_record.close_reason = update_data["close_reason"]
            
            if "pnl_analysis" in update_data:
                pnl_data = update_data["pnl_analysis"]
                trade_record.pnl_analysis.absolute_pnl = pnl_data.get("absolute_pnl", 0.0)
                trade_record.pnl_analysis.percentage_pnl = pnl_data.get("percentage_pnl", 0.0)
                trade_record.pnl_analysis.realized_pnl = pnl_data.get("realized_pnl", 0.0)
                trade_record.pnl_analysis.net_pnl = pnl_data.get("net_pnl", 0.0)
            
            # Update performance metrics
            await self._update_performance_metrics(trade_record)
            
            self.logger.info(f"✅ Updated trade: {trade_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
            return False
    
    def get_trade(self, trade_id: str) -> Optional[DetailedTradeRecord]:
        """Get a specific trade record."""
        return next((t for t in self.trade_records if t.trade_id == trade_id), None)
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[DetailedTradeRecord]:
        """
        Get trade history with filters.
        
        Args:
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Limit number of records
            
        Returns:
            List[DetailedTradeRecord]: Filtered trade records
        """
        try:
            filtered_records = self.trade_records
            
            # Apply filters
            if symbol:
                filtered_records = [t for t in filtered_records if t.symbol == symbol]
            
            if start_time:
                filtered_records = [t for t in filtered_records if t.timestamp >= start_time]
            
            if end_time:
                filtered_records = [t for t in filtered_records if t.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            filtered_records.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            if limit:
                filtered_records = filtered_records[:limit]
            
            return filtered_records
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        try:
            if not self.trade_records:
                return {}
            
            # Calculate portfolio metrics
            total_value = sum(t.position_sizing.absolute_size for t in self.trade_records if t.trade_status == "open")
            total_pnl = sum(t.pnl_analysis.net_pnl for t in self.trade_records)
            
            # Position distribution
            symbol_positions = {}
            for trade in self.trade_records:
                if trade.trade_status == "open":
                    symbol = trade.symbol
                    if symbol not in symbol_positions:
                        symbol_positions[symbol] = {
                            "quantity": 0.0,
                            "value": 0.0,
                            "pnl": 0.0,
                        }
                    symbol_positions[symbol]["quantity"] += trade.position_sizing.absolute_size
                    symbol_positions[symbol]["value"] += trade.position_sizing.absolute_size * trade.pnl_analysis.total_cost
                    symbol_positions[symbol]["pnl"] += trade.pnl_analysis.net_pnl
            
            return {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "symbol_positions": symbol_positions,
                "last_updated": datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="report generation",
    )
    async def generate_detailed_report(
        self,
        report_type: str = "comprehensive",
        export_formats: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate detailed trading report.
        
        Args:
            report_type: Type of report to generate
            export_formats: Formats to export (json, csv, html)
            
        Returns:
            Dict[str, Any]: Generated report
        """
        try:
            print(f"[DEBUG] Generating {report_type} report with formats: {export_formats}")
            self.logger.info(f"[DEBUG] Generating {report_type} report with formats: {export_formats}")
            if export_formats is None:
                export_formats = self.export_formats
            
            # Generate report data
            report_data = {
                "report_type": report_type,
                "generated_at": datetime.now().isoformat(),
                "performance_metrics": self.get_performance_metrics(),
                "portfolio_summary": self.get_portfolio_summary(),
                "trade_records": [asdict(trade) for trade in self.trade_records],
                "analysis": await self._generate_analysis(),
            }
            
            # Export reports
            for format_type in export_formats:
                print(f"[DEBUG] Exporting report as {format_type}")
                self.logger.info(f"[DEBUG] Exporting report as {format_type}")
                await self._export_report(report_data, format_type)
            self.logger.info(f"✅ Generated {report_type} report")
            print(f"[DEBUG] Generated {report_type} report")
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            print(f"[DEBUG] Error generating report: {e}")
            return {}
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="analysis generation",
    )
    async def _generate_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis of trading performance."""
        try:
            if not self.trade_records:
                return {}
            
            # Trade type analysis
            trade_types = {}
            for trade in self.trade_records:
                trade_type_key = f"{trade.trade_type.side}_{trade.trade_type.strategy}"
                if trade_type_key not in trade_types:
                    trade_types[trade_type_key] = {
                        "count": 0,
                        "total_pnl": 0.0,
                        "win_rate": 0.0,
                        "avg_confidence": 0.0,
                    }
                
                trade_types[trade_type_key]["count"] += 1
                trade_types[trade_type_key]["total_pnl"] += trade.pnl_analysis.net_pnl
                trade_types[trade_type_key]["avg_confidence"] += trade.ml_confidence_scores.ensemble_confidence
            
            # Calculate win rates and average confidence
            for trade_type in trade_types.values():
                if trade_type["count"] > 0:
                    trade_type["avg_confidence"] /= trade_type["count"]
                    # Win rate calculation would need closed trades
            
            # Market health analysis
            market_health_scores = [t.market_health.overall_health_score for t in self.trade_records]
            avg_market_health = np.mean(market_health_scores) if market_health_scores else 0.0
            
            # ML confidence analysis
            confidence_scores = [t.ml_confidence_scores.ensemble_confidence for t in self.trade_records]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return {
                "trade_type_analysis": trade_types,
                "market_health_analysis": {
                    "average_health_score": avg_market_health,
                    "health_score_distribution": self._calculate_distribution(market_health_scores),
                },
                "confidence_analysis": {
                    "average_confidence": avg_confidence,
                    "confidence_distribution": self._calculate_distribution(confidence_scores),
                },
            }
            
        except Exception as e:
            self.logger.error(f"Error generating analysis: {e}")
            return {}
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate distribution of values."""
        try:
            if not values:
                return {}
            
            # Create bins
            min_val, max_val = min(values), max(values)
            bin_size = (max_val - min_val) / 10 if max_val > min_val else 1.0
            
            distribution = {}
            for i in range(10):
                bin_start = min_val + i * bin_size
                bin_end = min_val + (i + 1) * bin_size
                bin_key = f"{bin_start:.2f}-{bin_end:.2f}"
                distribution[bin_key] = len([v for v in values if bin_start <= v < bin_end])
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error calculating distribution: {e}")
            return {}
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="report export",
    )
    async def _export_report(
        self,
        report_data: Dict[str, Any],
        format_type: str,
    ) -> None:
        """
        Export report in specified format.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            abs_report_dir = os.path.abspath(self.report_directory)
            os.makedirs(abs_report_dir, exist_ok=True)  # Ensure directory exists before every write
            if format_type == "json":
                filename = f"paper_trading_report_{timestamp}.json"
                filepath = os.path.join(abs_report_dir, filename)
                abs_filepath = os.path.abspath(filepath)
                with open(filepath, "w") as f:
                    json.dump(report_data, f, indent=2, default=str)
                print(f"[DEBUG] Exported JSON report: {abs_filepath}")
                self.logger.info(f"[DEBUG] Exported JSON report: {abs_filepath}")
            elif format_type == "csv":
                filename = f"paper_trading_report_{timestamp}.csv"
                filepath = os.path.join(abs_report_dir, filename)
                abs_filepath = os.path.abspath(filepath)
                df = self._convert_to_dataframe(report_data)
                df.to_csv(filepath, index=False)
                print(f"[DEBUG] Exported CSV report: {abs_filepath}")
                self.logger.info(f"[DEBUG] Exported CSV report: {abs_filepath}")
            elif format_type == "html":
                filename = f"paper_trading_report_{timestamp}.html"
                filepath = os.path.join(abs_report_dir, filename)
                abs_filepath = os.path.abspath(filepath)
                html_content = self._generate_html_report(report_data)
                with open(filepath, "w") as f:
                    f.write(html_content)
                print(f"[DEBUG] Exported HTML report: {abs_filepath}")
                self.logger.info(f"[DEBUG] Exported HTML report: {abs_filepath}")
        except Exception as e:
            self.logger.error(f"Error exporting {format_type} report: {e}")
            print(f"[DEBUG] Error exporting {format_type} report: {e}")
    
    def _convert_to_dataframe(self, report_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert report data to DataFrame for CSV export."""
        try:
            records = []
            
            for trade in report_data.get("trade_records", []):
                record = {
                    "trade_id": trade.get("trade_id"),
                    "symbol": trade.get("symbol"),
                    "exchange": trade.get("exchange"),
                    "timestamp": trade.get("timestamp"),
                    "side": trade.get("trade_type", {}).get("side"),
                    "leverage": trade.get("trade_type", {}).get("leverage"),
                    "strategy": trade.get("trade_type", {}).get("strategy"),
                    "absolute_size": trade.get("position_sizing", {}).get("absolute_size"),
                    "portfolio_percentage": trade.get("position_sizing", {}).get("portfolio_percentage"),
                    "absolute_pnl": trade.get("pnl_analysis", {}).get("absolute_pnl"),
                    "percentage_pnl": trade.get("pnl_analysis", {}).get("percentage_pnl"),
                    "ensemble_confidence": trade.get("ml_confidence_scores", {}).get("ensemble_confidence"),
                    "market_health_score": trade.get("market_health", {}).get("overall_health_score"),
                    "trade_status": trade.get("trade_status"),
                }
                records.append(record)
            
            return pd.DataFrame(records)
            
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            return pd.DataFrame()
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        try:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Paper Trading Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
                    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Paper Trading Report</h1>
                    <p>Generated at: {generated_at}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Metrics</h2>
                    <div class="metric">Total Trades: {total_trades}</div>
                    <div class="metric">Total PnL: ${total_pnl:.2f}</div>
                    <div class="metric">Win Rate: {win_rate:.2%}</div>
                    <div class="metric">Sharpe Ratio: {sharpe_ratio:.2f}</div>
                </div>
                
                <div class="section">
                    <h2>Trade Records</h2>
                    <table>
                        <tr>
                            <th>Trade ID</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Size</th>
                            <th>PnL</th>
                            <th>Confidence</th>
                            <th>Status</th>
                        </tr>
                        {trade_rows}
                    </table>
                </div>
            </body>
            </html>
            """
            
            # Format trade rows
            trade_rows = ""
            for trade in report_data.get("trade_records", [])[:50]:  # Limit to first 50 trades
                trade_rows += f"""
                <tr>
                    <td>{trade.get('trade_id', '')}</td>
                    <td>{trade.get('symbol', '')}</td>
                    <td>{trade.get('trade_type', {}).get('side', '')}</td>
                    <td>{trade.get('position_sizing', {}).get('absolute_size', 0):.2f}</td>
                    <td>${trade.get('pnl_analysis', {}).get('net_pnl', 0):.2f}</td>
                    <td>{trade.get('ml_confidence_scores', {}).get('ensemble_confidence', 0):.2f}</td>
                    <td>{trade.get('trade_status', '')}</td>
                </tr>
                """
            
            # Format metrics
            metrics = report_data.get("performance_metrics", {})
            
            return html_content.format(
                generated_at=report_data.get("generated_at", ""),
                total_trades=metrics.get("total_trades", 0),
                total_pnl=metrics.get("total_pnl", 0.0),
                win_rate=metrics.get("win_rate", 0.0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                trade_rows=trade_rows,
            )
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return "<html><body><h1>Error generating report</h1></body></html>"
    
    def get_status(self) -> Dict[str, Any]:
        """Get reporter status."""
        return {
            "total_trades": len(self.trade_records),
            "report_directory": self.report_directory,
            "enable_detailed_reporting": self.enable_detailed_reporting,
            "last_updated": datetime.now().isoformat(),
        }


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="paper trading reporter setup",
)
async def setup_paper_trading_reporter(
    config: Dict[str, Any] | None = None,
) -> PaperTradingReporter | None:
    """
    Setup paper trading reporter.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PaperTradingReporter: Configured reporter instance
    """
    try:
        if config is None:
            config = {}
        
        reporter = PaperTradingReporter(config)
        
        # Create report directory
        os.makedirs(reporter.report_directory, exist_ok=True)
        
        return reporter
        
    except Exception as e:
        system_logger.error(f"Error setting up paper trading reporter: {e}")
        return None

# --- TEST UTILITY ---
def test_generate_report():
    import asyncio
    print("[DEBUG] Running test_generate_report()...")
    config = {"paper_trading_reporter": {"report_directory": "reports/paper_trading", "export_formats": ["json", "csv", "html"]}}
    reporter = PaperTradingReporter(config)
    # Add a fake trade record for testing
    from datetime import datetime
    trade_data = {
        "side": "BUY",
        "leverage": 1.0,
        "duration": "scalping",
        "strategy": "test_strategy",
        "order_type": "market",
        "quantity": 1.0,
        "portfolio_percentage": 0.1,
        "risk_percentage": 0.01,
        "max_position_size": 0.1,
        "position_ranking": 1,
        "absolute_pnl": 10.0,
        "percentage_pnl": 0.01,
        "unrealized_pnl": 0.0,
        "realized_pnl": 10.0,
        "total_cost": 100.0,
        "total_proceeds": 110.0,
        "commission": 0.1,
        "slippage": 0.05,
        "net_pnl": 9.85,
        "symbol": "TEST",
        "exchange": "TESTEX",
        "timestamp": datetime.now().isoformat(),
    }
    market_indicators = {"rsi": 50, "macd": 0, "macd_signal": 0, "bollinger_upper": 0, "bollinger_lower": 0, "bollinger_middle": 0, "atr": 0, "volume_sma": 0, "price_sma_20": 0, "price_sma_50": 0, "price_sma_200": 0, "volatility": 0, "momentum": 0, "support_level": 0, "resistance_level": 0}
    market_health = {"overall_health_score": 1.0, "volatility_regime": "normal", "liquidity_score": 1.0, "stress_score": 0.0, "market_strength": 1.0, "volume_health": "good", "price_trend": "up", "market_regime": "bull"}
    ml_confidence = {"analyst_confidence": 0.9, "tactician_confidence": 0.8, "ensemble_confidence": 0.85, "meta_learner_confidence": 0.8, "individual_model_confidences": {}, "ensemble_agreement": 0.9, "model_diversity": 0.1, "prediction_consistency": 0.95}
    async def run():
        await reporter.record_trade(trade_data, market_indicators, market_health, ml_confidence)
        await reporter.generate_detailed_report("test_report", ["json", "csv", "html"])
    asyncio.run(run())
    print("[DEBUG] test_generate_report() complete.")