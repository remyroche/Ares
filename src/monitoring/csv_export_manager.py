#!/usr/bin/env python3
"""
CSV Export Manager for Enhanced ML Monitoring

Handles comprehensive CSV export functionality for monthly monitoring reports
with detailed trade decisions, model performance, and ensemble analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import time
import csv
from collections import defaultdict

from src.core.decorators import handles_errors, log_execution_time, traced
from src.utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
    safe_json_dump, safe_json_load, safe_file_exists,
    timed_operation, format_bytes, safe_log_metric, safe_log_params
)
from src.utils.logger import system_logger


@dataclass
class ExportConfig:
    """Configuration for CSV export."""
    export_directory: str = "monitoring_exports"
    include_raw_data: bool = True
    include_summary_stats: bool = True
    include_visualizations: bool = False
    compression: str = "none"  # "none", "gzip", "zip"
    max_rows_per_file: int = 100000
    date_format: str = "%Y-%m-%d %H:%M:%S"
    decimal_precision: int = 6


@dataclass
class ExportMetadata:
    """Metadata for exported files."""
    export_id: str
    timestamp: datetime
    file_type: str
    record_count: int
    file_size_bytes: int
    export_duration_ms: float
    compression_ratio: Optional[float] = None


class CSVExportManager:
    """
    Manages comprehensive CSV export functionality for monitoring data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the CSV export manager."""
        self.config = config
        self.logger = system_logger.getChild("CSVExportManager")
        
        # Configuration
        self.export_config = ExportConfig(**config.get("csv_export", {}))
        self.export_dir = Path(self.export_config.export_directory)
        self.export_dir.mkdir(exist_ok=True)
        
        # Export tracking
        self.export_history: List[ExportMetadata] = []
        self.last_export_time: Optional[datetime] = None
        
        self.logger.info("CSV Export Manager initialized")
    
    @handles_errors(default_return=False, context="csv_export_manager.export_trade_decisions")
    async def export_trade_decisions(self, trade_decisions: List[Any], 
                                   export_id: Optional[str] = None) -> bool:
        """Export trade decisions to CSV with comprehensive details."""
        try:
            start_time = time.time()
            export_id = export_id or f"trade_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if not trade_decisions:
                self.logger.warning("No trade decisions to export")
                return False
            
            # Create comprehensive DataFrame
            df = await self._create_trade_decisions_dataframe(trade_decisions)
            
            # Export main file
            main_file = await self._export_dataframe_to_csv(
                df, f"{export_id}_main.csv", "Trade Decisions Main Data"
            )
            
            # Export detailed breakdowns
            await self._export_trading_indicators_breakdown(trade_decisions, export_id)
            await self._export_ensemble_breakdown(trade_decisions, export_id)
            await self._export_model_breakdown(trade_decisions, export_id)
            await self._export_context_analysis(trade_decisions, export_id)
            
            # Create summary statistics
            if self.export_config.include_summary_stats:
                await self._export_trade_decisions_summary(df, export_id)
            
            # Record export metadata
            export_duration = (time.time() - start_time) * 1000
            metadata = ExportMetadata(
                export_id=export_id,
                timestamp=datetime.now(),
                file_type="trade_decisions",
                record_count=len(df),
                file_size_bytes=main_file.stat().st_size if main_file else 0,
                export_duration_ms=export_duration
            )
            self.export_history.append(metadata)
            
            self.logger.info(
                f"Exported {len(df)} trade decisions to {export_id} "
                f"in {export_duration:.1f}ms"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting trade decisions: {e}")
            return False
    
    async def _create_trade_decisions_dataframe(self, trade_decisions: List[Any]) -> pd.DataFrame:
        """Create comprehensive DataFrame from trade decisions."""
        data = []
        
        for decision in trade_decisions:
            # Base decision data
            row = {
                'decision_id': decision.decision_id,
                'timestamp': decision.timestamp.strftime(self.export_config.date_format),
                'trading_mode': decision.trading_mode.value,
                'exchange': decision.context.exchange,
                'token': decision.context.token,
                'price': round(decision.context.price, self.export_config.decimal_precision),
                'volume': round(decision.context.volume, self.export_config.decimal_precision),
                'timeframe': decision.context.timeframe,
                'regime': decision.context.regime,
                'action': decision.action,
                'position_size': round(decision.position_size, self.export_config.decimal_precision),
                'stop_loss': round(decision.stop_loss, self.export_config.decimal_precision) if decision.stop_loss else None,
                'take_profit': round(decision.take_profit, self.export_config.decimal_precision) if decision.take_profit else None,
                'overall_confidence': round(decision.overall_confidence, self.export_config.decimal_precision),
                'overall_risk_score': round(decision.overall_risk_score, self.export_config.decimal_precision),
                'execution_time_ms': round(decision.execution_time_ms, 2),
            }
            
            # Market conditions
            if decision.context.market_conditions:
                for key, value in decision.context.market_conditions.items():
                    if isinstance(value, (int, float)):
                        row[f'market_{key}'] = round(value, self.export_config.decimal_precision)
                    else:
                        row[f'market_{key}'] = str(value)
            
            # Ensemble decision data
            ensemble = decision.ensemble_decision
            row.update({
                'ensemble_id': ensemble.ensemble_id,
                'final_prediction': round(ensemble.final_prediction, self.export_config.decimal_precision),
                'final_confidence': round(ensemble.final_confidence, self.export_config.decimal_precision),
                'final_risk_score': round(ensemble.final_risk_score, self.export_config.decimal_precision),
                'voting_mechanism': ensemble.voting_mechanism,
                'consensus_score': round(ensemble.consensus_score, self.export_config.decimal_precision),
                'disagreement_level': round(ensemble.disagreement_level, self.export_config.decimal_precision),
            })
            
            # Model weights
            for model_id, weight in ensemble.model_weights.items():
                row[f'model_weight_{model_id}'] = round(weight, self.export_config.decimal_precision)
            
            # Trading indicators
            for i, indicator in enumerate(decision.trading_indicators):
                row[f'indicator_{i}_name'] = indicator.name
                row[f'indicator_{i}_value'] = round(indicator.value, self.export_config.decimal_precision)
                row[f'indicator_{i}_weight'] = round(indicator.weight, self.export_config.decimal_precision)
                row[f'indicator_{i}_confidence'] = round(indicator.confidence, self.export_config.decimal_precision)
                row[f'indicator_{i}_risk'] = round(indicator.risk_score, self.export_config.decimal_precision)
                row[f'indicator_{i}_description'] = indicator.description
            
            # Model decisions
            for i, model_decision in enumerate(ensemble.model_decisions):
                row[f'model_{i}_id'] = model_decision.model_id
                row[f'model_{i}_type'] = model_decision.model_type.value
                row[f'model_{i}_prediction'] = round(model_decision.prediction, self.export_config.decimal_precision)
                row[f'model_{i}_confidence'] = round(model_decision.confidence, self.export_config.decimal_precision)
                row[f'model_{i}_risk'] = round(model_decision.risk_score, self.export_config.decimal_precision)
                row[f'model_{i}_processing_time_ms'] = round(model_decision.processing_time_ms, 2)
                row[f'model_{i}_version'] = model_decision.model_version
            
            # Success metrics
            if decision.success_metrics:
                for key, value in decision.success_metrics.items():
                    if isinstance(value, (int, float)):
                        row[f'success_{key}'] = round(value, self.export_config.decimal_precision)
                    else:
                        row[f'success_{key}'] = str(value)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    async def _export_trading_indicators_breakdown(self, trade_decisions: List[Any], export_id: str):
        """Export detailed trading indicators breakdown."""
        try:
            indicators_data = []
            
            for decision in trade_decisions:
                for indicator in decision.trading_indicators:
                    indicators_data.append({
                        'decision_id': decision.decision_id,
                        'timestamp': decision.timestamp.strftime(self.export_config.date_format),
                        'token': decision.context.token,
                        'indicator_name': indicator.name,
                        'indicator_value': round(indicator.value, self.export_config.decimal_precision),
                        'indicator_weight': round(indicator.weight, self.export_config.decimal_precision),
                        'indicator_confidence': round(indicator.confidence, self.export_config.decimal_precision),
                        'indicator_risk': round(indicator.risk_score, self.export_config.decimal_precision),
                        'indicator_description': indicator.description,
                        'action': decision.action,
                        'overall_confidence': round(decision.overall_confidence, self.export_config.decimal_precision)
                    })
            
            if indicators_data:
                df = pd.DataFrame(indicators_data)
                await self._export_dataframe_to_csv(
                    df, f"{export_id}_trading_indicators.csv", "Trading Indicators Breakdown"
                )
                
        except Exception as e:
            self.logger.error(f"Error exporting trading indicators breakdown: {e}")
    
    async def _export_ensemble_breakdown(self, trade_decisions: List[Any], export_id: str):
        """Export detailed ensemble breakdown."""
        try:
            ensemble_data = []
            
            for decision in trade_decisions:
                ensemble = decision.ensemble_decision
                ensemble_data.append({
                    'decision_id': decision.decision_id,
                    'timestamp': decision.timestamp.strftime(self.export_config.date_format),
                    'token': decision.context.token,
                    'ensemble_id': ensemble.ensemble_id,
                    'final_prediction': round(ensemble.final_prediction, self.export_config.decimal_precision),
                    'final_confidence': round(ensemble.final_confidence, self.export_config.decimal_precision),
                    'final_risk_score': round(ensemble.final_risk_score, self.export_config.decimal_precision),
                    'voting_mechanism': ensemble.voting_mechanism,
                    'consensus_score': round(ensemble.consensus_score, self.export_config.decimal_precision),
                    'disagreement_level': round(ensemble.disagreement_level, self.export_config.decimal_precision),
                    'num_models': len(ensemble.model_decisions),
                    'action': decision.action,
                    'overall_confidence': round(decision.overall_confidence, self.export_config.decimal_precision)
                })
                
                # Model weights
                for model_id, weight in ensemble.model_weights.items():
                    ensemble_data[-1][f'weight_{model_id}'] = round(weight, self.export_config.decimal_precision)
            
            if ensemble_data:
                df = pd.DataFrame(ensemble_data)
                await self._export_dataframe_to_csv(
                    df, f"{export_id}_ensemble_breakdown.csv", "Ensemble Breakdown"
                )
                
        except Exception as e:
            self.logger.error(f"Error exporting ensemble breakdown: {e}")
    
    async def _export_model_breakdown(self, trade_decisions: List[Any], export_id: str):
        """Export detailed model breakdown."""
        try:
            model_data = []
            
            for decision in trade_decisions:
                ensemble = decision.ensemble_decision
                
                for model_decision in ensemble.model_decisions:
                    model_data.append({
                        'decision_id': decision.decision_id,
                        'timestamp': decision.timestamp.strftime(self.export_config.date_format),
                        'token': decision.context.token,
                        'model_id': model_decision.model_id,
                        'model_type': model_decision.model_type.value,
                        'model_prediction': round(model_decision.prediction, self.export_config.decimal_precision),
                        'model_confidence': round(model_decision.confidence, self.export_config.decimal_precision),
                        'model_risk': round(model_decision.risk_score, self.export_config.decimal_precision),
                        'model_processing_time_ms': round(model_decision.processing_time_ms, 2),
                        'model_version': model_decision.model_version,
                        'model_weight': round(ensemble.model_weights.get(model_decision.model_id, 0.0), self.export_config.decimal_precision),
                        'action': decision.action,
                        'overall_confidence': round(decision.overall_confidence, self.export_config.decimal_precision)
                    })
                    
                    # Feature importance
                    if model_decision.feature_importance:
                        for feature, importance in model_decision.feature_importance.items():
                            model_data[-1][f'feature_importance_{feature}'] = round(importance, self.export_config.decimal_precision)
            
            if model_data:
                df = pd.DataFrame(model_data)
                await self._export_dataframe_to_csv(
                    df, f"{export_id}_model_breakdown.csv", "Model Breakdown"
                )
                
        except Exception as e:
            self.logger.error(f"Error exporting model breakdown: {e}")
    
    async def _export_context_analysis(self, trade_decisions: List[Any], export_id: str):
        """Export context analysis."""
        try:
            context_data = []
            
            for decision in trade_decisions:
                context_data.append({
                    'decision_id': decision.decision_id,
                    'timestamp': decision.timestamp.strftime(self.export_config.date_format),
                    'exchange': decision.context.exchange,
                    'token': decision.context.token,
                    'price': round(decision.context.price, self.export_config.decimal_precision),
                    'volume': round(decision.context.volume, self.export_config.decimal_precision),
                    'timeframe': decision.context.timeframe,
                    'regime': decision.context.regime,
                    'action': decision.action,
                    'overall_confidence': round(decision.overall_confidence, self.export_config.decimal_precision),
                    'overall_risk_score': round(decision.overall_risk_score, self.export_config.decimal_precision)
                })
                
                # Market conditions
                if decision.context.market_conditions:
                    for key, value in decision.context.market_conditions.items():
                        if isinstance(value, (int, float)):
                            context_data[-1][f'market_{key}'] = round(value, self.export_config.decimal_precision)
                        else:
                            context_data[-1][f'market_{key}'] = str(value)
            
            if context_data:
                df = pd.DataFrame(context_data)
                await self._export_dataframe_to_csv(
                    df, f"{export_id}_context_analysis.csv", "Context Analysis"
                )
                
        except Exception as e:
            self.logger.error(f"Error exporting context analysis: {e}")
    
    async def _export_trade_decisions_summary(self, df: pd.DataFrame, export_id: str):
        """Export summary statistics for trade decisions."""
        try:
            summary_data = []
            
            # Overall summary
            summary_data.append({
                'metric': 'total_decisions',
                'value': len(df),
                'description': 'Total number of trade decisions'
            })
            
            # Trading mode distribution
            mode_counts = df['trading_mode'].value_counts()
            for mode, count in mode_counts.items():
                summary_data.append({
                    'metric': f'trading_mode_{mode}',
                    'value': count,
                    'description': f'Number of {mode} trading decisions'
                })
            
            # Action distribution
            action_counts = df['action'].value_counts()
            for action, count in action_counts.items():
                summary_data.append({
                    'metric': f'action_{action}',
                    'value': count,
                    'description': f'Number of {action} actions'
                })
            
            # Token distribution
            token_counts = df['token'].value_counts()
            for token, count in token_counts.head(10).items():  # Top 10 tokens
                summary_data.append({
                    'metric': f'token_{token}',
                    'value': count,
                    'description': f'Number of decisions for {token}'
                })
            
            # Performance metrics
            if 'overall_confidence' in df.columns:
                summary_data.extend([
                    {
                        'metric': 'avg_confidence',
                        'value': round(df['overall_confidence'].mean(), self.export_config.decimal_precision),
                        'description': 'Average overall confidence'
                    },
                    {
                        'metric': 'std_confidence',
                        'value': round(df['overall_confidence'].std(), self.export_config.decimal_precision),
                        'description': 'Standard deviation of confidence'
                    }
                ])
            
            if 'overall_risk_score' in df.columns:
                summary_data.extend([
                    {
                        'metric': 'avg_risk_score',
                        'value': round(df['overall_risk_score'].mean(), self.export_config.decimal_precision),
                        'description': 'Average risk score'
                    },
                    {
                        'metric': 'std_risk_score',
                        'value': round(df['overall_risk_score'].std(), self.export_config.decimal_precision),
                        'description': 'Standard deviation of risk score'
                    }
                ])
            
            # Time-based analysis
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.day_name()
                
                # Hourly distribution
                hourly_counts = df['hour'].value_counts().sort_index()
                for hour, count in hourly_counts.items():
                    summary_data.append({
                        'metric': f'hour_{hour:02d}',
                        'value': count,
                        'description': f'Number of decisions at hour {hour}'
                    })
                
                # Daily distribution
                daily_counts = df['day_of_week'].value_counts()
                for day, count in daily_counts.items():
                    summary_data.append({
                        'metric': f'day_{day}',
                        'value': count,
                        'description': f'Number of decisions on {day}'
                    })
            
            # Export summary
            summary_df = pd.DataFrame(summary_data)
            await self._export_dataframe_to_csv(
                summary_df, f"{export_id}_summary.csv", "Trade Decisions Summary"
            )
            
        except Exception as e:
            self.logger.error(f"Error exporting trade decisions summary: {e}")
    
    @handles_errors(default_return=False, context="csv_export_manager.export_model_performances")
    async def export_model_performances(self, model_performances: List[Any], 
                                      export_id: Optional[str] = None) -> bool:
        """Export model performance metrics to CSV."""
        try:
            start_time = time.time()
            export_id = export_id or f"model_performances_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if not model_performances:
                self.logger.warning("No model performances to export")
                return False
            
            # Create DataFrame
            data = []
            for perf in model_performances:
                row = {
                    'model_id': perf.model_id,
                    'model_type': perf.model_type.value,
                    'timestamp': perf.timestamp.strftime(self.export_config.date_format),
                    'accuracy': round(perf.accuracy, self.export_config.decimal_precision),
                    'precision': round(perf.precision, self.export_config.decimal_precision),
                    'recall': round(perf.recall, self.export_config.decimal_precision),
                    'f1_score': round(perf.f1_score, self.export_config.decimal_precision),
                    'auc_score': round(perf.auc_score, self.export_config.decimal_precision) if perf.auc_score else None,
                    'win_rate': round(perf.win_rate, self.export_config.decimal_precision),
                    'profit_factor': round(perf.profit_factor, self.export_config.decimal_precision),
                    'sharpe_ratio': round(perf.sharpe_ratio, self.export_config.decimal_precision),
                    'max_drawdown': round(perf.max_drawdown, self.export_config.decimal_precision),
                    'prediction_confidence_std': round(perf.prediction_confidence_std, self.export_config.decimal_precision),
                    'feature_importance_stability': round(perf.feature_importance_stability, self.export_config.decimal_precision),
                    'concept_drift_score': round(perf.concept_drift_score, self.export_config.decimal_precision),
                    'data_drift_score': round(perf.data_drift_score, self.export_config.decimal_precision),
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Export main file
            main_file = await self._export_dataframe_to_csv(
                df, f"{export_id}_main.csv", "Model Performance Metrics"
            )
            
            # Export summary statistics
            if self.export_config.include_summary_stats:
                await self._export_model_performance_summary(df, export_id)
            
            # Record export metadata
            export_duration = (time.time() - start_time) * 1000
            metadata = ExportMetadata(
                export_id=export_id,
                timestamp=datetime.now(),
                file_type="model_performances",
                record_count=len(df),
                file_size_bytes=main_file.stat().st_size if main_file else 0,
                export_duration_ms=export_duration
            )
            self.export_history.append(metadata)
            
            self.logger.info(
                f"Exported {len(df)} model performances to {export_id} "
                f"in {export_duration:.1f}ms"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting model performances: {e}")
            return False
    
    async def _export_model_performance_summary(self, df: pd.DataFrame, export_id: str):
        """Export model performance summary statistics."""
        try:
            summary_data = []
            
            # Overall statistics
            summary_data.append({
                'metric': 'total_models',
                'value': df['model_id'].nunique(),
                'description': 'Total number of unique models'
            })
            
            summary_data.append({
                'metric': 'total_performance_records',
                'value': len(df),
                'description': 'Total number of performance records'
            })
            
            # Model type distribution
            type_counts = df['model_type'].value_counts()
            for model_type, count in type_counts.items():
                summary_data.append({
                    'metric': f'model_type_{model_type}',
                    'value': count,
                    'description': f'Number of {model_type} model records'
                })
            
            # Performance metrics summary
            performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'win_rate', 'profit_factor', 'sharpe_ratio']
            
            for metric in performance_metrics:
                if metric in df.columns:
                    summary_data.extend([
                        {
                            'metric': f'{metric}_mean',
                            'value': round(df[metric].mean(), self.export_config.decimal_precision),
                            'description': f'Mean {metric}'
                        },
                        {
                            'metric': f'{metric}_std',
                            'value': round(df[metric].std(), self.export_config.decimal_precision),
                            'description': f'Standard deviation of {metric}'
                        },
                        {
                            'metric': f'{metric}_min',
                            'value': round(df[metric].min(), self.export_config.decimal_precision),
                            'description': f'Minimum {metric}'
                        },
                        {
                            'metric': f'{metric}_max',
                            'value': round(df[metric].max(), self.export_config.decimal_precision),
                            'description': f'Maximum {metric}'
                        }
                    ])
            
            # Export summary
            summary_df = pd.DataFrame(summary_data)
            await self._export_dataframe_to_csv(
                summary_df, f"{export_id}_summary.csv", "Model Performance Summary"
            )
            
        except Exception as e:
            self.logger.error(f"Error exporting model performance summary: {e}")
    
    @handles_errors(default_return=None, context="csv_export_manager._export_dataframe_to_csv")
    async def _export_dataframe_to_csv(self, df: pd.DataFrame, filename: str, 
                                     description: str) -> Optional[Path]:
        """Export DataFrame to CSV file with proper formatting."""
        try:
            file_path = self.export_dir / filename
            
            # Export with proper formatting
            df.to_csv(
                file_path,
                index=False,
                float_format=f'%.{self.export_config.decimal_precision}f',
                date_format=self.export_config.date_format,
                quoting=csv.QUOTE_NONNUMERIC
            )
            
            self.logger.debug(f"Exported {description}: {file_path} ({len(df)} rows)")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting DataFrame to CSV: {e}")
            return None
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get statistics about CSV exports."""
        if not self.export_history:
            return {'total_exports': 0}
        
        total_records = sum(metadata.record_count for metadata in self.export_history)
        total_size = sum(metadata.file_size_bytes for metadata in self.export_history)
        avg_duration = np.mean([metadata.export_duration_ms for metadata in self.export_history])
        
        return {
            'total_exports': len(self.export_history),
            'total_records_exported': total_records,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'average_export_duration_ms': round(avg_duration, 2),
            'last_export_time': self.export_history[-1].timestamp.isoformat() if self.export_history else None,
            'export_directory': str(self.export_dir)
        }