"""
Analyst Multi-Layer Training Metrics

CSV-based reporting utility for comprehensive metrics tracking across all layers.
Provides consistent metrics reporting for:
- Layer 1 (Base Models): Raw predictive power & diversity
- Layer 2 (Meta Model): Calibration & error correction
- Layer 3 (Gate Model): Risk avoidance & tail protection

Metrics Categories:
- Calibration: ECE, Brier, MCE, Probability Distribution Skew, Log Loss, Prediction Std
- Stability: Feature importance shift (Tree), coefficient stability (linear)
- Trading: Win Rate, Profit Factor, Avg Trade Expectancy
- Risk: Sortino, MaxDrawdown, Calmar, Omega Ratio, Tail Ratio
- Predictive: AUC-ROC, Information Coefficient, IR, Directional Accuracy
- Activity: Turnover, per-regime performance
"""

from __future__ import annotations

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, 
    mean_absolute_error, accuracy_score, precision_score, recall_score
)

try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(*args, **kwargs): print(*args)
    def tprint_success(*args, **kwargs): print(*args)
    def tprint_warning(*args, **kwargs): print(*args)
    def tprint_error(*args, **kwargs): print(*args)


@dataclass
class CalibrationMetrics:
    """Calibration metrics for probability predictions."""
    ece: float = 0.0  # Expected Calibration Error
    brier_score: float = 0.0  # Brier Score
    mce: float = 0.0  # Maximum Calibration Error
    prob_distribution_skew: float = 0.0  # Probability Distribution Skew
    log_loss_value: float = 0.0  # Log Loss (Cross-Entropy)
    prediction_std: float = 0.0  # Prediction Standard Deviation


@dataclass
class StabilityMetrics:
    """Model stability metrics."""
    feature_importance_shift: float = 0.0  # Feature importance shift (tree models)
    coefficient_stability: float = 0.0  # Coefficient stability (linear models)
    top_5_features_consistent: bool = False  # Top 5 features match across folds


@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    win_rate: float = 0.0  # Wins / Total Trades
    profit_factor: float = 0.0  # Gross Profit / Gross Loss
    avg_trade_expectancy: float = 0.0  # (Win% * AvgWin) - (Loss% * AvgLoss)
    total_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics."""
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0


@dataclass
class PredictiveMetrics:
    """Predictive power metrics."""
    auc_roc: float = 0.0
    information_coefficient: float = 0.0  # IC - Spearman Correlation
    information_ratio: float = 0.0  # IR = Mean(IC) / StdDev(IC)
    directional_accuracy: float = 0.0  # MDA: % Sign(Prediction) == Sign(True_Return)


@dataclass
class ActivityMetrics:
    """Activity and regime-based metrics."""
    turnover_daily: float = 0.0
    perf_high_vol: float = 0.0
    perf_mid_vol: float = 0.0
    perf_low_vol: float = 0.0
    perf_trend_up: float = 0.0
    perf_trend_down: float = 0.0
    perf_trend_sideways: float = 0.0
    perf_high_volume: float = 0.0
    perf_low_volume: float = 0.0


@dataclass
class DiversityMetrics:
    """Diversity metrics for base models."""
    pairwise_correlation: float = 0.0  # Average pairwise correlation
    max_pairwise_correlation: float = 0.0
    min_pairwise_correlation: float = 0.0


@dataclass 
class GateMetrics:
    """Gate model specific metrics."""
    delta_max_drawdown: float = 0.0  # MDD reduction
    delta_sortino: float = 0.0  # Sortino improvement
    rejection_balance: float = 0.0  # Loss avoided by rejected losers / Gain missed
    gating_frequency: float = 0.0  # Active percentage of time


@dataclass
class LayerMetrics:
    """Complete metrics for a model layer."""
    model_name: str = ""
    layer: str = ""  # "L1_base", "L2_meta", "L3_gate"
    timestamp: str = ""
    symbol: str = ""
    exchange: str = ""
    timeframe: str = ""
    direction: str = ""
    model_type: str = ""  # "lgbm", "extratrees", "linear", "average"
    
    # Sub-metrics
    calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)
    trading: TradingMetrics = field(default_factory=TradingMetrics)
    risk: RiskMetrics = field(default_factory=RiskMetrics)
    predictive: PredictiveMetrics = field(default_factory=PredictiveMetrics)
    activity: ActivityMetrics = field(default_factory=ActivityMetrics)
    diversity: DiversityMetrics = field(default_factory=DiversityMetrics)
    gate: GateMetrics = field(default_factory=GateMetrics)
    
    # Additional metadata
    n_samples: int = 0
    n_features: int = 0
    training_duration_sec: float = 0.0
    notes: str = ""


class MultiLayerMetricsReporter:
    """
    CSV-based reporting utility for multi-layer model training.
    
    Provides consistent metrics reporting across all layers with:
    - Automatic CSV file management
    - Flattened metrics for easy analysis
    - Markdown report generation
    """
    
    def __init__(
        self,
        output_dir: str = "outcomes/multi_layer_training",
        csv_filename: str = "multi_layer_metrics.csv",
        append_mode: bool = True
    ):
        """
        Initialize the metrics reporter.
        
        Args:
            output_dir: Directory for output files
            csv_filename: Name of the CSV metrics file
            append_mode: If True, append to existing CSV; otherwise overwrite
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / csv_filename
        self.append_mode = append_mode
        self._initialized = False
        
    def _get_csv_headers(self) -> List[str]:
        """Get all CSV column headers."""
        return [
            # Identification
            "model_name", "layer", "timestamp", "symbol", "exchange", 
            "timeframe", "direction", "model_type",
            # Calibration
            "ece", "brier_score", "mce", "prob_distribution_skew",
            "log_loss_value", "prediction_std",
            # Stability
            "feature_importance_shift", "coefficient_stability",
            "top_5_features_consistent",
            # Trading
            "win_rate", "profit_factor", "avg_trade_expectancy",
            "total_trades", "gross_profit", "gross_loss",
            # Risk
            "sortino_ratio", "max_drawdown", "calmar_ratio",
            "omega_ratio", "tail_ratio",
            # Predictive
            "auc_roc", "information_coefficient", "information_ratio",
            "directional_accuracy",
            # Activity
            "turnover_daily", "perf_high_vol", "perf_mid_vol", "perf_low_vol",
            "perf_trend_up", "perf_trend_down", "perf_trend_sideways",
            "perf_high_volume", "perf_low_volume",
            # Diversity
            "pairwise_correlation", "max_pairwise_correlation",
            "min_pairwise_correlation",
            # Gate
            "delta_max_drawdown", "delta_sortino", "rejection_balance",
            "gating_frequency",
            # Metadata
            "n_samples", "n_features", "training_duration_sec", "notes"
        ]
    
    def _flatten_metrics(self, metrics: LayerMetrics) -> Dict[str, Any]:
        """Flatten LayerMetrics to a single dict for CSV."""
        row = {
            # Identification
            "model_name": metrics.model_name,
            "layer": metrics.layer,
            "timestamp": metrics.timestamp,
            "symbol": metrics.symbol,
            "exchange": metrics.exchange,
            "timeframe": metrics.timeframe,
            "direction": metrics.direction,
            "model_type": metrics.model_type,
            # Calibration
            "ece": metrics.calibration.ece,
            "brier_score": metrics.calibration.brier_score,
            "mce": metrics.calibration.mce,
            "prob_distribution_skew": metrics.calibration.prob_distribution_skew,
            "log_loss_value": metrics.calibration.log_loss_value,
            "prediction_std": metrics.calibration.prediction_std,
            # Stability
            "feature_importance_shift": metrics.stability.feature_importance_shift,
            "coefficient_stability": metrics.stability.coefficient_stability,
            "top_5_features_consistent": metrics.stability.top_5_features_consistent,
            # Trading
            "win_rate": metrics.trading.win_rate,
            "profit_factor": metrics.trading.profit_factor,
            "avg_trade_expectancy": metrics.trading.avg_trade_expectancy,
            "total_trades": metrics.trading.total_trades,
            "gross_profit": metrics.trading.gross_profit,
            "gross_loss": metrics.trading.gross_loss,
            # Risk
            "sortino_ratio": metrics.risk.sortino_ratio,
            "max_drawdown": metrics.risk.max_drawdown,
            "calmar_ratio": metrics.risk.calmar_ratio,
            "omega_ratio": metrics.risk.omega_ratio,
            "tail_ratio": metrics.risk.tail_ratio,
            # Predictive
            "auc_roc": metrics.predictive.auc_roc,
            "information_coefficient": metrics.predictive.information_coefficient,
            "information_ratio": metrics.predictive.information_ratio,
            "directional_accuracy": metrics.predictive.directional_accuracy,
            # Activity
            "turnover_daily": metrics.activity.turnover_daily,
            "perf_high_vol": metrics.activity.perf_high_vol,
            "perf_mid_vol": metrics.activity.perf_mid_vol,
            "perf_low_vol": metrics.activity.perf_low_vol,
            "perf_trend_up": metrics.activity.perf_trend_up,
            "perf_trend_down": metrics.activity.perf_trend_down,
            "perf_trend_sideways": metrics.activity.perf_trend_sideways,
            "perf_high_volume": metrics.activity.perf_high_volume,
            "perf_low_volume": metrics.activity.perf_low_volume,
            # Diversity
            "pairwise_correlation": metrics.diversity.pairwise_correlation,
            "max_pairwise_correlation": metrics.diversity.max_pairwise_correlation,
            "min_pairwise_correlation": metrics.diversity.min_pairwise_correlation,
            # Gate
            "delta_max_drawdown": metrics.gate.delta_max_drawdown,
            "delta_sortino": metrics.gate.delta_sortino,
            "rejection_balance": metrics.gate.rejection_balance,
            "gating_frequency": metrics.gate.gating_frequency,
            # Metadata
            "n_samples": metrics.n_samples,
            "n_features": metrics.n_features,
            "training_duration_sec": metrics.training_duration_sec,
            "notes": metrics.notes,
        }
        return row
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers if needed."""
        if not self._initialized:
            if not self.csv_path.exists() or not self.append_mode:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self._get_csv_headers())
                    writer.writeheader()
            self._initialized = True
    
    def record_metrics(self, metrics: LayerMetrics) -> None:
        """
        Record metrics to CSV file.
        
        Args:
            metrics: LayerMetrics object containing all metrics
        """
        self._initialize_csv()
        
        row = self._flatten_metrics(metrics)
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_csv_headers())
            writer.writerow(row)
        
        tprint_success(f"✅ Recorded metrics for {metrics.model_name} to {self.csv_path}")
    
    def load_metrics(self) -> pd.DataFrame:
        """Load all recorded metrics from CSV."""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=self._get_csv_headers())
    
    def generate_comparison_table(
        self,
        layer: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a comparison table for models.
        
        Args:
            layer: Filter by layer ("L1_base", "L2_meta", "L3_gate")
            symbol: Filter by symbol
            
        Returns:
            DataFrame with comparison metrics
        """
        df = self.load_metrics()
        
        if layer:
            df = df[df['layer'] == layer]
        if symbol:
            df = df[df['symbol'] == symbol]
        
        # Select key comparison columns
        key_cols = [
            'model_name', 'layer', 'model_type',
            'auc_roc', 'information_coefficient', 'brier_score', 'ece',
            'win_rate', 'profit_factor', 'sortino_ratio', 'max_drawdown'
        ]
        
        available_cols = [c for c in key_cols if c in df.columns]
        return df[available_cols].sort_values('auc_roc', ascending=False)


# =============================================================================
# Metrics Computation Functions
# =============================================================================

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> CalibrationMetrics:
    """
    Compute calibration metrics for probability predictions.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_prob: Predicted probabilities [0, 1]
        n_bins: Number of bins for ECE calculation
        
    Returns:
        CalibrationMetrics object
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    # Ensure probabilities are in [0, 1]
    y_prob = np.clip(y_prob, 1e-8, 1 - 1e-8)
    
    metrics = CalibrationMetrics()
    
    # Brier Score
    try:
        metrics.brier_score = float(brier_score_loss(y_true, y_prob))
    except Exception:
        metrics.brier_score = 0.0
    
    # Log Loss
    try:
        metrics.log_loss_value = float(log_loss(y_true, y_prob))
    except Exception:
        metrics.log_loss_value = 0.0
    
    # ECE (Expected Calibration Error)
    try:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])
        
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                avg_confidence = np.mean(y_prob[mask])
                avg_accuracy = np.mean(y_true[mask])
                bin_error = abs(avg_confidence - avg_accuracy)
                bin_weight = mask.sum() / len(y_prob)
                ece += bin_weight * bin_error
                mce = max(mce, bin_error)
        
        metrics.ece = float(ece)
        metrics.mce = float(mce)
    except Exception:
        metrics.ece = 0.0
        metrics.mce = 0.0
    
    # Probability Distribution Skew
    try:
        metrics.prob_distribution_skew = float(stats.skew(y_prob))
    except Exception:
        metrics.prob_distribution_skew = 0.0
    
    # Prediction Standard Deviation
    try:
        metrics.prediction_std = float(np.std(y_prob))
    except Exception:
        metrics.prediction_std = 0.0
    
    return metrics


def compute_predictive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: Optional[np.ndarray] = None
) -> PredictiveMetrics:
    """
    Compute predictive power metrics.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_prob: Predicted probabilities [0, 1]
        returns: Optional actual returns for IC calculation
        
    Returns:
        PredictiveMetrics object
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    metrics = PredictiveMetrics()
    
    # AUC-ROC
    try:
        if len(np.unique(y_true)) > 1:
            metrics.auc_roc = float(roc_auc_score(y_true, y_prob))
        else:
            metrics.auc_roc = 0.5
    except Exception:
        metrics.auc_roc = 0.5
    
    # Information Coefficient (Spearman Correlation)
    if returns is not None:
        returns = np.asarray(returns).ravel()
        try:
            corr, _ = stats.spearmanr(y_prob, returns)
            metrics.information_coefficient = float(corr) if np.isfinite(corr) else 0.0
        except Exception:
            metrics.information_coefficient = 0.0
    else:
        # Use binary labels if returns not provided
        try:
            corr, _ = stats.spearmanr(y_prob, y_true)
            metrics.information_coefficient = float(corr) if np.isfinite(corr) else 0.0
        except Exception:
            metrics.information_coefficient = 0.0
    
    # Directional Accuracy (MDA)
    try:
        pred_direction = (y_prob > 0.5).astype(int)
        metrics.directional_accuracy = float(np.mean(pred_direction == y_true))
    except Exception:
        metrics.directional_accuracy = 0.5
    
    return metrics


def compute_trading_metrics(
    predictions: np.ndarray,
    returns: np.ndarray,
    threshold: float = 0.5
) -> TradingMetrics:
    """
    Compute trading performance metrics.
    
    Args:
        predictions: Predicted probabilities [0, 1]
        returns: Actual returns per period
        threshold: Prediction threshold for taking trades
        
    Returns:
        TradingMetrics object
    """
    predictions = np.asarray(predictions).ravel()
    returns = np.asarray(returns).ravel()
    
    metrics = TradingMetrics()
    
    # Filter to trades taken (prediction >= threshold)
    trade_mask = predictions >= threshold
    trade_returns = returns[trade_mask]
    
    metrics.total_trades = int(np.sum(trade_mask))
    
    if metrics.total_trades == 0:
        return metrics
    
    # Win Rate
    wins = trade_returns > 0
    metrics.win_rate = float(np.mean(wins))
    
    # Gross Profit and Loss
    metrics.gross_profit = float(np.sum(trade_returns[trade_returns > 0]))
    metrics.gross_loss = float(abs(np.sum(trade_returns[trade_returns < 0])))
    
    # Profit Factor
    if metrics.gross_loss > 0:
        metrics.profit_factor = float(metrics.gross_profit / metrics.gross_loss)
    else:
        metrics.profit_factor = float('inf') if metrics.gross_profit > 0 else 0.0
    
    # Average Trade Expectancy
    avg_win = np.mean(trade_returns[wins]) if wins.sum() > 0 else 0
    avg_loss = np.mean(trade_returns[~wins]) if (~wins).sum() > 0 else 0
    loss_rate = 1 - metrics.win_rate
    metrics.avg_trade_expectancy = float(
        metrics.win_rate * avg_win + loss_rate * avg_loss
    )
    
    return metrics


def compute_risk_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24  # Assuming hourly data
) -> RiskMetrics:
    """
    Compute risk-adjusted performance metrics.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        RiskMetrics object
    """
    returns = np.asarray(returns).ravel()
    returns = returns[np.isfinite(returns)]
    
    metrics = RiskMetrics()
    
    if len(returns) == 0:
        return metrics
    
    # Sortino Ratio
    try:
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        mean_excess = np.mean(excess_returns) * periods_per_year
        metrics.sortino_ratio = float(mean_excess / downside_std) if downside_std > 0 else 0.0
    except Exception:
        metrics.sortino_ratio = 0.0
    
    # Max Drawdown
    try:
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        metrics.max_drawdown = float(abs(np.min(drawdowns)))
    except Exception:
        metrics.max_drawdown = 0.0
    
    # Calmar Ratio
    try:
        annual_return = np.mean(returns) * periods_per_year
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = float(annual_return / metrics.max_drawdown)
        else:
            metrics.calmar_ratio = 0.0
    except Exception:
        metrics.calmar_ratio = 0.0
    
    # Omega Ratio
    try:
        threshold = 0
        positive = returns[returns > threshold]
        negative = returns[returns <= threshold]
        if len(negative) > 0 and abs(np.sum(negative)) > 0:
            metrics.omega_ratio = float(np.sum(positive) / abs(np.sum(negative)))
        else:
            metrics.omega_ratio = float('inf') if np.sum(positive) > 0 else 0.0
    except Exception:
        metrics.omega_ratio = 0.0
    
    # Tail Ratio (95th percentile gain / 5th percentile loss)
    try:
        if len(returns) >= 20:
            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)
            if abs(p5) > 0:
                metrics.tail_ratio = float(p95 / abs(p5))
            else:
                metrics.tail_ratio = float('inf') if p95 > 0 else 0.0
        else:
            metrics.tail_ratio = 1.0
    except Exception:
        metrics.tail_ratio = 1.0
    
    return metrics


def compute_diversity_metrics(
    predictions_matrix: np.ndarray
) -> DiversityMetrics:
    """
    Compute diversity metrics for multiple base model predictions.
    
    Args:
        predictions_matrix: Matrix of shape (n_samples, n_models)
        
    Returns:
        DiversityMetrics object
    """
    predictions_matrix = np.asarray(predictions_matrix)
    
    metrics = DiversityMetrics()
    
    if predictions_matrix.ndim != 2 or predictions_matrix.shape[1] < 2:
        return metrics
    
    # Compute pairwise correlations
    try:
        n_models = predictions_matrix.shape[1]
        correlations = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr, _ = stats.pearsonr(
                    predictions_matrix[:, i],
                    predictions_matrix[:, j]
                )
                if np.isfinite(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            metrics.pairwise_correlation = float(np.mean(correlations))
            metrics.max_pairwise_correlation = float(np.max(correlations))
            metrics.min_pairwise_correlation = float(np.min(correlations))
    except Exception:
        pass
    
    return metrics


def compute_gate_metrics(
    predictions_unfiltered: np.ndarray,
    predictions_filtered: np.ndarray,
    returns: np.ndarray,
    gate_decisions: np.ndarray
) -> GateMetrics:
    """
    Compute gate model specific metrics.
    
    Args:
        predictions_unfiltered: Predictions without gating
        predictions_filtered: Predictions with gating applied
        returns: Actual returns
        gate_decisions: Binary gate decisions (1 = trade allowed)
        
    Returns:
        GateMetrics object
    """
    predictions_unfiltered = np.asarray(predictions_unfiltered).ravel()
    predictions_filtered = np.asarray(predictions_filtered).ravel()
    returns = np.asarray(returns).ravel()
    gate_decisions = np.asarray(gate_decisions).ravel()
    
    metrics = GateMetrics()
    
    # Gating Frequency
    metrics.gating_frequency = float(1 - np.mean(gate_decisions))
    
    # Compute risk metrics before and after gating
    try:
        risk_before = compute_risk_metrics(returns)
        
        # Apply gate filter to returns
        gated_returns = returns.copy()
        gated_returns[gate_decisions == 0] = 0  # No trade = no return
        risk_after = compute_risk_metrics(gated_returns[gated_returns != 0] if (gated_returns != 0).any() else np.array([0]))
        
        # Delta Max Drawdown (positive = improvement)
        metrics.delta_max_drawdown = float(risk_before.max_drawdown - risk_after.max_drawdown)
        
        # Delta Sortino (positive = improvement)
        metrics.delta_sortino = float(risk_after.sortino_ratio - risk_before.sortino_ratio)
    except Exception:
        pass
    
    # Rejection Balance
    try:
        rejected_mask = gate_decisions == 0
        rejected_returns = returns[rejected_mask]
        
        if len(rejected_returns) > 0:
            # Loss avoided by rejected losers
            loss_avoided = abs(np.sum(rejected_returns[rejected_returns < 0]))
            # Gain missed by rejected winners  
            gain_missed = np.sum(rejected_returns[rejected_returns > 0])
            
            if gain_missed > 0:
                metrics.rejection_balance = float(loss_avoided / gain_missed)
            else:
                metrics.rejection_balance = float('inf') if loss_avoided > 0 else 0.0
    except Exception:
        pass
    
    return metrics


# =============================================================================
# Markdown Report Generation
# =============================================================================

def generate_layer_markdown_report(
    metrics: LayerMetrics,
    output_path: Optional[str] = None,
    thresholds: Optional[Dict[str, Dict[str, float]]] = None
) -> str:
    """
    Generate a markdown report for a single layer.
    
    Args:
        metrics: LayerMetrics object
        output_path: Optional path to save the report
        thresholds: Optional thresholds for success criteria
        
    Returns:
        Markdown content as string
    """
    # Default thresholds based on the specification
    default_thresholds = {
        "L1_base": {
            "ic_min": 0.03,
            "sortino_min": 1.5,
            "expectancy_min": 0.001,
            "pairwise_corr_max": 0.75,
            "pred_std_min": 0.05
        },
        "L2_meta": {
            "ic_improvement_pct": 0.20,
            "ece_max": 0.05,
            "profit_factor_min": 1.8
        },
        "L3_gate": {
            "delta_mdd_min": 0.20,
            "delta_sortino_min": 0.5,
            "rejection_balance_min": 1.2,
            "gating_freq_min": 0.05,
            "gating_freq_max": 0.30
        }
    }
    
    thresholds = thresholds or default_thresholds
    layer_thresholds = thresholds.get(metrics.layer, {})
    
    # Build markdown content
    md = []
    md.append(f"# {metrics.layer.upper()} - {metrics.model_name}\n")
    md.append(f"**Generated**: {metrics.timestamp}\n")
    md.append(f"**Config**: {metrics.symbol} | {metrics.exchange} | {metrics.timeframe} | {metrics.direction}\n")
    md.append(f"**Model Type**: {metrics.model_type}\n")
    md.append(f"**Samples**: {metrics.n_samples:,} | **Features**: {metrics.n_features}\n")
    md.append("")
    
    # Layer-specific sections
    if metrics.layer == "L1_base":
        md.append("## Layer 1 (Base): Raw Predictive Power & Diversity\n")
        md.append("| Metric | Value | Threshold | Status |")
        md.append("|--------|-------|-----------|--------|")
        
        ic_status = "✅" if metrics.predictive.information_coefficient > layer_thresholds.get("ic_min", 0.03) else "❌"
        md.append(f"| Information Coefficient (IC) | {metrics.predictive.information_coefficient:.4f} | > {layer_thresholds.get('ic_min', 0.03)} | {ic_status} |")
        
        sortino_status = "✅" if metrics.risk.sortino_ratio > layer_thresholds.get("sortino_min", 1.5) else "❌"
        md.append(f"| Sortino Ratio | {metrics.risk.sortino_ratio:.4f} | > {layer_thresholds.get('sortino_min', 1.5)} | {sortino_status} |")
        
        exp_status = "✅" if metrics.trading.avg_trade_expectancy > layer_thresholds.get("expectancy_min", 0.001) else "❌"
        md.append(f"| Avg Expectancy | {metrics.trading.avg_trade_expectancy:.6f} | > {layer_thresholds.get('expectancy_min', 0.001)} | {exp_status} |")
        
        corr_status = "✅" if metrics.diversity.pairwise_correlation < layer_thresholds.get("pairwise_corr_max", 0.75) else "❌"
        md.append(f"| Pairwise Correlation | {metrics.diversity.pairwise_correlation:.4f} | < {layer_thresholds.get('pairwise_corr_max', 0.75)} | {corr_status} |")
        
        std_status = "✅" if metrics.calibration.prediction_std > layer_thresholds.get("pred_std_min", 0.05) else "❌"
        md.append(f"| Prediction Std | {metrics.calibration.prediction_std:.4f} | > {layer_thresholds.get('pred_std_min', 0.05)} | {std_status} |")
        
    elif metrics.layer == "L2_meta":
        md.append("## Layer 2 (Meta): Calibration & Error Correction\n")
        md.append("| Metric | Value | Threshold | Status |")
        md.append("|--------|-------|-----------|--------|")
        
        ece_status = "✅" if metrics.calibration.ece < layer_thresholds.get("ece_max", 0.05) else "❌"
        md.append(f"| Expected Calibration Error (ECE) | {metrics.calibration.ece:.4f} | < {layer_thresholds.get('ece_max', 0.05)} | {ece_status} |")
        
        md.append(f"| Brier Score | {metrics.calibration.brier_score:.4f} | - | - |")
        
        pf_status = "✅" if metrics.trading.profit_factor > layer_thresholds.get("profit_factor_min", 1.8) else "❌"
        md.append(f"| Profit Factor | {metrics.trading.profit_factor:.4f} | > {layer_thresholds.get('profit_factor_min', 1.8)} | {pf_status} |")
        
        md.append(f"| AUC-ROC | {metrics.predictive.auc_roc:.4f} | - | - |")
        md.append(f"| IC | {metrics.predictive.information_coefficient:.4f} | - | - |")
        
    elif metrics.layer == "L3_gate":
        md.append("## Layer 3 (Gate): Risk Avoidance & Tail Protection\n")
        md.append("| Metric | Value | Threshold | Status |")
        md.append("|--------|-------|-----------|--------|")
        
        mdd_status = "✅" if metrics.gate.delta_max_drawdown > layer_thresholds.get("delta_mdd_min", 0.20) else "❌"
        md.append(f"| Delta Max Drawdown | {metrics.gate.delta_max_drawdown:.4f} | > {layer_thresholds.get('delta_mdd_min', 0.20)} | {mdd_status} |")
        
        sortino_status = "✅" if metrics.gate.delta_sortino > layer_thresholds.get("delta_sortino_min", 0.5) else "❌"
        md.append(f"| Delta Sortino | {metrics.gate.delta_sortino:.4f} | > {layer_thresholds.get('delta_sortino_min', 0.5)} | {sortino_status} |")
        
        rb_status = "✅" if metrics.gate.rejection_balance > layer_thresholds.get("rejection_balance_min", 1.2) else "❌"
        md.append(f"| Rejection Balance | {metrics.gate.rejection_balance:.4f} | > {layer_thresholds.get('rejection_balance_min', 1.2)} | {rb_status} |")
        
        freq_min = layer_thresholds.get("gating_freq_min", 0.05)
        freq_max = layer_thresholds.get("gating_freq_max", 0.30)
        freq_status = "✅" if freq_min <= metrics.gate.gating_frequency <= freq_max else "❌"
        md.append(f"| Gating Frequency | {metrics.gate.gating_frequency:.4f} | [{freq_min}, {freq_max}] | {freq_status} |")
    
    md.append("")
    
    # Additional metrics section
    md.append("## Additional Metrics\n")
    md.append("### Calibration")
    md.append(f"- ECE: {metrics.calibration.ece:.4f}")
    md.append(f"- Brier Score: {metrics.calibration.brier_score:.4f}")
    md.append(f"- MCE: {metrics.calibration.mce:.4f}")
    md.append(f"- Log Loss: {metrics.calibration.log_loss_value:.4f}")
    md.append(f"- Prediction Std: {metrics.calibration.prediction_std:.4f}")
    md.append("")
    
    md.append("### Trading Performance")
    md.append(f"- Win Rate: {metrics.trading.win_rate:.4f}")
    md.append(f"- Profit Factor: {metrics.trading.profit_factor:.4f}")
    md.append(f"- Avg Expectancy: {metrics.trading.avg_trade_expectancy:.6f}")
    md.append(f"- Total Trades: {metrics.trading.total_trades}")
    md.append("")
    
    md.append("### Risk Metrics")
    md.append(f"- Sortino Ratio: {metrics.risk.sortino_ratio:.4f}")
    md.append(f"- Max Drawdown: {metrics.risk.max_drawdown:.4f}")
    md.append(f"- Calmar Ratio: {metrics.risk.calmar_ratio:.4f}")
    md.append(f"- Omega Ratio: {metrics.risk.omega_ratio:.4f}")
    md.append(f"- Tail Ratio: {metrics.risk.tail_ratio:.4f}")
    md.append("")
    
    md.append("### Predictive Power")
    md.append(f"- AUC-ROC: {metrics.predictive.auc_roc:.4f}")
    md.append(f"- Information Coefficient: {metrics.predictive.information_coefficient:.4f}")
    md.append(f"- Information Ratio: {metrics.predictive.information_ratio:.4f}")
    md.append(f"- Directional Accuracy: {metrics.predictive.directional_accuracy:.4f}")
    md.append("")
    
    if metrics.notes:
        md.append("## Notes")
        md.append(metrics.notes)
        md.append("")
    
    content = "\n".join(md)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        tprint_success(f"✅ Saved markdown report to {output_path}")
    
    return content


def generate_multi_layer_summary_report(
    reporter: MultiLayerMetricsReporter,
    output_path: Optional[str] = None,
    symbol: Optional[str] = None
) -> str:
    """
    Generate a comprehensive summary report for all layers.
    
    Args:
        reporter: MultiLayerMetricsReporter instance
        output_path: Optional path to save the report
        symbol: Optional symbol to filter by
        
    Returns:
        Markdown content as string
    """
    df = reporter.load_metrics()
    
    if symbol:
        df = df[df['symbol'] == symbol]
    
    if df.empty:
        return "# Multi-Layer Training Summary\n\nNo metrics recorded yet."
    
    md = []
    md.append("# Multi-Layer Training Summary\n")
    md.append(f"**Generated**: {datetime.now().isoformat()}\n")
    if symbol:
        md.append(f"**Symbol**: {symbol}\n")
    md.append("")
    
    # Layer 1 Summary
    l1 = df[df['layer'] == 'L1_base']
    if not l1.empty:
        md.append("## Layer 1 (Base Models): Raw Predictive Power & Diversity\n")
        md.append("| Model | IC | Sortino | Expectancy | Pairwise Corr | AUC |")
        md.append("|-------|-----|---------|------------|---------------|-----|")
        for _, row in l1.iterrows():
            md.append(
                f"| {row['model_name']} | {row['information_coefficient']:.4f} | "
                f"{row['sortino_ratio']:.4f} | {row['avg_trade_expectancy']:.6f} | "
                f"{row['pairwise_correlation']:.4f} | {row['auc_roc']:.4f} |"
            )
        md.append("")
    
    # Layer 2 Summary
    l2 = df[df['layer'] == 'L2_meta']
    if not l2.empty:
        md.append("## Layer 2 (Meta Model): Calibration & Error Correction\n")
        md.append("| Model | ECE | Brier | Profit Factor | IC | AUC |")
        md.append("|-------|-----|-------|---------------|-----|-----|")
        for _, row in l2.iterrows():
            md.append(
                f"| {row['model_name']} | {row['ece']:.4f} | "
                f"{row['brier_score']:.4f} | {row['profit_factor']:.4f} | "
                f"{row['information_coefficient']:.4f} | {row['auc_roc']:.4f} |"
            )
        md.append("")
    
    # Layer 3 Summary
    l3 = df[df['layer'] == 'L3_gate']
    if not l3.empty:
        md.append("## Layer 3 (Gate Model): Risk Avoidance & Tail Protection\n")
        md.append("| Model | Delta MDD | Delta Sortino | Rejection Balance | Gating Freq |")
        md.append("|-------|-----------|---------------|-------------------|-------------|")
        for _, row in l3.iterrows():
            md.append(
                f"| {row['model_name']} | {row['delta_max_drawdown']:.4f} | "
                f"{row['delta_sortino']:.4f} | {row['rejection_balance']:.4f} | "
                f"{row['gating_frequency']:.4f} |"
            )
        md.append("")
    
    # Best Combination
    if not l2.empty:
        best_meta = l2.loc[l2['auc_roc'].idxmax()]
        md.append("## Best Combination\n")
        md.append(f"**Best Meta Model**: {best_meta['model_name']} (AUC: {best_meta['auc_roc']:.4f})\n")
        
        if not l3.empty:
            best_gate = l3.loc[l3['delta_max_drawdown'].idxmax()]
            md.append(f"**Best Gate Model**: {best_gate['model_name']} (Delta MDD: {best_gate['delta_max_drawdown']:.4f})\n")
    
    content = "\n".join(md)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        tprint_success(f"✅ Saved multi-layer summary to {output_path}")
    
    return content
