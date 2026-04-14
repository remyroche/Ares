"""
Trade Logger for Inference.

This module handles CSV logging of trade decisions with detailed metrics:
- Log all trade decisions with full context
- Columns for audit trail
- Human-readable trade explanations
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
from dataclasses import dataclass, field

import pandas as pd

from extreme_price_movements.utils import tprint


# Default log directory
DEFAULT_LOG_DIR = "extreme_price_movements/logs"


# Expanded CSV columns for detailed trade logging
TRADE_LOG_COLUMNS = [
    # Core identifiers
    "timestamp", "run_id", "symbol", "side", "action", "mode",
    
    # Asset & market context
    "entry_price", "atr", "atr_frac", "volume", "vol_zscore",
    "ret24h", "range_12h_pct", "volatility_zscore",
    
    # Model predictions - Alpha (base) models
    "alpha_long_mr_pred", "alpha_long_tf_pred",
    "alpha_short_mr_pred", "alpha_short_tf_pred",
    
    # Model predictions - Meta model
    "meta_pred", "meta_confidence",
    
    # Model predictions - Ridge position sizer
    "ridge_position_size", "ridge_confidence",
    
    # Entry policy
    "place_order", "eu_star", "u_hat_z", "mae_hat_z", "mfe_hat_z",
    "limit_offset_bps", "sl_distance_atr", "tp_distance_atr",
    "trail_mult_eff", "giveback_pct_eff",
    
    # Regime features (for explaining why)
    "G_VOL", "G_TREND", "G_VOLUME",
    "vol_z", "trend_pct", "mkt_rv_ratio",
    
    # Candidate selection thresholds used
    "threshold_extreme_pct", "threshold_min_range", "threshold_min_vol_zscore",
    
    # Disagreement features
    "disagree_mr_std", "disagree_tf_std", "agree_tf_minus_mr",
    
    # OCO order details (live mode)
    "oco_id", "stop_price", "limit_price",
    
    # Aggtrades data (live mode)
    "aggtrades_count",
    
    # Status
    "status", "error"
]


@dataclass
class TradeLogger:
    """Logs trade decisions to CSV for audit trail with detailed metrics."""
    
    output_path: str = "inference_trades.csv"
    run_id: Optional[str] = None
    
    # Internal state
    _log_file: str = field(init=False, repr=False)
    _initialized: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Initialize the logger after dataclass initialization."""
        self.run_id = self.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directory exists
        log_dir = os.path.dirname(self.output_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        self._log_file = self.output_path
        
        # Initialize CSV file with headers
        if not os.path.exists(self._log_file):
            self._write_header()
        
        self._initialized = True
        tprint(f"TradeLogger initialized: {self._log_file}")
    
    @property
    def columns(self) -> List[str]:
        """Return the list of columns for trade logging."""
        return TRADE_LOG_COLUMNS
    
    def _write_header(self):
        """Write CSV header."""
        with open(self._log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS)
            writer.writeheader()
    
    def log_trade(
        self,
        decision: Dict[str, Any],
        model_results: Dict[str, Any],
        market_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Log a trade decision with full context.
        
        Args:
            decision: Output from model_orchestrator.run_full_chain()
            model_results: All model predictions and intermediate results
            market_data: Current market data (price, volume, ATR, etc.)
            config: Config used for this inference run
            
        Returns:
            The record that was written to the CSV
        """
        # Extract nested values with defaults
        alpha_preds = model_results.get("alpha_preds", {})
        entry_policy = decision.get("entry_policy", {})
        disagreement_features = model_results.get("disagreement_features", {})
        
        record = {
            # Core identifiers
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "run_id": config.get("run_id", self.run_id),
            "symbol": decision.get("symbol"),
            "side": decision.get("side"),
            "action": decision.get("action"),
            "mode": config.get("mode", "shadow"),
            
            # Asset & market context
            "entry_price": market_data.get("close"),
            "atr": market_data.get("atr"),
            "atr_frac": market_data.get("atr_frac"),
            "volume": market_data.get("volume"),
            "vol_zscore": market_data.get("vol_zscore"),
            "ret24h": market_data.get("ret24h"),
            "range_12h_pct": market_data.get("range_12h_pct"),
            "volatility_zscore": market_data.get("volatility_zscore"),
            
            # Alpha model predictions
            "alpha_long_mr_pred": alpha_preds.get("long_mr"),
            "alpha_long_tf_pred": alpha_preds.get("long_tf"),
            "alpha_short_mr_pred": alpha_preds.get("short_mr"),
            "alpha_short_tf_pred": alpha_preds.get("short_tf"),
            
            # Meta model predictions
            "meta_pred": model_results.get("meta_pred"),
            "meta_confidence": model_results.get("meta_confidence"),
            
            # Ridge position sizer
            "ridge_position_size": model_results.get("position_size"),
            "ridge_confidence": model_results.get("ridge_confidence"),
            
            # Entry policy
            "place_order": entry_policy.get("place_order"),
            "eu_star": entry_policy.get("eu_star"),
            "u_hat_z": entry_policy.get("u_hat_z"),
            "mae_hat_z": entry_policy.get("mae_hat_z"),
            "mfe_hat_z": entry_policy.get("mfe_hat_z"),
            "limit_offset_bps": entry_policy.get("limit_offset_bps_dynamic"),
            "sl_distance_atr": entry_policy.get("sl_distance_atr_eff"),
            "tp_distance_atr": entry_policy.get("tp_distance_atr_eff"),
            "trail_mult_eff": entry_policy.get("trail_mult_eff"),
            "giveback_pct_eff": entry_policy.get("giveback_pct_eff"),
            
            # Regime features
            "G_VOL": market_data.get("G_VOL"),
            "G_TREND": market_data.get("G_TREND"),
            "G_VOLUME": market_data.get("G_VOLUME"),
            "vol_z": market_data.get("vol_z"),
            "trend_pct": market_data.get("trend_pct"),
            "mkt_rv_ratio": market_data.get("mkt_rv_ratio"),
            
            # Candidate thresholds
            "threshold_extreme_pct": config.get("extreme_pct"),
            "threshold_min_range": config.get("min_range_pct"),
            "threshold_min_vol_zscore": config.get("min_vol_zscore"),
            
            # Disagreement features
            "disagree_mr_std": disagreement_features.get("disagree_mr_std"),
            "disagree_tf_std": disagreement_features.get("disagree_tf_std"),
            "agree_tf_minus_mr": disagreement_features.get("agree_tf_minus_mr_avg"),
            
            # OCO (live mode)
            "oco_id": decision.get("oco_id"),
            "stop_price": decision.get("stop_price"),
            "limit_price": decision.get("limit_price"),
            
            # Aggtrades (live mode)
            "aggtrades_count": len(decision.get("aggtrades", [])) if decision.get("aggtrades") else 0,
            
            # Status
            "status": decision.get("status", "completed"),
            "error": decision.get("error", "")
        }
        
        # Write to CSV
        with open(self._log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS)
            writer.writerow(record)
        
        # Generate and print explanation
        explanation = self.explain_trade(record)
        tprint(f"Logged trade: {record['action']} {record['side']} {record['symbol']} @ {record['entry_price']}")
        
        return record
    
    def explain_trade(self, record: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of why a trade was taken.
        
        Args:
            record: Trade record dictionary
            
        Returns:
            Human-readable explanation string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"TRADE EXPLANATION: {record['action'].upper()} {record['side'].upper()}")
        lines.append("=" * 60)
        
        # Core info
        lines.append(f"\n📊 Symbol: {record['symbol']}")
        lines.append(f"   Entry Price: ${record['entry_price']}")
        lines.append(f"   Mode: {record['mode']}")
        
        # Market context
        lines.append("\n📈 Market Context:")
        lines.append(f"   24h Return: {record.get('ret24h', 'N/A'):.2%}" if record.get('ret24h') else "   24h Return: N/A")
        lines.append(f"   12h Range: {record.get('range_12h_pct', 'N/A'):.2%}" if record.get('range_12h_pct') else "   12h Range: N/A")
        lines.append(f"   Volatility Z-Score: {record.get('volatility_zscore', 'N/A'):.2f}" if record.get('volatility_zscore') else "   Volatility Z-Score: N/A")
        lines.append(f"   Volume Z-Score: {record.get('vol_zscore', 'N/A'):.2f}" if record.get('vol_zscore') else "   Volume Z-Score: N/A")
        lines.append(f"   ATR: {record.get('atr', 'N/A'):.4f}" if record.get('atr') else "   ATR: N/A")
        
        # Model predictions
        lines.append("\n🤖 Model Predictions:")
        
        if record['side'] == 'long':
            lines.append(f"   Alpha Long MR:  {record.get('alpha_long_mr_pred', 'N/A'):.4f}" if record.get('alpha_long_mr_pred') is not None else "   Alpha Long MR:  N/A")
            lines.append(f"   Alpha Long TF:  {record.get('alpha_long_tf_pred', 'N/A'):.4f}" if record.get('alpha_long_tf_pred') is not None else "   Alpha Long TF:  N/A")
        else:
            lines.append(f"   Alpha Short MR: {record.get('alpha_short_mr_pred', 'N/A'):.4f}" if record.get('alpha_short_mr_pred') is not None else "   Alpha Short MR: N/A")
            lines.append(f"   Alpha Short TF: {record.get('alpha_short_tf_pred', 'N/A'):.4f}" if record.get('alpha_short_tf_pred') is not None else "   Alpha Short TF: N/A")
        
        lines.append(f"   Meta Prediction: {record.get('meta_pred', 'N/A'):.4f}" if record.get('meta_pred') is not None else "   Meta Prediction: N/A")
        lines.append(f"   Meta Confidence: {record.get('meta_confidence', 'N/A'):.2%}" if record.get('meta_confidence') else "   Meta Confidence: N/A")
        
        # Position sizing
        lines.append("\n💰 Position Sizing:")
        lines.append(f"   Ridge Position Size: {record.get('ridge_position_size', 'N/A'):.4f}" if record.get('ridge_position_size') else "   Ridge Position Size: N/A")
        lines.append(f"   Ridge Confidence: {record.get('ridge_confidence', 'N/A'):.2%}" if record.get('ridge_confidence') else "   Ridge Confidence: N/A")
        
        # Entry policy
        lines.append("\n🎯 Entry Policy:")
        place_order = record.get('place_order', False)
        lines.append(f"   Place Order: {'✅ YES' if place_order else '❌ NO'}")
        if place_order:
            lines.append(f"   EU* (Expected Utility): {record.get('eu_star', 'N/A'):.4f}" if record.get('eu_star') is not None else "   EU*: N/A")
            lines.append(f"   ũ (Predicted Return): {record.get('u_hat_z', 'N/A'):.4f}" if record.get('u_hat_z') is not None else "   ũ: N/A")
            lines.append(f"   MAÊ (Max Adverse): {record.get('mae_hat_z', 'N/A'):.4f}" if record.get('mae_hat_z') is not None else "   MAÊ: N/A")
            lines.append(f"   MFÊ (Max Favorable): {record.get('mfe_hat_z', 'N/A'):.4f}" if record.get('mfe_hat_z') is not None else "   MFÊ: N/A")
            lines.append(f"   Limit Offset (bps): {record.get('limit_offset_bps', 'N/A')}" if record.get('limit_offset_bps') else "   Limit Offset: N/A")
            lines.append(f"   SL Distance (ATR): {record.get('sl_distance_atr', 'N/A'):.2f}" if record.get('sl_distance_atr') else "   SL Distance: N/A")
            lines.append(f"   TP Distance (ATR): {record.get('tp_distance_atr', 'N/A'):.2f}" if record.get('tp_distance_atr') else "   TP Distance: N/A")
        
        # Regime context
        lines.append("\n🔄 Regime Features:")
        g_vol = record.get('G_VOL', 'N/A')
        g_trend = record.get('G_TREND', 'N/A')
        g_volume = record.get('G_VOLUME', 'N/A')
        lines.append(f"   Volatility Regime: {g_vol}")
        lines.append(f"   Trend Regime: {g_trend}")
        lines.append(f"   Liquidity Regime: {g_volume}")
        lines.append(f"   Vol Z-Score: {record.get('vol_z', 'N/A'):.2f}" if record.get('vol_z') is not None else "   Vol Z-Score: N/A")
        lines.append(f"   Trend %: {record.get('trend_pct', 'N/A'):.2%}" if record.get('trend_pct') is not None else "   Trend %: N/A")
        
        # Disagreement features
        lines.append("\n⚖️ Model Disagreement:")
        lines.append(f"   Disagree MR Std: {record.get('disagree_mr_std', 'N/A'):.4f}" if record.get('disagree_mr_std') is not None else "   Disagree MR Std: N/A")
        lines.append(f"   Disagree TF Std: {record.get('disagree_tf_std', 'N/A'):.4f}" if record.get('disagree_tf_std') is not None else "   Disagree TF Std: N/A")
        lines.append(f"   Agree TF - MR: {record.get('agree_tf_minus_mr', 'N/A'):.4f}" if record.get('agree_tf_minus_mr') is not None else "   Agree TF - MR: N/A")
        
        # Why trade was taken
        lines.append("\n💡 WHY THIS TRADE:")
        if place_order:
            reasons = []
            
            # Check meta prediction
            meta_pred = record.get('meta_pred')
            if meta_pred is not None:
                if record['side'] == 'long' and meta_pred > 0.5:
                    reasons.append("Strong long signal from meta model")
                elif record['side'] == 'short' and meta_pred > 0.5:
                    reasons.append("Strong short signal from meta model")
            
            # Check regime alignment
            if record.get('G_VOL') == 'HIGH':
                reasons.append("Trading in high volatility regime (favorable)")
            elif record.get('G_VOL') == 'LOW':
                reasons.append("Trading in low volatility regime")
            
            # Check trend alignment
            trend_pct = record.get('trend_pct')
            if trend_pct is not None:
                if record['side'] == 'long' and trend_pct > 0:
                    reasons.append("Long aligned with positive trend")
                elif record['side'] == 'short' and trend_pct < 0:
                    reasons.append("Short aligned with negative trend")
            
            # Check expected utility
            eu_star = record.get('eu_star')
            if eu_star is not None and eu_star > 0:
                reasons.append(f"Positive expected utility (EU*={eu_star:.4f})")
            
            # Check disagreement
            disagree_mr = record.get('disagree_mr_std')
            if disagree_mr is not None and disagree_mr < 0.1:
                reasons.append("Low disagreement among MR models (high confidence)")
            
            if reasons:
                for reason in reasons:
                    lines.append(f"   • {reason}")
            else:
                lines.append("   • Entry policy conditions met")
        else:
            lines.append("   • Entry policy conditions NOT met")
            eu_star = record.get('eu_star')
            if eu_star is not None:
                lines.append(f"   • EU* ({eu_star:.4f}) below threshold")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_log_path(self) -> str:
        """Get the path to the log file."""
        return self._log_file
    
    def read_logs(self) -> pd.DataFrame:
        """Read trade logs into DataFrame.
        
        Returns:
            DataFrame of trade logs
        """
        if not os.path.exists(self._log_file):
            return pd.DataFrame()
        
        return pd.read_csv(self._log_file)

    def get_last_trade_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """Return the latest logged trade timestamp for a symbol."""
        df = self.read_logs()
        if df.empty or "symbol" not in df.columns or "timestamp" not in df.columns:
            return None
        sym_df = df[df["symbol"] == symbol]
        if sym_df.empty:
            return None
        ts = pd.to_datetime(sym_df["timestamp"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return None
        return pd.Timestamp(ts.max())
    
    # =========================================================================
    # Legacy methods for backward compatibility
    # =========================================================================
    
    def log_trade_legacy(
        self,
        symbol: str,
        side: str,
        action: str,
        size: float,
        price: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "shadow",
        status: str = "pending",
        error: Optional[str] = None,
    ):
        """Legacy method for logging trade decisions.
        
        Args:
            symbol: Trading symbol
            side: "long" or "short"
            action: "enter" or "exit"
            size: Position size
            price: Entry/exit price
            context: Additional context (predictions, features, etc.)
            mode: "live" or "shadow"
            status: Trade status
            error: Error message if any
        """
        row = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "symbol": symbol,
            "side": side,
            "action": action,
            "size": size,
            "price": price,
            "mode": mode,
            "status": status,
            "error": error or "",
        }
        
        # Add context fields (map to new columns)
        if context:
            row["position_size"] = context.get("position_size", "")
            row["meta_pred"] = context.get("meta_mr_pred", "")
            row["alpha_long_mr_pred"] = context.get("alpha_mr_pred", "")
            row["alpha_long_tf_pred"] = context.get("alpha_tf_pred", "")
            row["disagree_mr_std"] = context.get("disagreement_mr", "")
            row["disagree_tf_std"] = context.get("disagreement_tf", "")
            row["ret24h"] = context.get("ret24h", "")
            row["range_12h_pct"] = context.get("range_12h_pct", "")
            row["volatility_zscore"] = context.get("volatility_zscore", "")
        
        # Write row
        with open(self._log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS)
            writer.writerow(row)
        
        tprint(f"Logged trade: {action} {side} {symbol} {size}@{price}")


def log_trade_decision(
    logger: TradeLogger,
    symbol: str,
    side: str,
    action: str,
    size: float,
    price: Optional[float],
    predictions: Optional[Dict[str, float]] = None,
    features: Optional[Dict[str, float]] = None,
    mode: str = "shadow",
) -> None:
    """Log a trade decision.
    
    Convenience function.
    
    Args:
        logger: TradeLogger instance
        symbol: Trading symbol
        side: "long" or "short"
        action: "enter" or "exit"
        size: Position size
        price: Price
        predictions: Model predictions
        features: Feature values
        mode: Execution mode
    """
    # Build decision dict for legacy logging
    decision = {
        "symbol": symbol,
        "side": side,
        "action": action,
        "status": "completed" if mode == "shadow" else "pending",
    }
    
    # Build model results
    model_results = {}
    if predictions:
        model_results = {
            "alpha_preds": {
                "long_mr": predictions.get("alpha_mr"),
                "long_tf": predictions.get("alpha_tf"),
                "short_mr": predictions.get("alpha_mr"),
                "short_tf": predictions.get("alpha_tf"),
            },
            "meta_pred": predictions.get("meta_mr"),
            "meta_confidence": predictions.get("meta_confidence"),
            "position_size": predictions.get("position_size"),
            "disagreement_features": {
                "disagree_mr_std": predictions.get("disagreement_mr"),
                "disagree_tf_std": predictions.get("disagreement_tf"),
            },
        }
    
    # Build market data
    market_data = features or {}
    market_data["close"] = price
    
    # Config
    config = {"mode": mode, "run_id": logger.run_id}
    
    logger.log_trade(decision, model_results, market_data, config)
