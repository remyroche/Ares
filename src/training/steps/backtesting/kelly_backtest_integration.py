"""
Kelly Backtesting Integration

Integrates dampened Kelly sizing with paper trading engine for backtesting.
Tracks trade outcomes, updates Kelly bins, enforces temporal integrity,
and generates calibration reports.

Key features:
- Realized R calculation after each trade
- Kelly bin updates with regime tracking
- Config version logging
- Calibration report generation (posterior vs actual)
- Embargo and purging enforcement
- Regime switch tracking
"""

import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error

# Import Kelly components
from src.trading.sizing.dampened_kelly_engine import DampenedKellyEngine, KellyResult
from src.trading.sizing.kelly_history_tracker import KellyHistoryTracker
from src.trading.sizing.portfolio_correlation_handler import PortfolioCorrelationHandler

logger = system_logger.getChild('KellyBacktestIntegration')


@dataclass
class TradeRecord:
    """Record of a trade for Kelly tracking."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    stop_loss_price: float
    take_profit_price: Optional[float]
    quantity: float
    side: str  # 'buy' or 'sell'
    
    # Kelly inputs at entry
    model_score: float
    volatility: float
    regime_id: Optional[int]
    ess: float
    entropy: float
    
    # Kelly outputs
    position_size: float
    leverage: float
    config_version: int
    reason_codes: List[str]
    
    # Outcome
    pnl: float
    is_win: bool
    r_realized: float  # Actual reward/risk ratio
    
    # Regime tracking
    regime_at_exit: Optional[int] = None
    regime_switched: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss_price': self.stop_loss_price,
            'quantity': self.quantity,
            'side': self.side,
            'model_score': self.model_score,
            'volatility': self.volatility,
            'regime_id': self.regime_id,
            'ess': self.ess,
            'entropy': self.entropy,
            'position_size': self.position_size,
            'leverage': self.leverage,
            'config_version': self.config_version,
            'reason_codes': self.reason_codes,
            'pnl': self.pnl,
            'is_win': self.is_win,
            'r_realized': self.r_realized,
            'regime_at_exit': self.regime_at_exit,
            'regime_switched': self.regime_switched
        }


@dataclass
class CalibrationBin:
    """Calibration data for a bin."""
    bin_key: str
    regime_id: Optional[int]
    score_range: Tuple[float, float]
    vol_range: Tuple[float, float]
    
    # Posterior predictions
    posterior_mean: float
    posterior_var: float
    
    # Actual outcomes
    n_trades: int
    n_wins: int
    actual_win_rate: float
    
    # Calibration metrics
    calibration_error: float  # |actual - predicted|
    is_well_calibrated: bool  # Error < 10%
    
    # R statistics
    r_predicted: float
    r_actual_mean: float
    r_actual_std: float


class KellyBacktestIntegration:
    """
    Integrates Kelly sizing with backtesting engine.
    
    Wraps the paper trading engine and adds:
    - Trade outcome tracking
    - Kelly bin updates
    - Realized R calculation
    - Config version logging
    - Calibration reporting
    - Temporal integrity (embargo/purging)
    """
    
    def __init__(
        self,
        kelly_engine: DampenedKellyEngine,
        kelly_tracker: KellyHistoryTracker,
        correlation_handler: Optional[PortfolioCorrelationHandler] = None,
        output_dir: str = "outcomes/kelly_backtest"
    ):
        """
        Initialize Kelly backtest integration.
        
        Args:
            kelly_engine: Dampened Kelly engine
            kelly_tracker: Kelly history tracker
            correlation_handler: Optional correlation handler
            output_dir: Directory for output artifacts
        """
        self.kelly_engine = kelly_engine
        self.kelly_tracker = kelly_tracker
        self.correlation_handler = correlation_handler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.getChild('Integration')
        
        # Trade tracking
        self.trade_records: List[TradeRecord] = []
        self.open_trades: Dict[str, Dict[str, Any]] = {}  # trade_id -> entry data
        
        # Regime tracking
        self.regime_switches: List[Tuple[datetime, Optional[int], Optional[int]]] = []
        self.current_regime: Optional[int] = None
        
        # Calibration tracking
        self.calibration_bins: Dict[str, CalibrationBin] = {}
        
        # Statistics
        self.total_trades = 0
        self.high_leverage_trades = 0
        self.bin_merge_count = {0: 0, 1: 0, 2: 0, 3: 0}  # Count by merge level
        
        tprint_info("✅ Kelly Backtest Integration initialized")
        self.logger.info("Kelly backtest integration initialized")
    
    @handles_errors
    def record_trade_entry(
        self,
        trade_id: str,
        symbol: str,
        entry_time: datetime,
        entry_price: float,
        stop_loss_price: float,
        quantity: float,
        side: str,
        model_score: float,
        volatility: float,
        regime_id: Optional[int],
        ess: float,
        entropy: float,
        kelly_result: KellyResult
    ) -> None:
        """
        Record trade entry with Kelly sizing inputs.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            entry_time: Entry timestamp
            entry_price: Entry price
            stop_loss_price: Stop loss price
            quantity: Position quantity
            side: 'buy' or 'sell'
            model_score: Model confidence score
            volatility: Market volatility
            regime_id: Regime at entry
            ess: Effective sample size
            entropy: Ensemble entropy
            kelly_result: Kelly sizing result
        """
        # Track regime switches
        if regime_id != self.current_regime:
            self.regime_switches.append((entry_time, self.current_regime, regime_id))
            if self.current_regime is not None:
                self.kelly_tracker.track_regime_switch(entry_time, self.current_regime, regime_id)
            self.current_regime = regime_id
        
        # Store entry data for later outcome recording
        self.open_trades[trade_id] = {
            'trade_id': trade_id,
            'symbol': symbol,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'quantity': quantity,
            'side': side,
            'model_score': model_score,
            'volatility': volatility,
            'regime_id': regime_id,
            'ess': ess,
            'entropy': entropy,
            'position_size': kelly_result.f_final,
            'leverage': kelly_result.leverage_final,
            'config_version': kelly_result.config_version,
            'reason_codes': kelly_result.reason_codes.copy(),
            'bin_merge_level': kelly_result.bin_merge_level
        }
        
        # Update statistics
        self.bin_merge_count[kelly_result.bin_merge_level] += 1
        
        if kelly_result.leverage_final >= self.kelly_engine.safety_limits.get('high_leverage_threshold', 2.0):
            self.high_leverage_trades += 1
        
        self.logger.debug(f"Recorded entry for trade {trade_id}: score={model_score:.3f}, regime={regime_id}")
    
    @handles_errors
    def record_trade_exit(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        pnl: float,
        current_regime: Optional[int] = None
    ) -> None:
        """
        Record trade exit and update Kelly bins.
        
        Args:
            trade_id: Trade identifier
            exit_time: Exit timestamp
            exit_price: Exit price
            pnl: Profit/loss
            current_regime: Regime at exit (for regime switch tracking)
        """
        if trade_id not in self.open_trades:
            self.logger.warning(f"Trade {trade_id} not found in open trades")
            return
        
        entry_data = self.open_trades.pop(trade_id)
        
        # Calculate realized R
        risk = abs(entry_data['entry_price'] - entry_data['stop_loss_price'])
        if risk > 0:
            r_realized = abs(pnl / (entry_data['quantity'] * risk))
        else:
            r_realized = 1.0  # Default if stop loss not set properly
        
        # Determine if win
        is_win = pnl > 0
        
        # Check for regime switch
        regime_at_exit = current_regime if current_regime is not None else entry_data['regime_id']
        regime_switched = regime_at_exit != entry_data['regime_id']
        
        # Create trade record
        trade_record = TradeRecord(
            trade_id=trade_id,
            symbol=entry_data['symbol'],
            entry_time=entry_data['entry_time'],
            exit_time=exit_time,
            entry_price=entry_data['entry_price'],
            exit_price=exit_price,
            stop_loss_price=entry_data['stop_loss_price'],
            take_profit_price=None,
            quantity=entry_data['quantity'],
            side=entry_data['side'],
            model_score=entry_data['model_score'],
            volatility=entry_data['volatility'],
            regime_id=entry_data['regime_id'],
            ess=entry_data['ess'],
            entropy=entry_data['entropy'],
            position_size=entry_data['position_size'],
            leverage=entry_data['leverage'],
            config_version=entry_data['config_version'],
            reason_codes=entry_data['reason_codes'],
            pnl=pnl,
            is_win=is_win,
            r_realized=r_realized,
            regime_at_exit=regime_at_exit,
            regime_switched=regime_switched
        )
        
        # Store trade record
        self.trade_records.append(trade_record)
        self.total_trades += 1
        
        # Update Kelly bins
        self.kelly_tracker.update_bin(
            score=entry_data['model_score'],
            volatility=entry_data['volatility'],
            regime_id=entry_data['regime_id'],
            is_win=is_win,
            r_realized=r_realized,
            timestamp=exit_time
        )
        
        self.logger.debug(f"Recorded exit for trade {trade_id}: PnL={pnl:.2f}, R={r_realized:.2f}, win={is_win}")
    
    def apply_embargo_and_purging(
        self,
        train_end: datetime,
        test_start: datetime,
        max_trade_duration_days: int = 7
    ) -> int:
        """
        Apply embargo and purging to enforce temporal integrity.
        
        Args:
            train_end: End of training period
            test_start: Start of test period
            max_trade_duration_days: Maximum expected trade duration
            
        Returns:
            Number of trades purged
        """
        purged = self.kelly_tracker.purge_overlapping_trades(
            train_end=train_end,
            test_start=test_start,
            max_trade_duration_days=max_trade_duration_days
        )
        
        if purged > 0:
            tprint_info(f"📊 Purged {purged} overlapping trades for temporal integrity")
        
        return purged
    
    def generate_calibration_report(self) -> Dict[str, Any]:
        """
        Generate calibration report comparing posterior predictions vs actual outcomes.
        
        Returns:
            Calibration report dictionary
        """
        tprint_info("📊 Generating Kelly calibration report...")
        
        # Group trades by bin
        bin_trades: Dict[str, List[TradeRecord]] = {}
        
        for trade in self.trade_records:
            # Determine bin
            score_bucket = self.kelly_tracker._digitize_score(trade.model_score)
            vol_bucket = self.kelly_tracker._digitize_volatility(trade.volatility)
            regime_key = self.kelly_tracker._get_regime_key(trade.regime_id)
            bin_key = self.kelly_tracker._get_bin_key(score_bucket, vol_bucket)
            full_key = f"{regime_key}/{bin_key}"
            
            if full_key not in bin_trades:
                bin_trades[full_key] = []
            bin_trades[full_key].append(trade)
        
        # Calculate calibration for each bin
        calibration_results = []
        total_error = 0.0
        well_calibrated_bins = 0
        total_bins_with_data = 0
        
        for full_key, trades in bin_trades.items():
            if len(trades) < 5:  # Minimum samples for calibration check
                continue
            
            total_bins_with_data += 1
            
            # Parse key
            regime_key, bin_key = full_key.split('/')
            
            # Get bin data
            regime_id_str = regime_key.replace('regime_', '')
            regime_id = int(regime_id_str) if regime_id_str.isdigit() else None
            
            # Get bin from tracker
            score_bucket = int(bin_key.split('_')[0].replace('s', ''))
            vol_bucket = int(bin_key.split('_')[1].replace('v', ''))
            
            # Calculate actual outcomes
            n_trades = len(trades)
            n_wins = sum(1 for t in trades if t.is_win)
            actual_win_rate = n_wins / n_trades
            
            # Get posterior prediction (use first trade's data as proxy)
            first_trade = trades[0]
            regime_key_str = self.kelly_tracker._get_regime_key(regime_id)
            if regime_key_str in self.kelly_tracker.bins and bin_key in self.kelly_tracker.bins[regime_key_str]:
                bin_data = self.kelly_tracker.bins[regime_key_str][bin_key]
                
                # Get regime params for prior
                params = self.kelly_engine.get_regime_params(regime_id)
                prior_alpha = params.get('prior_alpha', 30.0)
                
                # Calculate posterior
                posterior_mean, posterior_var = self.kelly_engine.compute_posterior_mean_var(
                    wins=bin_data.wins,
                    losses=bin_data.losses,
                    a=prior_alpha,
                    b=prior_alpha
                )
            else:
                # Use simple estimate if bin not found
                posterior_mean = n_wins / n_trades
                posterior_var = posterior_mean * (1 - posterior_mean) / n_trades
            
            # Calibration error
            calibration_error = abs(actual_win_rate - posterior_mean)
            is_well_calibrated = calibration_error < 0.10  # 10% threshold
            
            total_error += calibration_error
            if is_well_calibrated:
                well_calibrated_bins += 1
            
            # R statistics
            r_values = [t.r_realized for t in trades]
            r_actual_mean = np.mean(r_values)
            r_actual_std = np.std(r_values)
            
            # Score and vol ranges
            if score_bucket < len(self.kelly_tracker.score_bins):
                score_min = self.kelly_tracker.score_bins[score_bucket] if score_bucket > 0 else 0.0
                score_max = self.kelly_tracker.score_bins[score_bucket]
            else:
                score_min = self.kelly_tracker.score_bins[-1]
                score_max = 1.0
            
            if vol_bucket < len(self.kelly_tracker.volatility_bins):
                vol_min = self.kelly_tracker.volatility_bins[vol_bucket] if vol_bucket > 0 else 0.0
                vol_max = self.kelly_tracker.volatility_bins[vol_bucket]
            else:
                vol_min = self.kelly_tracker.volatility_bins[-1]
                vol_max = 0.10
            
            calibration_results.append({
                'bin_key': full_key,
                'regime_id': regime_id,
                'score_range': [score_min, score_max],
                'vol_range': [vol_min, vol_max],
                'posterior_mean': posterior_mean,
                'posterior_var': posterior_var,
                'n_trades': n_trades,
                'n_wins': n_wins,
                'actual_win_rate': actual_win_rate,
                'calibration_error': calibration_error,
                'is_well_calibrated': is_well_calibrated,
                'r_actual_mean': r_actual_mean,
                'r_actual_std': r_actual_std
            })
        
        # Overall calibration metrics
        mean_calibration_error = total_error / total_bins_with_data if total_bins_with_data > 0 else 0.0
        calibration_rate = well_calibrated_bins / total_bins_with_data if total_bins_with_data > 0 else 0.0
        
        # Regime switch analysis
        mid_trade_switches = sum(1 for t in self.trade_records if t.regime_switched)
        regime_stability_pct = 1.0 - (mid_trade_switches / self.total_trades) if self.total_trades > 0 else 1.0
        
        # Bin coverage analysis
        bin_coverage = self.kelly_tracker.get_bin_coverage_stats()
        
        # High-leverage analysis
        high_lev_trades = [t for t in self.trade_records if t.leverage >= 2.0]
        high_lev_win_rate = sum(1 for t in high_lev_trades if t.is_win) / len(high_lev_trades) if high_lev_trades else 0.0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_trades': self.total_trades,
                'total_bins_analyzed': total_bins_with_data,
                'mean_calibration_error': mean_calibration_error,
                'calibration_rate': calibration_rate,
                'well_calibrated_bins': well_calibrated_bins,
                'regime_stability_pct': regime_stability_pct,
                'mid_trade_regime_switches': mid_trade_switches,
                'high_leverage_trades': len(high_lev_trades),
                'high_leverage_win_rate': high_lev_win_rate
            },
            'bin_coverage': bin_coverage,
            'bin_merge_distribution': self.bin_merge_count,
            'calibration_by_bin': calibration_results,
            'regime_switches': [
                {'timestamp': ts.isoformat(), 'from': old_r, 'to': new_r}
                for ts, old_r, new_r in self.regime_switches
            ]
        }
        
        # Save report
        report_file = self.output_dir / f"kelly_calibration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        tprint_success(f"✅ Calibration report saved: {report_file}")
        self.logger.info(f"Calibration report saved to {report_file}")
        
        # Print summary
        tprint_info(f"\n📊 Calibration Summary:")
        tprint_info(f"  Total trades: {self.total_trades}")
        tprint_info(f"  Bins analyzed: {total_bins_with_data}")
        tprint_info(f"  Mean calibration error: {mean_calibration_error:.2%}")
        tprint_info(f"  Well-calibrated bins: {well_calibrated_bins}/{total_bins_with_data} ({calibration_rate:.1%})")
        tprint_info(f"  Regime stability: {regime_stability_pct:.1%}")
        tprint_info(f"  High-leverage win rate: {high_lev_win_rate:.1%}")
        
        return report
    
    def save_trade_records(self, filename: Optional[str] = None) -> Path:
        """
        Save all trade records to file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"kelly_trade_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        records = [t.to_dict() for t in self.trade_records]
        
        with open(filepath, 'w') as f:
            json.dump(records, f, indent=2)
        
        tprint_success(f"✅ Trade records saved: {filepath}")
        return filepath
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get integration statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_trades': self.total_trades,
            'open_trades': len(self.open_trades),
            'high_leverage_trades': self.high_leverage_trades,
            'regime_switches': len(self.regime_switches),
            'bin_merge_distribution': self.bin_merge_count,
            'current_regime': self.current_regime
        }

