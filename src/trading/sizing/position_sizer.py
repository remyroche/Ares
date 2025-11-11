"""
Position Sizer - Dampened Kelly Integration

Production-hardened position sizing using dampened, posterior-aware Kelly criterion
with regime conditioning, adaptive binning, and comprehensive risk management.

Replaces basic Kelly with:
- Bayesian posterior estimation (Beta distribution)
- Ensemble uncertainty integration (ESS, entropy)
- Regime-aware parameters
- Adaptive bin merging fallback
- Realized R tracking
- Correlation-adjusted sizing
- Drawdown dampening
"""

import logging
import math
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug
from src.utils.tprint import tprint
from ..config.trading_config import TradingConfig
from .leverage_manager import LeverageManager
from .risk_calculator import RiskCalculator

# Import dampened Kelly components
from .dampened_kelly_engine import DampenedKellyEngine, KellyResult
from .kelly_history_tracker import KellyHistoryTracker
from .portfolio_correlation_handler import PortfolioCorrelationHandler

logger = system_logger.getChild('PositionSizer')


@dataclass
class PositionSizeResult:
    """Enhanced position sizing result with Kelly metadata."""
    symbol: str
    recommended_size: float
    max_size: float
    min_size: float
    leverage: float
    confidence: float
    kelly_size: float
    ml_size: float
    sizing_method: str
    metadata: Dict[str, Any]


class PositionSizer:
    """
    Production-hardened position sizing using dampened Kelly criterion.
    
    Features:
    - Unified dampened Kelly for position sizing and leverage
    - Regime-aware parameters
    - Adaptive bin merging (handles sparse data)
    - Realized R tracking (uses actual outcomes)
    - Correlation-adjusted sizing
    - Drawdown dampening
    - Hot-swappable safety limits
    """
    
    def __init__(
        self,
        config: TradingConfig,
        leverage_manager: Optional[LeverageManager] = None,
        risk_calculator: Optional[RiskCalculator] = None,
        kelly_config_path: Optional[str] = None
    ):
        """
        Initialize position sizer with dampened Kelly engine.
        
        Args:
            config: Trading configuration
            leverage_manager: Optional leverage manager
            risk_calculator: Optional risk calculator
            kelly_config_path: Path to Kelly sizing config (defaults to src/config/kelly_sizing_config.yaml)
        """
        self.config = config
        self.logger = logger.getChild('PositionSizer')
        
        # External dependencies (optional, can be set later)
        self.leverage_manager = leverage_manager
        self.risk_calculator = risk_calculator
        
        # Load Kelly configuration
        if kelly_config_path is None:
            kelly_config_path = "src/config/kelly_sizing_config.yaml"
        
        self.kelly_config = self._load_kelly_config(kelly_config_path)
        
        # Initialize dampened Kelly components
        self.kelly_engine: Optional[DampenedKellyEngine] = None
        self.kelly_tracker: Optional[KellyHistoryTracker] = None
        self.correlation_handler: Optional[PortfolioCorrelationHandler] = None
        
        # Backward compatibility: Basic Kelly parameters
        self.kelly_multiplier: float = 0.25
        self.max_position_size: float = getattr(config, 'max_position_size', 0.5)
        self.min_position_size: float = getattr(config, 'min_position_size', 0.01)
        self.confidence_threshold: float = 0.6
        self.ml_weight: float = 0.7
        
        # State management
        self.is_initialized: bool = False
        self.position_sizing_history: List[Dict[str, Any]] = []
        
        # Track current drawdown for dampening
        self.current_drawdown: float = 0.0
        
        # Portfolio state for correlation tracking
        self.current_positions: Dict[str, Dict[str, Any]] = {}
    
    def _load_kelly_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load Kelly sizing configuration.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Kelly configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                tprint_warning(f"⚠️ Kelly config not found at {config_path}, using defaults")
                return self._get_default_kelly_config()
            
            with open(config_file, 'r') as f:
                full_config = yaml.safe_load(f)
            
            if 'dampened_kelly' not in full_config:
                tprint_warning("⚠️ 'dampened_kelly' section not found in config, using defaults")
                return self._get_default_kelly_config()
            
            tprint_info(f"✅ Loaded Kelly config from {config_path}")
            return full_config['dampened_kelly']
        
        except Exception as e:
            tprint_error(f"❌ Error loading Kelly config: {e}, using defaults")
            self.logger.error(f"Error loading Kelly config: {e}")
            return self._get_default_kelly_config()
    
    def _get_default_kelly_config(self) -> Dict[str, Any]:
        """Get default Kelly configuration if file not found."""
        return {
            'regime_params': {},
            'global_fallback': {
                'lambda_base': 0.15,
                'beta_position': 2.0,
                'beta_leverage': 1.5,
                'prior_alpha': 30.0,
                'ess_threshold': 60,
                'entropy_threshold': 1.0,
                'n_min_samples': 25,
                'f_floor': 0.005,
                'lev_floor': 1.2,
                'decay_theta': 0.90
            },
            'lambda_eff_components': {
                'ess_sigmoid_kappa': 0.1,
                'entropy_scale': 0.5,
                'variance_penalty': 2.0
            },
            'binning': {
                'score_bins': [0.5, 0.6, 0.7, 0.8, 0.9],
                'volatility_bins': [0.005, 0.01, 0.02, 0.04],
                'enable_adaptive_merging': True,
                'stale_bin_days': 90
            },
            'r_tracking': {
                'use_realized_r': True,
                'r_percentile': 25,
                'r_instability_threshold': 2.0,
                'r_instability_prior_boost': 2.0,
                'default_r': 2.0
            },
            'safety_limits': {
                'max_leverage': 3.0,
                'max_per_trade_pct': 0.15,
                'max_exposure_per_asset': 0.30,
                'max_kelly_fraction': 0.5,
                'max_acceptable_drawdown': 0.15
            },
            'correlation': {
                'enabled': True,
                'window_days': 30,
                'high_corr_threshold': 0.7,
                'high_corr_penalty': 0.30,
                'moderate_corr_threshold': 0.4,
                'moderate_corr_penalty': 0.15,
                'per_trade_corr_limit': 0.8
            }
        }
    
    def set_leverage_manager(self, leverage_manager: LeverageManager) -> None:
        """Set the leverage manager for integration."""
        self.leverage_manager = leverage_manager
        tprint_debug("Leverage manager set for Position Sizer")
    
    def set_risk_calculator(self, risk_calculator: RiskCalculator) -> None:
        """Set the risk calculator for integration."""
        self.risk_calculator = risk_calculator
        tprint_debug("Risk calculator set for Position Sizer")
    
    @handles_errors
    async def initialize(self) -> bool:
        """Initialize position sizer with dampened Kelly components."""
        try:
            tprint_info("🔄 Initializing Position Sizer with Dampened Kelly...")
            self.logger.info("Initializing Position Sizer...")
            
            # Validate configuration
            if not self._validate_configuration():
                tprint_error("❌ Position Sizer configuration validation failed")
                return False
            
            # Initialize dampened Kelly engine
            self.kelly_engine = DampenedKellyEngine(self.kelly_config)
            tprint_info("✅ Dampened Kelly Engine initialized")
            
            # Initialize Kelly history tracker
            self.kelly_tracker = KellyHistoryTracker(self.kelly_config)
            
            # Try to load existing bins
            bins_loaded = self._load_kelly_bins()
            if bins_loaded:
                tprint_info("✅ Kelly bins loaded from artifacts")
            else:
                tprint_info("📊 Starting with fresh Kelly bins")
            
            # Initialize correlation handler
            self.correlation_handler = PortfolioCorrelationHandler(self.kelly_config)
            tprint_info("✅ Correlation handler initialized")
            
            self.is_initialized = True
            tprint_success("✅ Position Sizer with Dampened Kelly initialized successfully")
            self.logger.info("✅ Position Sizer initialized successfully")
            return True
        
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Position Sizer: {e}")
            self.logger.error(f"❌ Failed to initialize Position Sizer: {e}")
            return False
    
    def _load_kelly_bins(self) -> bool:
        """
        Load Kelly bins from artifacts if available.
        
        Returns:
            True if bins loaded successfully
        """
        try:
            # Get symbol and timeframe from config
            symbol = getattr(self.config, 'symbol', 'BTCUSDT')
            timeframe = getattr(self.config, 'timeframe', '15m')
            
            # Build artifact path
            artifacts_dir = self.kelly_config.get('persistence', {}).get('artifacts_dir', 'checkpoints/kelly_sizing')
            pattern = self.kelly_config.get('persistence', {}).get('bins_filename_pattern', 'kelly_bins_{symbol}_{timeframe}.pkl')
            filename = pattern.format(symbol=symbol, timeframe=timeframe)
            bin_file = Path(artifacts_dir) / filename
            
            if bin_file.exists():
                self.kelly_tracker = KellyHistoryTracker.load_from_file(bin_file)
                self.logger.info(f"Loaded Kelly bins from {bin_file}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.warning(f"Could not load Kelly bins: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate position sizer configuration."""
        try:
            if self.max_position_size <= self.min_position_size:
                tprint_error("max_position_size must be greater than min_position_size")
                return False
            
            # Validate Kelly config has required sections
            required_sections = ['regime_params', 'global_fallback', 'binning', 'safety_limits']
            for section in required_sections:
                if section not in self.kelly_config:
                    tprint_error(f"Kelly config missing required section: {section}")
                    return False
            
            tprint_debug("✅ Position Sizer configuration validated")
            return True
        
        except Exception as e:
            tprint_error(f"Configuration validation failed: {e}")
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_inputs(
        self,
        symbol: str,
        current_price: float,
        account_balance: float,
        analyst_confidence: float,
        tactician_confidence: float
    ) -> None:
        """Validate inputs for position sizing."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"symbol must be a non-empty string, got {symbol}")
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError(f"current_price must be a positive finite number, got {current_price}")
        if not math.isfinite(account_balance) or account_balance <= 0:
            raise ValueError(f"account_balance must be a positive finite number, got {account_balance}")
        if not math.isfinite(analyst_confidence) or not (0 <= analyst_confidence <= 1):
            raise ValueError(f"analyst_confidence must be between 0 and 1, got {analyst_confidence}")
        if not math.isfinite(tactician_confidence) or not (0 <= tactician_confidence <= 1):
            raise ValueError(f"tactician_confidence must be between 0 and 1, got {tactician_confidence}")
    
    def _extract_volatility(self, ml_predictions: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Extract or calculate volatility for binning.
        
        Args:
            ml_predictions: ML predictions dictionary
            market_data: Optional market data with OHLC
            
        Returns:
            Normalized volatility (ATR-based or similar)
        """
        # Try to get from predictions first
        if 'volatility' in ml_predictions:
            return ml_predictions['volatility']
        
        if 'atr_normalized' in ml_predictions:
            return ml_predictions['atr_normalized']
        
        # Try to calculate from market data
        if market_data and 'atr' in market_data and 'close' in market_data:
            atr = market_data['atr']
            close = market_data['close']
            if close > 0:
                return atr / close  # Normalized ATR
        
        # Fallback: use default moderate volatility
        self.logger.warning("Could not extract volatility, using default 0.015")
        return 0.015
    
    def _extract_regime_id(self, ml_predictions: Dict[str, Any]) -> Optional[int]:
        """
        Extract regime ID from predictions.
        
        Args:
            ml_predictions: ML predictions dictionary
            
        Returns:
            Regime ID or None if unknown
        """
        # Try various keys
        for key in ['regime_id', 'regime', 'hmm_regime', 'cluster_id']:
            if key in ml_predictions:
                regime = ml_predictions[key]
                if regime is not None and regime >= 0:
                    return int(regime)
        
        return None
    
    def _extract_ensemble_uncertainty(self, ml_predictions: Dict[str, Any]) -> tuple[float, float]:
        """
        Extract ESS and entropy from ensemble predictions.
        
        Args:
            ml_predictions: ML predictions dictionary
            
        Returns:
            Tuple of (ESS, entropy)
        """
        ess = ml_predictions.get('ess', ml_predictions.get('effective_sample_size', 100.0))
        entropy = ml_predictions.get('entropy', ml_predictions.get('ensemble_entropy', 0.5))
        
        return float(ess), float(entropy)
    
    def update_drawdown(self, current_dd: float) -> None:
        """
        Update current drawdown for dampening.
        
        Args:
            current_dd: Current drawdown as fraction (e.g., 0.10 for 10%)
        """
        self.current_drawdown = max(0.0, current_dd)
    
    def update_position(self, symbol: str, size: float, leverage: float) -> None:
        """
        Update position tracking for correlation.
        
        Args:
            symbol: Asset symbol
            size: Position size (fraction of portfolio)
            leverage: Position leverage
        """
        if self.correlation_handler:
            self.correlation_handler.update_position(symbol, size, leverage)
        
        if size > 0:
            self.current_positions[symbol] = {'size': size, 'leverage': leverage}
        elif symbol in self.current_positions:
            del self.current_positions[symbol]
    
    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update price history for correlation calculation.
        
        Args:
            symbol: Asset symbol
            price: Current price
            timestamp: Price timestamp (defaults to now)
        """
        if self.correlation_handler:
            self.correlation_handler.update_price(symbol, price, timestamp)
    
    @handles_errors
    @log_execution_time()
    @traced(span_name="calculate_position_size")
    async def calculate_position_size(
        self,
        symbol: str,
        ml_predictions: Dict[str, Any],
        current_price: float,
        account_balance: float,
        analyst_confidence: float = 0.5,
        tactician_confidence: float = 0.5,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> PositionSizeResult:
        """
        Calculate position size using dampened Kelly criterion.

        This is the main entry point that replaces the basic Kelly calculation
        with the production-hardened dampened Kelly system.

        Args:
            symbol: Trading symbol
            ml_predictions: ML confidence predictions (must include combined_confidence)
            current_price: Current market price
            account_balance: Account balance for position sizing
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score
            stop_loss_price: Stop loss price (optional, for R calculation)
            volatility: Market volatility (optional, will be extracted if not provided)
            market_data: Optional market data with OHLC for volatility calculation

        Returns:
            PositionSizeResult with complete metadata
        """
        tprint(f"calculate_position_size called: symbol={symbol}, current_price={current_price:.2f}, account_balance={account_balance:.2f}, analyst_conf={analyst_confidence:.3f}, tactician_conf={tactician_confidence:.3f}")

        try:
            if not self.is_initialized:
                tprint("Position Sizer not initialized, raising error")
                raise RuntimeError("Position Sizer not initialized")

            # Validate inputs
            self._validate_inputs(symbol, current_price, account_balance, analyst_confidence, tactician_confidence)
            tprint("Input validation passed")

            # Extract required inputs for Kelly engine
            model_score = ml_predictions.get('combined_confidence', 0.5)
            vol = volatility if volatility is not None else self._extract_volatility(ml_predictions, market_data)
            regime_id = self._extract_regime_id(ml_predictions)
            ess, entropy = self._extract_ensemble_uncertainty(ml_predictions)
            tprint(f"Extracted inputs: model_score={model_score:.3f}, vol={vol:.4f}, regime_id={regime_id}, ess={ess:.2f}, entropy={entropy:.3f}")

            # Lookup bin with adaptive fallback
            params = self.kelly_engine.get_regime_params(regime_id)
            n_min = params.get('n_min_samples', 25)
            tprint(f"Regime params retrieved, n_min={n_min}")

            bin_data, merge_level = self.kelly_tracker.lookup_bin(
                score=model_score,
                volatility=vol,
                regime_id=regime_id,
                n_min=n_min,
                current_time=datetime.now()
            )
            tprint(f"Bin lookup complete: merge_level={merge_level}, samples={bin_data.total_samples():.1f}, win_rate={bin_data.win_rate():.3f}")
            
            # Calculate dampened Kelly
            tprint("Calculating dampened Kelly position and leverage...")
            kelly_result: KellyResult = self.kelly_engine.calculate_position_and_leverage(
                wins=bin_data.wins,
                losses=bin_data.losses,
                regime_id=regime_id,
                ess=ess,
                entropy=entropy,
                r_realized=bin_data.r_realized,
                current_dd=self.current_drawdown,
                bin_merge_level=merge_level,
                bin_last_updated=bin_data.last_updated,
                is_bin_stale=bin_data.is_stale
            )
            tprint(f"Kelly calculation complete: f_final={kelly_result.f_final:.4f}, leverage_final={kelly_result.leverage_final:.2f}, reason_codes={kelly_result.reason_codes}")

            # Get base position size from Kelly
            base_size = kelly_result.f_final
            tprint(f"Base position size from Kelly: {base_size:.4f}")

            # Apply correlation adjustment if enabled
            correlation_adjusted = False
            if self.correlation_handler and self.correlation_handler.enabled:
                tprint("Checking correlation adjustment...")
                adjusted_size, was_adjusted, corr_metadata = self.correlation_handler.calculate_correlation_adjusted_size(
                    symbol=symbol,
                    base_size=base_size,
                    proposed_leverage=kelly_result.leverage_final,
                    current_time=datetime.now()
                )

                if was_adjusted:
                    tprint(f"Correlation adjustment applied: {base_size:.4f} -> {adjusted_size:.4f}")
                    base_size = adjusted_size
                    correlation_adjusted = True
                    kelly_result.correlation_adjusted = True
                    kelly_result.metadata['correlation_adjustment'] = corr_metadata
                else:
                    tprint("No correlation adjustment needed")

            # Calculate final size in account currency
            final_size = base_size * account_balance
            tprint(f"Final size before limits: {final_size:.2f} ({base_size:.4f} * {account_balance:.2f})")

            # Apply min/max limits
            min_size = self.min_position_size * account_balance
            max_size = self.max_position_size * account_balance
            final_size = max(min_size, min(final_size, max_size))
            tprint(f"Final size after limits: {final_size:.2f} (min={min_size:.2f}, max={max_size:.2f})")

            # Get leverage (already calculated by Kelly engine)
            leverage = kelly_result.leverage_final
            tprint(f"Final leverage: {leverage:.2f}x")

            # Validate risk if RiskCalculator is available
            risk_warnings = []
            if self.risk_calculator and stop_loss_price:
                tprint("Risk calculator available, validation skipped (placeholder)")
                # Risk validation would go here
                pass
            
            # Build result
            result = PositionSizeResult(
                symbol=symbol,
                recommended_size=final_size,
                max_size=max_size,
                min_size=min_size,
                leverage=leverage,
                confidence=model_score,
                kelly_size=kelly_result.f_kelly * account_balance,
                ml_size=base_size * account_balance,  # Before correlation adjustment
                sizing_method="dampened_kelly",
                metadata={
                    'kelly_result': kelly_result.to_dict(),
                    'bin_info': {
                        'total_samples': bin_data.total_samples(),
                        'win_rate': bin_data.win_rate(),
                        'merge_level': merge_level,
                        'is_stale': bin_data.is_stale
                    },
                    'inputs': {
                        'model_score': model_score,
                        'volatility': vol,
                        'regime_id': regime_id,
                        'ess': ess,
                        'entropy': entropy
                    },
                    'adjustments': {
                        'correlation_adjusted': correlation_adjusted,
                        'drawdown_dampened': kelly_result.dd_dampening_factor < 1.0,
                        'kelly_fraction_clipped': kelly_result.kelly_fraction_clip_applied
                    },
                    'config_version': kelly_result.config_version,
                    'timestamp': datetime.now().isoformat()
                }
            )
            tprint(f"Position size result created: recommended_size={final_size:.2f}, leverage={leverage:.2f}x, confidence={model_score:.3f}")

            # Track in history
            self.position_sizing_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'size': final_size,
                'leverage': leverage,
                'config_version': kelly_result.config_version,
                'reason_codes': kelly_result.reason_codes
            })

            # Keep history limited
            if len(self.position_sizing_history) > 1000:
                self.position_sizing_history = self.position_sizing_history[-1000:]
                tprint("Position sizing history trimmed to 1000 entries")

            tprint(f"calculate_position_size returning: size={result.recommended_size:.2f}, leverage={result.leverage:.2f}x, method={result.sizing_method}")
            return result
        
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            tprint_error(f"❌ Position sizing error: {e}")
            
            # Return safe fallback
            return PositionSizeResult(
                symbol=symbol,
                recommended_size=self.min_position_size * account_balance,
                max_size=self.max_position_size * account_balance,
                min_size=self.min_position_size * account_balance,
                leverage=1.0,
                confidence=0.0,
                kelly_size=0.0,
                ml_size=0.0,
                sizing_method="fallback",
                metadata={'error': str(e)}
            )
    
    def record_trade_outcome(
        self,
        symbol: str,
        score: float,
        volatility: float,
        regime_id: Optional[int],
        is_win: bool,
        entry_price: float,
        exit_price: float,
        stop_loss_price: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record trade outcome in Kelly bins for future sizing.

        Args:
            symbol: Trading symbol
            score: Model score used for entry
            volatility: Volatility at entry
            regime_id: Regime at entry
            is_win: Whether trade was a win
            entry_price: Entry price
            exit_price: Exit price
            stop_loss_price: Stop loss price
            timestamp: Trade close timestamp
        """
        tprint(f"record_trade_outcome called: symbol={symbol}, score={score:.3f}, volatility={volatility:.4f}, regime_id={regime_id}, is_win={is_win}, entry={entry_price:.2f}, exit={exit_price:.2f}, stop_loss={stop_loss_price:.2f}")

        if not self.kelly_tracker:
            tprint("No kelly_tracker available, skipping trade outcome recording")
            return

        try:
            # Calculate realized R
            if stop_loss_price > 0:
                risk = abs(entry_price - stop_loss_price)
                profit = exit_price - entry_price if is_win else -(entry_price - exit_price)
                r_realized = abs(profit / risk) if risk > 0 else 1.0
                tprint(f"Realized R calculated: {r_realized:.2f} (risk={risk:.2f}, profit={profit:.2f})")
            else:
                r_realized = 2.0  # Default
                tprint(f"Using default R: {r_realized:.2f} (no stop loss price)")

            # Update bin
            self.kelly_tracker.update_bin(
                score=score,
                volatility=volatility,
                regime_id=regime_id,
                is_win=is_win,
                r_realized=r_realized,
                timestamp=timestamp or datetime.now()
            )
            tprint("Trade outcome recorded in Kelly bins")

            # Periodically save bins (every 100 trades)
            if len(self.position_sizing_history) % 100 == 0:
                tprint(f"Periodic save triggered (every 100 trades, current history: {len(self.position_sizing_history)})")
                self._save_kelly_bins()

        except Exception as e:
            self.logger.error(f"Error recording trade outcome: {e}")
            tprint(f"Error recording trade outcome: {e}")
    
    def _save_kelly_bins(self) -> None:
        """Save Kelly bins to artifacts."""
        try:
            symbol = getattr(self.config, 'symbol', 'BTCUSDT')
            timeframe = getattr(self.config, 'timeframe', '15m')
            
            artifacts_dir = self.kelly_config.get('persistence', {}).get('artifacts_dir', 'checkpoints/kelly_sizing')
            pattern = self.kelly_config.get('persistence', {}).get('bins_filename_pattern', 'kelly_bins_{symbol}_{timeframe}.pkl')
            filename = pattern.format(symbol=symbol, timeframe=timeframe)
            bin_file = Path(artifacts_dir) / filename
            
            self.kelly_tracker.save_to_file(bin_file)
            self.logger.info(f"Saved Kelly bins to {bin_file}")
        
        except Exception as e:
            self.logger.error(f"Error saving Kelly bins: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get position sizer statistics.
        
        Returns:
            Dictionary with stats
        """
        stats = {
            'is_initialized': self.is_initialized,
            'total_sizing_decisions': len(self.position_sizing_history),
            'current_drawdown': self.current_drawdown,
            'kelly_engine_version': self.kelly_engine.get_config_version() if self.kelly_engine else None
        }
        
        if self.kelly_tracker:
            stats['bin_coverage'] = self.kelly_tracker.get_bin_coverage_stats()
            stats['staleness'] = self.kelly_tracker.check_staleness_all_bins()
        
        if self.correlation_handler:
            stats['portfolio'] = self.correlation_handler.get_portfolio_stats()
        
        return stats


async def setup_position_sizer(
    config: TradingConfig,
    leverage_manager: Optional[LeverageManager] = None,
    risk_calculator: Optional[RiskCalculator] = None
) -> Optional[PositionSizer]:
    """Setup and initialize position sizer."""
    try:
        tprint_info("🔄 Setting up Position Sizer...")
        position_sizer = PositionSizer(config, leverage_manager, risk_calculator)
        success = await position_sizer.initialize()
        if success:
            tprint_success("✅ Position Sizer setup completed successfully")
            return position_sizer
        tprint_warning("⚠️ Position Sizer setup completed but initialization failed")
        return None
    except Exception as e:
        tprint_error(f"❌ Failed to setup position sizer: {e}")
        logger.error(f"❌ Failed to setup position sizer: {e}")
        return None
