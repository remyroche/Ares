"""
Spectral Specialists for Adaptive Event-Driven Labeling (AEDL)

This module transforms traditional specialists into 5-scale spectral versions
optimized for frequency-dependent analysis and cross-scale resonance detection.

Key Features:
- Transform 4 priority specialists to spectral domain
- 5-scale decomposition for each specialist
- Integration with existing causal specialist framework
- Optimized for 2-4 hour trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass, asdict

from src.feature_generation.utils.step06_labeling_components.optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling,
)
from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


def _rolling_mad(values: np.ndarray) -> float:
    """Median absolute deviation helper for rolling windows."""
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    median = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - median)))


@dataclass
class TBMConfig:
    """Triple-barrier configuration shared across specialists."""

    profit_take_multiplier: float = 0.015  # Tripled from 0.004
    stop_loss_multiplier: float = 0.006   # Tripled from 0.0025
    time_barrier_minutes: int = 720        # 48 bars * 15m = 720m
    max_lookahead: int = 100               # Cover 48 bars with buffer
    binary_classification: bool = True
    transaction_cost: float = DEFAULT_TRANSACTION_COST

    def merge(self, overrides: Optional[Dict[str, Any]] = None) -> "TBMConfig":
        data = asdict(self)
        if overrides:
            data.update(overrides)
        return TBMConfig(**data)

    def to_kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveVolatilityFilterConfig:
    """Adaptive volatility filter parameters."""

    window: int = 48
    floor_multiplier: float = 1.25
    min_vol_rank: float = 0.05
    hard_floor: float = 1e-4
    max_surprise: float = 8.0
    eps: float = 1e-9

    def merge(self, overrides: Optional[Dict[str, Any]] = None) -> "AdaptiveVolatilityFilterConfig":
        data = asdict(self)
        if overrides:
            data.update(overrides)
        return AdaptiveVolatilityFilterConfig(**data)


@dataclass
class SpecialistEventConfig:
    """Event calibration configuration."""

    base_activation_zscore: float = 1.5
    min_coverage: float = 0.04
    max_coverage: float = 0.25
    surprise_scaler: float = 0.75
    min_events: int = 30
    responsiveness_floor: float = 0.05
    correlation_threshold: float = 0.85

    def merge(self, overrides: Optional[Dict[str, Any]] = None) -> "SpecialistEventConfig":
        data = asdict(self)
        if overrides:
            data.update(overrides)
        return SpecialistEventConfig(**data)


class SpectralSpecialists:
    """
    Transform traditional specialists into 5-scale spectral versions.
    
    Priority Specialists for 2-4h trades:
    1. Inventory Specialist (Priority 1) - Dealer exhaustion detection
    2. Volume Specialist (Priority 2) - Micro-surge vs macro-trend resonance
    3. Volatility Specialist (Priority 3) - Dynamic wavelet thresholding
    4. Information Specialist (Causal Addition) - PIN/VPIN informed flow resonance
    """
    
    def __init__(
        self,
        priority_specialists: List[str] = None,
        verbose: bool = True,
        tbm_config: Optional[Dict[str, Any]] = None,
        avf_config: Optional[Dict[str, Any]] = None,
        event_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Spectral Specialists transformer.
        
        Args:
            priority_specialists: List of specialist names to prioritize
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        # Default now includes the new 2026 De Prado specialists
        self.priority_specialists = priority_specialists or [
            'inventory_specialist',
            'volume_specialist', 
            'volatility_specialist',
            'information_specialist',
            'cusum_break_specialist',
            'entropy_specialist',
            'tick_rule_specialist',
            'fractal_efficiency_specialist',
            'liquidity_shock_specialist',
            'gap_specialist',
            'trend_specialist',
            'reversal_specialist',
            'volatility_breakout_specialist'
        ]
        
        # Specialist descriptions
        self.specialist_descriptions = {
            'inventory_specialist': {
                'priority': 1,
                'description': 'Dealer exhaustion detection',
                'role': 'Micro-divergence in 15m wavelet before 4h trend impact',
                'key_scales': ['d2', 'd3'],  # 15m-1h, 1h-4h
                'resonance_pairs': [('d1', 'd3'), ('d2', 'd4')]
            },
            'volume_specialist': {
                'priority': 2,
                'description': 'Micro-surge vs macro-trend resonance',
                'role': 'Detect volume micro-surge resonating with macro-trend',
                'key_scales': ['d1', 'd2'],  # 5m-15m, 15m-1h
                'resonance_pairs': [('d1', 'd3'), ('d2', 'd4')]
            },
            'volatility_specialist': {
                'priority': 3,
                'description': 'Volatility Z-Score and Shock Detection',
                'role': 'Detects volatility shocks as causal precursors to risk events',
                'key_scales': ['d1', 'd3'],  # 5m-15m, 1h-4h
                'resonance_pairs': [('d1', 'd3'), ('d2', 'd4')]
            },
            'information_specialist': {
                'priority': 4,
                'description': 'Price Action and Microstructure Signatures',
                'role': 'Strongest predictor of permanent price moves via PA info',
                'key_scales': ['d2', 'd3'],  # 15m-1h, 1h-4h
                'resonance_pairs': [('d2', 'd4'), ('d1', 'd3')]
            },
            'cusum_break_specialist': {
                'priority': 5,
                'description': 'Structural Break Detection (CUSUM)',
                'role': 'Detect regime shifts where underlying process changes',
                'key_scales': ['d3', 'd4'],  # 1h-4h, 4h-Daily
                'resonance_pairs': [('d2', 'd4')]
            },
            'entropy_specialist': {
                'priority': 6,
                'description': 'Market Entropy / Unpredictability',
                'role': 'Measure information content and signal-to-noise breakdown',
                'key_scales': ['d2', 'd3'],
                'resonance_pairs': [('d1', 'd3')]
            },
            'tick_rule_specialist': {
                'priority': 7,
                'description': 'Aggressor Flow Proxy',
                'role': 'Approximates buy vs sell pressure within bars',
                'key_scales': ['d1', 'd2'],
                'resonance_pairs': [('d1', 'd3')]
            },
            'fractal_efficiency_specialist': {
                'priority': 8,
                'description': 'Fractal Efficiency (Kaufman/Hurst)',
                'role': 'Distinguish directional trends (clean) from random walks (noisy)',
                'key_scales': ['d2', 'd4'],
                'resonance_pairs': [('d2', 'd4')]
            },
            'liquidity_shock_specialist': {
                'priority': 9,
                'description': 'Liquidity Shock (Amihud Proxy)',
                'role': 'Detects structural liquidity failures (Price Ease)',
                'key_scales': ['d1', 'd2'],
                'resonance_pairs': [('d1', 'd3')]
            },
            'gap_specialist': {
                'priority': 10,
                'description': 'Exogenous Shock (Gap)',
                'role': 'Detects overnight/weekend information injection',
                'key_scales': ['d1', 'd4'],
                'resonance_pairs': [('d1', 'd4')]
            },
            'trend_specialist': {
                'priority': 11,
                'description': 'Trend Persistence (Rolling Returns)',
                'role': 'Captures directional alpha and trend persistence',
                'key_scales': ['d3', 'd4'],
                'resonance_pairs': [('d3', 'd4')]
            },
            'reversal_specialist': {
                'priority': 12,
                'description': 'Mean Reversion (Oscillator)',
                'role': 'Detects overextended price action and mean-reversion events',
                'key_scales': ['d1', 'd2'],
                'resonance_pairs': [('d1', 'd3')]
            },
            'volatility_breakout_specialist': {
                'priority': 13,
                'description': 'Volatility Breakout (HL vs Baseline)',
                'role': 'Detects unexpected volatility/range expansion vs baseline',
                'key_scales': ['d2', 'd3'],
                'resonance_pairs': [('d2', 'd4')]
            }
        }
        
        self.tbm_config = TBMConfig().merge(tbm_config or {})
        self.avf_config = AdaptiveVolatilityFilterConfig().merge(avf_config or {})
        self.event_config = SpecialistEventConfig().merge(event_config or {})
        self._tbm_engine = OptimizedTripleBarrierLabeling(**self.tbm_config.to_kwargs())
        self._reliability_registry: Dict[str, Dict[str, Any]] = {}
        self._last_extracted_specialists: List[str] = []
        self._cached_diversity_report: Dict[str, Any] = {}

        if self.verbose:
            tprint_info("🎯 Spectral Specialists: Initializing...")
            tprint_info(f"   ⚙️ Priority specialists: {len(self.priority_specialists)}")
            for specialist in self.priority_specialists:
                desc = self.specialist_descriptions.get(specialist, {})
                tprint_info(f"      - {specialist}: {desc.get('description', 'N/A')}")
            tprint_success("   ✅ Spectral Specialists: Initialization complete")
    
    def extract_specialist_signals(
        self,
        df: pd.DataFrame,
        specialist_configs: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, pd.Series]:
        """
        Extract raw specialist signals from market data.
        
        Args:
            df: Market data with OHLCV and derived features
            specialist_configs: Configuration for each specialist
            
        Returns:
            Dictionary of specialist time series
        """
        try:
            if self.verbose:
                tprint_info("📊 Extracting raw specialist signals...")
            
            # Early data validation
            self._validate_input_data(df)
            
            specialist_signals = {}
            configs = specialist_configs or {}
            
            # Helper to safely extract and add signal with validation
            def _add_signal(name, extraction_func):
                if name in self.priority_specialists:
                    try:
                        signal = extraction_func(df, configs.get(name, {}))
                        if signal is not None:
                            # Validate signal quality
                            signal_quality = self._validate_signal_quality(signal, name)
                            if signal_quality['is_degenerate']:
                                tprint_warning(f"⚠️ {name} signal is degenerate: {signal_quality['issue']}")
                                # Still include but with warning
                            specialist_signals[name] = signal
                            if self.verbose:
                                tprint_info(f"      - {name}: mean={signal_quality['mean']:.6f}, std={signal_quality['std']:.6f}, nan%={signal_quality['nan_pct']:.2f}%")
                    except Exception as e:
                        tprint_error(f"❌ {name} extraction failed: {e}")
            
            _add_signal('inventory_specialist', self._extract_inventory_signal)
            _add_signal('volume_specialist', self._extract_volume_signal)
            _add_signal('volatility_specialist', self._extract_volatility_signal)
            _add_signal('information_specialist', self._extract_information_signal)
            
            # New Specialists
            _add_signal('cusum_break_specialist', self._extract_cusum_break_signal)
            _add_signal('entropy_specialist', self._extract_entropy_signal)
            _add_signal('tick_rule_specialist', self._extract_tick_rule_signal)
            _add_signal('fractal_efficiency_specialist', self._extract_fractal_efficiency_signal)
            _add_signal('liquidity_shock_specialist', self._extract_liquidity_shock_signal)
            _add_signal('gap_specialist', self._extract_gap_signal)
            _add_signal('trend_specialist', self._extract_trend_signal)
            _add_signal('reversal_specialist', self._extract_reversal_signal)
            _add_signal('volatility_breakout_specialist', self._extract_volatility_breakout_signal)

            # Optional: Cross-asset / market-state feature signals (ca__/ms__ prefixes)
            cross_asset_cfg = configs.get("cross_asset", {})
            prefixes = tuple(cross_asset_cfg.get("prefixes", ("ca__", "ms__")))
            max_signals = int(cross_asset_cfg.get("max_signals", 6))
            cross_asset_cols = [
                col for col in df.columns if isinstance(col, str) and col.startswith(prefixes)
            ]
            cross_asset_added = 0
            if cross_asset_cols:
                numeric_cols = [
                    col
                    for col in cross_asset_cols
                    if pd.api.types.is_numeric_dtype(df[col])
                ]
                if numeric_cols:
                    var_rank = df[numeric_cols].var().sort_values(ascending=False)
                    selected_cols = var_rank.head(max_signals).index.tolist()
                    for col in selected_cols:
                        signal = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
                        if signal.notna().sum() == 0:
                            continue
                        mean = signal.mean()
                        std = signal.std()
                        normalized = (signal - mean) / (std + 1e-9)
                        name = f"{col}_specialist"
                        if name in specialist_signals:
                            continue
                        signal_quality = self._validate_signal_quality(normalized, name)
                        if signal_quality["is_degenerate"]:
                            tprint_warning(
                                f"⚠️ {name} signal is degenerate: {signal_quality['issue']}"
                            )
                        specialist_signals[name] = normalized
                        cross_asset_added += 1
                if self.verbose and cross_asset_added > 0:
                    tprint_info(
                        f"   🌐 Added {cross_asset_added} cross-asset specialists from {len(cross_asset_cols)} features"
                    )
            
            if self.verbose:
                tprint_success(f"   ✅ Extracted {len(specialist_signals)} specialist signals:")
                for name, signal in specialist_signals.items():
                    tprint_info(f"      - {name}: {len(signal)} samples")
                
                # Overall signal quality report
                self._log_signal_quality_summary(specialist_signals)
            
            self._last_extracted_specialists = list(specialist_signals.keys())
            
            return specialist_signals
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist signal extraction failed: {e}")
            return {}

    def _validate_input_data(self, df: pd.DataFrame):
        """Validate input dataframe for common issues."""
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for non-finite values
        for col in required_cols:
            non_finite_count = df[col].isna().sum() + np.isinf(df[col]).sum()
            if non_finite_count > 0:
                tprint_warning(f"⚠️ {col} contains {non_finite_count} non-finite values")
        
        # Check data quality
        if len(df) < 100:
            tprint_warning(f"⚠️ Small dataset: {len(df)} rows may cause unreliable signals")
        
        # Check price consistency
        price_cols = ['open', 'high', 'low', 'close']
        for i in range(len(df)):
            if not (df.iloc[i]['low'] <= df.iloc[i]['open'] <= df.iloc[i]['high'] and
                    df.iloc[i]['low'] <= df.iloc[i]['close'] <= df.iloc[i]['high']):
                tprint_warning(f"⚠️ Price inconsistency detected at index {i}")
                break

    def _validate_signal_quality(self, signal: pd.Series, name: str) -> Dict[str, Any]:
        """Validate signal quality and detect degenerate cases."""
        quality = {
            'mean': signal.mean(),
            'std': signal.std(),
            'nan_pct': signal.isna().sum() / len(signal) * 100,
            'is_degenerate': False,
            'issue': None
        }
        
        # Check for zero variance (constant signal)
        if quality['std'] < 1e-10:
            quality['is_degenerate'] = True
            quality['issue'] = 'Zero variance (constant signal)'
        
        # Check for excessive NaN values
        elif quality['nan_pct'] > 50:
            quality['is_degenerate'] = True
            quality['issue'] = f'High NaN percentage: {quality["nan_pct"]:.1f}%'
        
        # Check for extreme values
        elif np.abs(signal).max() > 1e6:
            quality['is_degenerate'] = True
            quality['issue'] = 'Extreme values detected'
        
        # Check for very small signal magnitude
        elif np.abs(signal).max() < 1e-8:
            quality['is_degenerate'] = True
            quality['issue'] = 'Signal magnitude too small'
        
        return quality

    def _log_signal_quality_summary(self, specialist_signals: Dict[str, pd.Series]):
        """Log comprehensive signal quality summary."""
        if not specialist_signals:
            tprint_error("❌ No specialist signals to validate")
            return
        
        tprint_info("📊 Signal Quality Summary:")
        
        degenerate_count = 0
        for name, signal in specialist_signals.items():
            quality = self._validate_signal_quality(signal, name)
            status = "✅ OK" if not quality['is_degenerate'] else "❌ DEGENERATE"
            tprint_info(f"   {name}: {status} (std={quality['std']:.2e}, nan%={quality['nan_pct']:.1f}%)")
            if quality['is_degenerate']:
                degenerate_count += 1
                tprint_warning(f"      Issue: {quality['issue']}")
        
        if degenerate_count > 0:
            tprint_error(f"❌ {degenerate_count}/{len(specialist_signals)} specialist signals are degenerate")
            tprint_error("   This will cause zero resonance in spectral analysis")
        else:
            tprint_success("✅ All specialist signals have acceptable quality")

    def generate_specialist_event_dataset(
        self,
        df: pd.DataFrame,
        specialist_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        tbm_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate fully-labeled specialist events with TBM alignment, AVF filtering,
        surprise scores, and reliability diagnostics.
        """
        if self.verbose:
            tprint_info("🧠 Generating specialist event dataset with TBM + AVF safeguards")

        specialist_signals = self.extract_specialist_signals(df, specialist_configs)
        if not specialist_signals:
            return {}

        tbm_labels = self._label_market_events(df, tbm_overrides)
        market_context = self._prepare_market_context(df)

        specialist_payload: Dict[str, Dict[str, Any]] = {}
        metrics_summary: List[Dict[str, Any]] = []

        for name, signal in specialist_signals.items():
            (
                filtered_signal,
                surprise,
                activation_mask,
                vol_floor,
                avf_metadata,
            ) = self._apply_adaptive_volatility_filter(name, signal, market_context)

            event_frame = pd.DataFrame(index=signal.index)
            event_frame["raw_signal"] = signal
            event_frame["filtered_signal"] = filtered_signal
            event_frame["surprise"] = surprise
            event_frame["activation"] = activation_mask.astype(bool)
            event_frame["vol_floor"] = vol_floor
            event_frame["direction"] = np.sign(filtered_signal).replace(0, np.nan)
            event_frame["tbm_label"] = tbm_labels["label"]
            event_frame["potential_profit_pct"] = tbm_labels["potential_profit_pct"]
            event_frame["zone_score"] = self._compute_zone_score(
                event_frame["surprise"], event_frame["tbm_label"]
            )
            event_frame["meta_label"] = np.where(
                event_frame["activation"] & (event_frame["tbm_label"] != 0),
                (event_frame["direction"] == event_frame["tbm_label"]).astype(float),
                np.nan,
            )

            metrics = self._compute_specialist_metrics(name, event_frame, tbm_labels)
            self._reliability_registry[name] = metrics
            metrics_summary.append({"specialist": name, **metrics})

            specialist_payload[name] = {
                "events": event_frame[event_frame["activation"]].copy(),
                "full_frame": event_frame,
                "metrics": metrics,
                "avf_metadata": avf_metadata,
            }

            if self.verbose:
                tprint_info(
                    f"   ↳ {name}: events={metrics.get('event_count', 0)} "
                    f"precision={metrics.get('precision', np.nan):.2f} "
                    f"recall={metrics.get('recall', np.nan):.2f}"
                )

        summary_df = pd.DataFrame(metrics_summary) if metrics_summary else pd.DataFrame()
        if self.verbose and not summary_df.empty:
            tprint_success("   ✅ Specialist event dataset ready (TBM-aligned)")

        self._cached_diversity_report = self._compute_diversity_diagnostics(
            specialist_payload,
            metrics_summary
        )

        return {
            "specialists": specialist_payload,
            "tbm_labels": tbm_labels,
            "summary": summary_df.sort_values("precision", ascending=False)
            if not summary_df.empty
            else summary_df,
            "diversity_diagnostics": self._cached_diversity_report
        }

    def get_reliability_report(self) -> Dict[str, Dict[str, Any]]:
        """Return latest reliability metrics for each specialist."""
        return self._reliability_registry
    
    def get_last_extracted_specialists(self) -> List[str]:
        """Return names of specialists successfully extracted in the last run."""
        return list(self._last_extracted_specialists)
    
    def get_diversity_report(self) -> Dict[str, Any]:
        """Return latest specialist diversity diagnostics."""
        return self._cached_diversity_report

    def _prepare_market_context(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        if "close" not in df.columns:
            raise ValueError("Market data must include 'close' for volatility context")

        returns = df["close"].pct_change().fillna(0)
        volatility = returns.rolling(self.avf_config.window).std()
        rolling_mad = returns.rolling(self.avf_config.window).apply(_rolling_mad, raw=True)
        vol_rank = volatility.rank(pct=True)
        # Fix for deprecated .mad()
        global_mad = float((returns - returns.mean()).abs().mean()) if not returns.empty else 0.0

        return {
            "returns": returns,
            "volatility": volatility,
            "rolling_mad": rolling_mad,
            "vol_rank": vol_rank,
            "global_mad": global_mad,
        }

    def _apply_adaptive_volatility_filter(
        self,
        specialist_name: str,
        signal: pd.Series,
        market_context: Dict[str, pd.Series],
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, float]]:
        floor = (
            market_context["rolling_mad"] * self.avf_config.floor_multiplier
        ).reindex(signal.index)
        fallback_floor = max(
            market_context["global_mad"] * self.avf_config.floor_multiplier,
            self.avf_config.hard_floor,
        )
        floor = floor.fillna(fallback_floor).clip(lower=self.avf_config.hard_floor)

        base_mask = signal.abs() >= floor
        surprise = signal / (floor + self.avf_config.eps)
        surprise = surprise.clip(-self.avf_config.max_surprise, self.avf_config.max_surprise)

        threshold = self._calibrate_activation_threshold(
            surprise.abs(), base_mask, self.event_config
        )
        activation_mask = base_mask & (surprise.abs() >= threshold)
        filtered_signal = signal.where(activation_mask, 0.0)

        avf_metadata = {
            "floor_median": float(np.nanmedian(floor)),
            "base_coverage": float(base_mask.mean() if len(base_mask) else 0.0),
            "activation_coverage": float(activation_mask.mean() if len(activation_mask) else 0.0),
            "activation_threshold": float(threshold),
        }

        if self.verbose:
            tprint_info(
                f"   • {specialist_name} AVF: coverage={avf_metadata['activation_coverage']:.2%}, "
                f"threshold={avf_metadata['activation_threshold']:.2f}"
            )

        return filtered_signal, surprise, activation_mask, floor, avf_metadata

    def _calibrate_activation_threshold(
        self,
        abs_surprise: pd.Series,
        base_mask: pd.Series,
        event_config: SpecialistEventConfig,
    ) -> float:
        if abs_surprise.empty or base_mask.sum() == 0:
            return event_config.base_activation_zscore

        candidate = event_config.base_activation_zscore
        coverage = (base_mask & (abs_surprise >= candidate)).mean()

        if np.isnan(coverage):
            coverage = 0.0

        abs_surprise_base = abs_surprise.where(base_mask)
        if coverage < event_config.min_coverage:
            quantile_target = max(0.0, 1 - event_config.min_coverage)
            candidate = float(abs_surprise_base.quantile(quantile_target))
        elif coverage > event_config.max_coverage:
            quantile_target = max(0.0, 1 - event_config.max_coverage)
            candidate = float(abs_surprise_base.quantile(quantile_target))

        if np.isnan(candidate) or candidate <= 0:
            candidate = event_config.base_activation_zscore

        return candidate

    def _label_market_events(
        self,
        df: pd.DataFrame,
        tbm_overrides: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Market data missing required columns: {missing}")

        tbm_cfg = self.tbm_config.merge(tbm_overrides or {})
        tbm_engine = (
            self._tbm_engine
            if tbm_overrides is None
            else OptimizedTripleBarrierLabeling(**tbm_cfg.to_kwargs())
        )
        tbm_result = tbm_engine.apply_triple_barrier_labeling_vectorized(df[required].copy())
        labeled = pd.DataFrame(
            {"label": 0, "potential_profit_pct": 0.0},
            index=df.index,
        )
        if isinstance(tbm_result, pd.DataFrame) and not tbm_result.empty:
            labeled.loc[tbm_result.index, "label"] = tbm_result["label"]
            labeled.loc[tbm_result.index, "potential_profit_pct"] = tbm_result[
                "potential_profit_pct"
            ]

        return labeled

    def _compute_zone_score(self, surprise: pd.Series, tbm_label: pd.Series) -> pd.Series:
        alignment = surprise * tbm_label
        score = 1.0 / (1.0 + np.exp(-(alignment * self.event_config.surprise_scaler)))
        return score.fillna(0.5)

    def _compute_specialist_metrics(
        self,
        specialist_name: str,
        event_frame: pd.DataFrame,
        tbm_labels: pd.DataFrame,
    ) -> Dict[str, Any]:
        activation_mask = event_frame["activation"].astype(bool)
        tbm_event_mask = tbm_labels["label"] != 0
        active = activation_mask & tbm_event_mask

        metrics: Dict[str, Any] = {
            "event_count": int(activation_mask.sum()),
            "coverage": float(activation_mask.mean() if len(activation_mask) else 0.0),
        }

        if active.sum() < self.event_config.min_events:
            metrics.update(
                {
                    "precision": np.nan,
                    "recall": np.nan,
                    "responsiveness": np.nan,
                    "marginal_value": np.nan,
                    "consensus_correlation": np.nan,
                    "avg_zone_score": np.nan,
                    "avg_surprise": np.nan,
                }
            )
            return metrics

        directions = event_frame.loc[active, "direction"]
        realized = tbm_labels.loc[active, "label"]
        profits = tbm_labels.loc[active, "potential_profit_pct"]

        correct = (directions == realized)
        precision = float(correct.mean())
        recall = float(active.sum() / max(tbm_event_mask.sum(), 1))
        responsiveness = float(
            event_frame.loc[active, "surprise"].corr(profits) or 0.0
        )
        marginal_value = float(
            profits.mean() - tbm_labels.loc[tbm_event_mask, "potential_profit_pct"].mean()
        )
        consensus_correlation = float(directions.corr(realized) or 0.0)
        avg_zone_score = float(event_frame.loc[active, "zone_score"].mean())
        avg_surprise = float(event_frame.loc[active, "surprise"].abs().mean())

        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "responsiveness": responsiveness,
                "marginal_value": marginal_value,
                "consensus_correlation": consensus_correlation,
                "avg_zone_score": avg_zone_score,
                "avg_surprise": avg_surprise,
            }
        )
        metrics["composite_reliability"] = self._score_specialist_reliability(metrics)

        return metrics

    def _score_specialist_reliability(self, metrics: Dict[str, Any]) -> float:
        """Blend precision/recall/responsiveness into a composite reliability score."""
        precision = float(np.clip(metrics.get("precision", 0.0), 0.0, 1.0) or 0.0)
        recall = float(np.clip(metrics.get("recall", 0.0), 0.0, 1.0) or 0.0)
        responsiveness = float(np.clip(metrics.get("responsiveness", 0.0), 0.0, 1.0) or 0.0)
        consensus_corr = float(np.clip(metrics.get("consensus_correlation", 0.0), -1.0, 1.0) or 0.0)
        avg_zone_score = float(np.clip(metrics.get("avg_zone_score", 0.0), 0.0, 1.0) or 0.0)

        score = (
            0.35 * responsiveness +
            0.30 * precision +
            0.20 * recall +
            0.10 * max(0.0, consensus_corr) +
            0.05 * avg_zone_score
        )
        return float(np.clip(score, 0.0, 1.0))

    def _compute_diversity_diagnostics(
        self,
        specialist_payload: Dict[str, Dict[str, Any]],
        metrics_summary: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute specialist diversity and redundancy diagnostics."""
        diagnostics: Dict[str, Any] = {}
        if not metrics_summary:
            return diagnostics

        metrics_df = pd.DataFrame(metrics_summary)
        if metrics_df.empty:
            return diagnostics

        metrics_df = metrics_df.set_index("specialist", drop=False)
        responsiveness = metrics_df["responsiveness"].astype(float)
        recall = metrics_df["recall"].astype(float)

        low_resp_mask = responsiveness.abs() < self.event_config.responsiveness_floor
        low_resp_specialists = metrics_df.loc[low_resp_mask.fillna(False), "specialist"].tolist()

        resp_recall_corr = None
        corr_sample = metrics_df[["responsiveness", "recall"]].dropna()
        if corr_sample.shape[0] >= 3:
            resp_recall_corr = corr_sample.corr(method="spearman").iloc[0, 1]

        filtered_series: List[pd.Series] = []
        for name, payload in specialist_payload.items():
            full_frame = payload.get("full_frame")
            if full_frame is None:
                continue
            if "filtered_signal" not in full_frame:
                continue
            series = full_frame["filtered_signal"].rename(name)
            filtered_series.append(series)

        correlated_pairs: List[Dict[str, Any]] = []
        if filtered_series:
            filtered_df = pd.concat(filtered_series, axis=1).dropna(how="all")
            if filtered_df.shape[1] > 1:
                corr_matrix = filtered_df.corr(method="spearman")
                for i, col_i in enumerate(corr_matrix.columns):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col_j = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        if np.isnan(corr_val):
                            continue
                        if abs(corr_val) >= self.event_config.correlation_threshold:
                            correlated_pairs.append(
                                {
                                    "pair": (col_i, col_j),
                                    "correlation": float(corr_val),
                                }
                            )
                diagnostics["correlation_matrix"] = corr_matrix.round(3).to_dict()

        coverage_stats = {
            "median": float(metrics_df["coverage"].median()),
            "min": float(metrics_df["coverage"].min()),
            "max": float(metrics_df["coverage"].max()),
        }

        diagnostics.update(
            {
                "resp_recall_corr": float(resp_recall_corr) if resp_recall_corr is not None else None,
                "avf_recalibration_candidates": low_resp_specialists,
                "redundant_pairs": correlated_pairs,
                "coverage": coverage_stats,
                "metrics_table": metrics_df[
                    ["precision", "recall", "responsiveness", "coverage", "composite_reliability"]
                ].to_dict("index"),
            }
        )

        if self.verbose and low_resp_specialists:
            tprint_warning(
                f"   ⚠️ AVF recalibration suggested for: {', '.join(low_resp_specialists)}"
            )
        if self.verbose and correlated_pairs:
            formatted = ", ".join(
                f"{a}↔{b} ({corr:.2f})" for (a, b), corr in
                [((pair["pair"][0], pair["pair"][1]), pair["correlation"]) for pair in correlated_pairs[:4]]
            )
            tprint_warning(f"   ⚠️ High specialist correlation detected: {formatted}")

        return diagnostics

    def _extract_trend_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract Trend Persistence signal.
        Logic: Rolling return (close - close[-N])
        """
        if self.verbose:
            tprint_info("🏹 Extracting trend specialist signal")
        try:
            if 'close' in df.columns:
                # Rolling return (e.g., 20 bars / 4-5 hours on 15m)
                window = 20
                rolling_return = df['close'].pct_change(window)
                
                # Normalize by rolling volatility to get Sharpe-like trend strength
                vol = df['close'].pct_change().rolling(window).std() * np.sqrt(window)
                
                trend_signal = rolling_return / (vol + 1e-9)
                
                # Z-Score normalize
                trend_signal = (trend_signal - trend_signal.rolling(50).mean()) / (trend_signal.rolling(50).std() + 1e-9)
                
                # DIAGNOSTIC LOGGING
                if self.verbose and trend_signal is not None:
                    # Check for effective constant signal
                    sig_std = trend_signal.std()
                    if sig_std < 1e-6:
                        raw_std = trend_pressure.std()
                        raw_mean = trend_pressure.mean()
                        if self.verbose:
                            tprint_warning(f"      ⚠️ Trend signal low variance: std={sig_std:.6f} (raw_std={raw_std:.6f}, raw_mean={raw_mean:.6f})")
                            tprint_warning(f"         Raw Range: [{trend_pressure.min():.4f}, {trend_pressure.max():.4f}]")
                
                return trend_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose: tprint_warning(f"      ⚠️ Trend signal extraction failed: {e}")
            return None

    def _extract_reversal_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract Mean Reversion signal.
        Logic: Stochastic oscillator proxy - (close - rolling_min)/(rolling_max - rolling_min)
        Centered around 0.
        """
        if self.verbose:
            tprint_info("↩️ Extracting reversal specialist signal")
        try:
            required = ['close', 'high', 'low']
            if all(c in df.columns for c in required):
                window = 20
                roll_low = df['low'].rolling(window).min()
                roll_high = df['high'].rolling(window).max()
                
                # Stochastic %K
                stoch = (df['close'] - roll_low) / (roll_high - roll_low + 1e-9)
                
                # Center around 0.5 -> -0.5 to 0.5
                # But we want "Surprise" -> Reversal pressure.
                # If stoch is high (1.0), reversal pressure is Downative (Sell).
                # If stoch is low (0.0), reversal pressure is Positive (Buy).
                # So we invert: (0.5 - stoch)
                # High Stoch (1.0) -> -0.5 signal (Sell)
                # Low Stoch (0.0) -> +0.5 signal (Buy)
                
                reversal_pressure = 0.5 - stoch
                
                # Normalize
                reversal_signal = (reversal_pressure - reversal_pressure.rolling(50).mean()) / (reversal_pressure.rolling(50).std() + 1e-9)
                
                return reversal_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose: tprint_warning(f"      ⚠️ Reversal signal extraction failed: {e}")
            return None

    def _extract_volatility_breakout_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract Volatility Breakout signal.
        Logic: Rolling High-Low / Baseline.
        """
        if self.verbose:
            tprint_info("💥 Extracting volatility breakout specialist signal")
        try:
            required = ['high', 'low', 'close']
            if all(c in df.columns for c in required):
                # Normalized Range (Parkinson proxy input)
                hl_range = (df['high'] - df['low']) / df['close']
                
                # Short and Long windows
                short_window = 10
                long_baseline = 50
                
                short_ma = hl_range.rolling(short_window).mean()
                baseline_ma = hl_range.rolling(long_baseline).mean()
                
                # Breakout Ratio
                breakout_ratio = short_ma / (baseline_ma + 1e-9)
                
                # We want "Surprise" when ratio is high.
                # If ratio > 1.0 -> Vol expansion.
                
                # KEY CHANGE: Directional Breakout
                # Range expansion is only a signal if we know WHICH WAY it broke out.
                direction = np.sign(df['close'].diff())
                vol_break_signal_raw = breakout_ratio * direction
                
                # Normalize
                vol_break_signal = (vol_break_signal_raw - vol_break_signal_raw.rolling(100).mean()) / (vol_break_signal_raw.rolling(100).std() + 1e-9)
                
                return vol_break_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose: tprint_warning(f"      ⚠️ Volatility Breakout signal extraction failed: {e}")
            return None
    
    def _extract_inventory_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """Extract inventory specialist signal (dealer inventory proxy) with temporal weighting."""
        if self.verbose:
            tprint_info("📈 Extracting inventory specialist signal")
        try:
            # Use volume-weighted price changes as inventory proxy
            if 'close' in df.columns and 'volume' in df.columns:
                price_change = df['close'].pct_change()
                # Normalize volume by its moving average to handle daily cycles
                vol_norm = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
                
                # Inventory pressure = price change * normalized volume
                inventory_pressure = price_change * vol_norm
                
                # Apply temporal weighting (EMA) to emphasize recent inventory accumulation
                inventory_signal = inventory_pressure.ewm(span=10).mean()
                
                # Normalize by rolling volatility of the signal
                inventory_signal = (inventory_signal - inventory_signal.rolling(50).mean()) / \
                                (inventory_signal.rolling(50).std() + 1e-9)
                
                if self.verbose and inventory_signal.std() < 1e-6:
                     tprint_warning(f"      ⚠️ Inventory signal low variance: raw_std={inventory_pressure.std():.6f}")

                return inventory_signal.fillna(0)
            
            return None
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Inventory signal extraction failed: {e}")
            return None
    
    def _extract_volume_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """Extract volume specialist signal with volatility normalization and AVF."""
        if self.verbose:
            tprint_info("📊 Extracting volume specialist signal")
        try:
            if 'volume' in df.columns and 'close' in df.columns:
                # 1. Volume-Weighted Price Efficiency (Informed Flow)
                # Concept: High volume is only "signal" if it results in efficient price movement.
                # High volume + Low movement = Churn/Noise (Absorption) -> Filtered out
                # High volume + High movement = Informed Breakout -> Signal
                
                # Bar Efficiency (Signed: -1.0 to 1.0)
                # +1.0 = Marubozu Up (Pure Buy Pressure)
                # -1.0 = Marubozu Down (Pure Sell Pressure)
                # ~0.0 = Doji (Indecision/Churn)
                price_range = (df['high'] - df['low'])
                body = (df['close'] - df['open'])
                efficiency = body / (price_range + 1e-9)
                
                # Relative Volume (Log-space to dampen extreme outliers)
                vol_ma = df['volume'].rolling(20).mean()
                volume_ratio = df['volume'] / (vol_ma + 1e-9)
                log_volume_ratio = np.log1p(volume_ratio)
                
                # Signal: Efficiency amplified by Volume
                # Efficient moves on high volume are the strongest causal events
                volume_signal = efficiency * log_volume_ratio
                
                # Adaptive Volatility Filter (AVF) integration
                returns = df['close'].pct_change()
                volatility = returns.rolling(20).std()
                vol_rank = volatility.rolling(100).rank(pct=True)
                
                # Allow signal if:
                # 1. Volatility is healthy (> 10th percentile)
                # 2. OR Volume is massive (> 3x average) - distinct event
                avf_mask = (vol_rank > 0.1) | (volume_ratio > 3.0)
                
                volume_signal = volume_signal * avf_mask.astype(float)
                
                # Z-score normalization
                volume_signal = (volume_signal - volume_signal.rolling(50).mean()) / \
                               (volume_signal.rolling(50).std() + 1e-9)
                
                # Momentum confirmation (optional but helpful for persistence)
                # Is the signal separating from its recent average?
                sig_mom = volume_signal.diff()
                final_sig = (volume_signal + 0.5 * sig_mom).fillna(0)
                
                if self.verbose and final_sig.std() < 1e-6:
                     tprint_warning(f"      ⚠️ Volume signal low variance: raw_std={volume_signal.std():.6f}")
                     
                return final_sig
            
            return None
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Volume signal extraction failed: {e}")
            return None
    
    def _extract_volatility_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """Extract volatility specialist signal focusing on volatility changes."""
        if self.verbose:
            tprint_info("📈 Extracting volatility specialist signal")
        try:
            if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
                # High-Frequency Realized Volatility Measure (Intraday)
                # Parkinson / Garman-Klass proxy using High-Low
                hl_range = (df['high'] - df['low']) / df['close']
                
                # Realized Volatility (Returns based)
                returns = df['close'].pct_change()
                realized_vol = returns.rolling(20).std()
                
                # Volatility Change (Delta Vol)
                # We care about expanding or contracting volatility
                vol_change = realized_vol.diff()
                
                # Range-based surprise
                range_ma = hl_range.rolling(20).mean()
                range_surprise = (hl_range - range_ma) / (range_ma + 1e-9)
                
                # Combined Signal: Volatility Expansion + Intraday Range Expansion
                # KEY CHANGE: Multiply by direction (sign of returns) to make it predictive
                # High Vol + Up = Bullish thrust
                # High Vol + Down = Bearish crash
                raw_mag = vol_change + (range_surprise * realized_vol)
                direction = np.sign(returns)
                
                volatility_signal = raw_mag * direction
                
                # Normalize
                volatility_signal = (volatility_signal - volatility_signal.rolling(50).mean()) / \
                                   (volatility_signal.rolling(50).std() + 1e-9)
                
                if self.verbose and volatility_signal.std() < 1e-6:
                     tprint_warning(f"      ⚠️ Volatility signal low variance: raw_change_std={vol_change.std():.6f}")

                return volatility_signal.fillna(0)
            
            return None
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Volatility signal extraction failed: {e}")
            return None
    
    def _extract_information_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract 'Information' signal using Price Action & Candle Ratios.
        Replaces dead VPIN metric with Microstructure/Price Action features.
        """
        if self.verbose:
            tprint_info("📊 Extracting information (price action) signal")
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required_cols):
                # 1. Candle Ratios
                price_range = df['high'] - df['low']
                body = np.abs(df['close'] - df['open'])
                
                # Trendiness: Body / Range (1.0 = Marubozu, 0.0 = Doji)
                trend_efficiency = body / (price_range + 1e-9)
                
                # Direction
                direction = np.sign(df['close'] - df['open'])
                
                # 2. Wick Rejection (Upper/Lower shadows)
                upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
                lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
                
                # Wick asymmetry (Positive = Selling Pressure/Rejection at top, Negative = Buying at bottom)
                # We interpret "Information" as "Informed Directional Flow"
                # Large Upper Wick = Rejection (Bearish Info)
                # Large Lower Wick = Support (Bullish Info)
                wick_balance = (lower_shadow - upper_shadow) / (price_range + 1e-9)
                
                # 3. Volume Verification
                # Does volume confirm the move?
                vol_rel = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
                
                # Combined Price Action Signal
                # Strong body + Volume = Trend
                # Strong Wick + Volume = Rejection (Reversal)
                
                # Signal is directional: Positive = Bullish Info, Negative = Bearish Info
                pa_signal = (trend_efficiency * direction * vol_rel) + (wick_balance * vol_rel)
                
                # Normalize
                information_signal = (pa_signal - pa_signal.rolling(50).mean()) / \
                                  (pa_signal.rolling(50).std() + 1e-9)
                
                return information_signal.fillna(0)
            
            return None
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Information signal extraction failed: {e}")
            return None
    
    def transform_to_spectral(
        self,
        specialist_signals: Dict[str, pd.Series],
        wavelet_engine
    ) -> Dict[str, np.ndarray]:
        """
        Transform specialists to spectral domain using wavelet decomposition.
        
        Args:
            specialist_signals: Raw specialist time series
            wavelet_engine: Wavelet decomposition engine
            
        Returns:
            Dictionary with spectral components
        """
        try:
            if self.verbose:
                tprint_info("🌊 Transforming specialists to spectral domain...")
            
            spectral_components = wavelet_engine.decompose_all_specialists(specialist_signals)
            
            if self.verbose:
                tprint_success(f"   ✅ Spectral transformation complete:")
                tprint_info(f"      - Specialists transformed: {len(specialist_signals)}")
                tprint_info(f"      - Spectral components: {len(spectral_components)}")
                tprint_info(f"      - Scales per specialist: 5")
            
            return spectral_components
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Spectral transformation failed: {e}")
            return {}
    
    def get_specialist_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all priority specialists."""
        if self.verbose:
            tprint_info("📋 Retrieving specialist metadata")
        return {
            name: self.specialist_descriptions.get(name, {})
            for name in self.priority_specialists
        }
    
    def validate_specialist_signals(
        self,
        specialist_signals: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate extracted specialist signals.
        
        Args:
            specialist_signals: Extracted specialist signals
            
        Returns:
            Validation results for each specialist
        """
        try:
            validation_results = {}
            
            for specialist_name, signal in specialist_signals.items():
                if len(signal) == 0:
                    validation_results[specialist_name] = {
                        'valid': False,
                        'error': 'Empty signal'
                    }
                    continue
                
                # Basic validation checks
                nan_count = signal.isna().sum()
                zero_count = (signal == 0).sum()
                signal_std = signal.std()
                signal_mean = signal.mean()
                
                validation_results[specialist_name] = {
                    'valid': True,
                    'length': len(signal),
                    'nan_count': nan_count,
                    'zero_count': zero_count,
                    'mean': signal_mean,
                    'std': signal_std,
                    'quality_score': self._calculate_quality_score(signal)
                }
            
            return validation_results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Signal validation failed: {e}")
            return {}
    
    def _calculate_quality_score(self, signal: pd.Series) -> float:
        """Calculate quality score for a specialist signal."""
        if self.verbose:
            tprint_info("⭐ Calculating quality score")
        try:
            # Remove NaN and zeros
            clean_signal = signal.dropna()
            clean_signal = clean_signal[clean_signal != 0]
            
            if len(clean_signal) < 100:
                return 0.0
            
            # Quality metrics
            variance = clean_signal.var()
            autocorr = clean_signal.autocorr(lag=1)
            
            # Higher variance and moderate autocorrelation = better quality
            variance_score = min(variance / 1.0, 1.0)  # Normalize variance
            autocorr_score = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.0
            
            quality_score = 0.6 * variance_score + 0.4 * autocorr_score
            return quality_score
            
        except Exception:
            return 0.0

    def _extract_cusum_break_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract CUSUM structural break signal.
        Detects shifts in the mean of price changes.
        """
        if self.verbose:
            tprint_info("⚡ Extracting CUSUM break specialist signal")
        try:
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                
                # CUSUM Calculation
                # S[t] = max(0, S[t-1] + y[t] - k) for positive shift
                # We use a simplified two-sided cumulative deviation
                
                mean_ret = returns.rolling(100).mean()
                std_ret = returns.rolling(100).std()
                
                # Standardized deviation
                z = (returns - mean_ret) / (std_ret + 1e-9)
                
                # Cumulative Sum of Deviations
                # We want to detect *trends* in deviation -> persistent shift
                cusum = z.cumsum()
                
                # Detrend: Remove linear trend to find breaks in the trend
                # Simple way: CUSUM - Moving Average of CUSUM
                break_signal = (cusum - cusum.rolling(50).mean()) / (cusum.rolling(50).std() + 1e-9)
                
                # Re-index to match df
                return break_signal.reindex(df.index).fillna(0)
            
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ CUSUM signal extraction failed: {e}")
            return None

    def _extract_entropy_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract Shannon Entropy signal.
        Measures information content / unpredictability.
        """
        if self.verbose:
            tprint_info("🧩 Extracting entropy specialist signal")
        try:
            from scipy.stats import entropy
            
            if 'close' in df.columns:
                # Rolling Entropy on Returns Distribution
                returns = df['close'].pct_change().dropna()
                
                def rolling_entropy(x):
                    # Discretize into bins
                    counts, _ = np.histogram(x, bins=10, density=True)
                    # Compute Shannon entropy
                    return entropy(counts + 1e-9)
                
                # Calculate rolling entropy
                # Window should be large enough to form a distribution (e.g., 50)
                entropy_sig = returns.rolling(50).apply(rolling_entropy, raw=True)
                
                # Normalize
                # High entropy = High unpredictability
                entropy_signal = (entropy_sig - entropy_sig.rolling(100).mean()) / (entropy_sig.rolling(100).std() + 1e-9)
                
                return entropy_signal.reindex(df.index).fillna(0)
            
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Entropy signal extraction failed: {e}")
            return None

    def _extract_tick_rule_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract Tick Rule proxy signal (Aggressor Flow).
        Approximates net buy/sell pressure.
        """
        if self.verbose:
            tprint_info("🌊 Extracting tick rule specialist signal")
        try:
            required = ['close', 'open', 'volume']
            if all(c in df.columns for c in required):
                # Tick Rule Proxy:
                # Close > Open -> Buy (1)
                # Close < Open -> Sell (-1)
                # Close == Open -> 0 (or prev)
                
                direction = np.sign(df['close'] - df['open'])
                
                # Weight by volume
                signed_volume = direction * df['volume']
                
                # Accumulate (Cumulative Volume Delta - CVD proxy)
                cvd = signed_volume.cumsum()
                
                # Signal is the *divergence* or *acceleration* of CVD
                # We use local trend of CVD
                tick_signal = (cvd - cvd.rolling(20).mean()) / (cvd.rolling(20).std() + 1e-9)
                
                return tick_signal.fillna(0)
            
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Tick rule signal extraction failed: {e}")
            return None

    def _extract_fractal_efficiency_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract Fractal Efficiency (Kaufman) signal.
        Measures trend cleanliness/linearity.
        Logic: Sign(Return) * (Net_Move / Total_Path_Length)
        """
        if self.verbose:
            tprint_info("📏 Extracting fractal efficiency specialist signal")
        try:
            required = ['close']
            if all(c in df.columns for c in required):
                # Fractal Efficiency Ratio (Kaufman)
                # Optimized vectorized implementation using Pandas rolling
                window = 10
                
                # Numba-friendly logic simulation via optimized Pandas
                diffs = df['close'].diff()
                abs_diffs = diffs.abs()
                
                # Efficiency = |Change(N)| / Sum(|Change(1)|..|Change(N)|)
                net_change = df['close'].diff(window)
                path_length = abs_diffs.rolling(window).sum()
                
                # Avoid division by zero
                efficiency = net_change.abs() / (path_length + 1e-9)
                
                # Make Directional: Multiply by sign of the net change
                # Up Trend Efficient = +ve
                # Down Trend Efficient = -ve
                # Choppy/Noise = ~0
                direction = np.sign(net_change)
                directional_efficiency = efficiency * direction
                
                # Normalize (Z-Score)
                # We want to detect Anomalous Efficiency (Pure Trends)
                fractal_signal = (directional_efficiency - directional_efficiency.rolling(50).mean()) / \
                                (directional_efficiency.rolling(50).std() + 1e-9)
                
                return fractal_signal.fillna(0)
            
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Fractal efficiency extraction failed: {e}")
            return None

    def _extract_liquidity_shock_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract Liquidity Shock signal (Amihud Proxy).
        Measures Price Impact per Unit of Volume.
        """
        if self.verbose:
            tprint_info("💧 Extracting liquidity shock specialist signal")
        try:
            required = ['close', 'volume']
            if all(c in df.columns for c in required):
                # Amihud Illiquidity: |Return| / (Price * Volume)
                # We want "Price Ease" -> Directional Impact
                
                returns = df['close'].pct_change()
                
                # Volume in dollars approx (Volume * Price) or just Volume if FX/Crypto
                # Using Dollar Volume is safer for comparing across price levels
                dollar_volume = df['volume'] * df['close']
                
                # Illiquidity: How much price moves per dollar traded
                # High = Illiquid (Fragile)
                # Low = Liquid (Robust)
                illiquidity = returns.abs() / (dollar_volume + 1e-9)
                
                # We want to detect SHOCKS in illiquidity that coincide with direction
                # i.e., Price moving easily (thin liquidity) in a direction
                
                # Directional Liquidity Shock = Sign(Return) * Illiquidity
                liq_shock_raw = np.sign(returns) * illiquidity
                
                # Log-space handling might be needed if illiquidity spans orders of magnitude?
                # Usually standardizing rolling window handles it.
                
                # Normalize
                liq_shock_signal = (liq_shock_raw - liq_shock_raw.rolling(50).mean()) / \
                                  (liq_shock_raw.rolling(50).std() + 1e-9)
                                  
                return liq_shock_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Liquidity shock extraction failed: {e}")
            return None

    def _extract_gap_signal(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Extract Exogenous Gap signal.
        Measures Overnight/Weekend information injection (Open - PrevClose).
        """
        if self.verbose:
            tprint_info("🕳️ Extracting gap specialist signal")
        try:
            required = ['open', 'close']
            if all(c in df.columns for c in required):
                # Gap = Open - Prev Close
                prev_close = df['close'].shift(1)
                gap = df['open'] - prev_close
                
                # Normalize by recent volatility (Standardized Gap)
                # A 1% gap in low vol is huge; in high vol is noise.
                returns = df['close'].pct_change()
                volatility = returns.rolling(20).std()
                
                # Standardized Gap (Sigma)
                gap_sigma = gap / (prev_close * volatility + 1e-9) # Gap % / Vol % approx
                
                # We can also just z-score the raw gap if we want local context
                # But Vol-adjusted is more physically meaningful (Exogenous Shock Magnitude)
                
                # Let's z-score the sigma to fit the distribution
                gap_signal = (gap_sigma - gap_sigma.rolling(50).mean()) / \
                            (gap_sigma.rolling(50).std() + 1e-9)
                            
                return gap_signal.fillna(0)
            return None
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Gap signal extraction failed: {e}")
            return None


# Convenience functions for quick usage
def quick_spectral_transformation(
    df: pd.DataFrame,
    wavelet_engine,
    priority_specialists: List[str] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Quick spectral transformation for market data."""
    if verbose:
        tprint_info("🚀 Quick spectral transformation")
    spectral_specialists = SpectralSpecialists(priority_specialists, verbose=verbose)
    
    # Extract signals
    specialist_signals = spectral_specialists.extract_specialist_signals(df)
    
    # Transform to spectral
    spectral_components = spectral_specialists.transform_to_spectral(
        specialist_signals, wavelet_engine
    )
    
    return spectral_components



