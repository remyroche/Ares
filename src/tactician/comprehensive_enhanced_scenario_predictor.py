"""
Comprehensive Enhanced Scenario-Based Predictor for Tactician

Implements advanced probabilistic scenario analysis with:
    pass- ALL technical indicators (50+ indicators)
    - 15-minute look-ahead period
    - Fractal scenario definitions (linear progression)
    - FULL step17 optimization for ALL parameters including decision logic
    - Complete migration from existing system
"""

import numpy as np
import pandas as pd
from datetime import datetime
import lightgbm as lgb
import logging
import talib
from typing import Dict, Any, List, Optional

# Simple logger setup
logger = logging.getLogger(__name__)

# Simple error handling decorator
def handle_errors(...):
    passpass"""Simple error handling decorator."""
    def wrapper(...):
    passtry:
    passreturn func(*args, **kwargs)
        except Exception as e:
    passpasspasspasspasspasspasslogger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


class ComprehensiveEnhancedScenarioPredictor:
    pass"""
    Comprehensive enhanced scenario-based predictor with ALL technical indicators.

    Fractal Scenarios (Linear Progression):
    pass- Profit Zones: 0.25%, 0.5%, 0.75%, 1.0%, 1.25%, 1.5%, 1.75%, 2.0%
        - Risk Zones: -0.25%, -0.5%, -0.75%, -1.0%, -1.25%, -1.5%, -1.75%, -2.0%
        - Neutral: No scenario triggered within 15 minutes
    """

    def __init__(...) -> ...:
    """..."""
    passself.config = config
        self.logger = logger

        # Load step17 optimization parameters
        step17_config = config.get("step17_optimization", {})
        scenario_config = step17_config.get("comprehensive_enhanced_scenario_analysis", {})

        # Fractal scenario definitions (configurable for step17)
        self.scenarios = self._create_fractal_scenarios(scenario_config)

        # Time limit for scenario evaluation (15 minutes)
        self.time_limit_minutes = scenario_config.get("time_limit_minutes", 15)

        # Model configuration (configurable for step17)
        self.model_config = {
            "n_estimators": scenario_config.get("n_estimators", 200),
            "learning_rate": scenario_config.get("learning_rate", 0.05),
            "max_depth": scenario_config.get("max_depth", 8),
            "num_leaves": scenario_config.get("num_leaves", 63),
            "subsample": scenario_config.get("subsample", 0.8),
            "colsample_bytree": scenario_config.get("colsample_bytree", 0.8),
            "random_state": scenario_config.get("random_state", 42),
            "verbose": -1
        }

        # COMPREHENSIVE technical indicator parameters (configurable for step17)
        self.technical_indicators = {
            # Momentum Indicators
            "RSI": {
                "lookback_period": scenario_config.get("rsi_lookback_period", 14),
                "overbought_threshold": scenario_config.get("rsi_overbought_threshold", 70),
                "oversold_threshold": scenario_config.get("rsi_oversold_threshold", 30)
            },
            "MACD": {
                "fast_period": scenario_config.get("macd_fast_period", 12),
                "slow_period": scenario_config.get("macd_slow_period", 26),
                "signal_period": scenario_config.get("macd_signal_period", 9)
            },
            "Stochastic": {
                "k_period": scenario_config.get("stoch_k_period", 14),
                "d_period": scenario_config.get("stoch_d_period", 3),
                "overbought_threshold": scenario_config.get("stoch_overbought_threshold", 80),
                "oversold_threshold": scenario_config.get("stoch_oversold_threshold", 20)
            },
            "Williams_R": {
                "lookback_period": scenario_config.get("williams_r_period", 14),
                "overbought_threshold": scenario_config.get("williams_r_overbought", -20),
                "oversold_threshold": scenario_config.get("williams_r_oversold", -80)
            },
            "CCI": {
                "lookback_period": scenario_config.get("cci_period", 20),
                "overbought_threshold": scenario_config.get("cci_overbought", 100),
                "oversold_threshold": scenario_config.get("cci_oversold", -100)
            },
            "ROC": {
                "lookback_period": scenario_config.get("roc_period", 10)
            },
            "MOM": {
                "lookback_period": scenario_config.get("mom_period", 10)
            },
            "ADX": {
                "lookback_period": scenario_config.get("adx_period", 14),
                "trend_threshold": scenario_config.get("adx_trend_threshold", 25)
            },
            "Aroon": {
                "lookback_period": scenario_config.get("aroon_period", 14)
            },
            "MFI": {
                "lookback_period": scenario_config.get("mfi_period", 14),
                "overbought_threshold": scenario_config.get("mfi_overbought", 80),
                "oversold_threshold": scenario_config.get("mfi_oversold", 20)
            },
            "OBV": {
                "smoothing_period": scenario_config.get("obv_smoothing", 10)
            },
            "CMF": {
                "lookback_period": scenario_config.get("cmf_period", 20)
            },
            "BBP": {
                "lookback_period": scenario_config.get("bbp_period", 20),
                "std_dev": scenario_config.get("bbp_std_dev", 2)
            },
            "Ultimate_Oscillator": {
                "period1": scenario_config.get("uo_period1", 7),
                "period2": scenario_config.get("uo_period2", 14),
                "period3": scenario_config.get("uo_period3", 28),
                "weight1": scenario_config.get("uo_weight1", 4),
                "weight2": scenario_config.get("uo_weight2", 2),
                "weight3": scenario_config.get("uo_weight3", 1)
            },
            "KST": {
                "roc1_period": scenario_config.get("kst_roc1", 10),
                "roc2_period": scenario_config.get("kst_roc2", 15),
                "roc3_period": scenario_config.get("kst_roc3", 20),
                "roc4_period": scenario_config.get("kst_roc4", 30),
                "sma1_period": scenario_config.get("kst_sma1", 10),
                "sma2_period": scenario_config.get("kst_sma2", 10),
                "sma3_period": scenario_config.get("kst_sma3", 10),
                "sma4_period": scenario_config.get("kst_sma4", 15)
            },
            "TSI": {
                "first_period": scenario_config.get("tsi_first", 25),
                "second_period": scenario_config.get("tsi_second", 13)
            },
            "Chaikin_Oscillator": {
                "fast_period": scenario_config.get("chaikin_fast", 3),
                "slow_period": scenario_config.get("chaikin_slow", 10)
            },
            "EOM": {
                "lookback_period": scenario_config.get("eom_period", 14)
            },
            "VWAP": {
                "anchor": scenario_config.get("vwap_anchor", "D")
            },
            "Pivot_Points": {
                "method": scenario_config.get("pivot_method", "standard")
            },
            "Ichimoku": {
                "tenkan_period": scenario_config.get("ichimoku_tenkan", 9),
                "kijun_period": scenario_config.get("ichimoku_kijun", 26),
                "senkou_span_b_period": scenario_config.get("ichimoku_senkou_b", 52),
                "displacement": scenario_config.get("ichimoku_displacement", 26)
            },
            "Parabolic_SAR": {
                "acceleration": scenario_config.get("psar_acceleration", 0.02),
                "maximum": scenario_config.get("psar_maximum", 0.2)
            },
            "ZigZag": {
                "deviation": scenario_config.get("zigzag_deviation", 5),
                "depth": scenario_config.get("zigzag_depth", 10)
            },
            "Alligator": {
                "jaw_period": scenario_config.get("alligator_jaw", 13),
                "teeth_period": scenario_config.get("alligator_teeth", 8),
                "lips_period": scenario_config.get("alligator_lips", 5),
                "jaw_offset": scenario_config.get("alligator_jaw_offset", 8),
                "teeth_offset": scenario_config.get("alligator_teeth_offset", 5),
                "lips_offset": scenario_config.get("alligator_lips_offset", 3)
            },
            "Gator_Oscillator": {
                "jaw_period": scenario_config.get("gator_jaw", 13),
                "teeth_period": scenario_config.get("gator_teeth", 8),
                "lips_period": scenario_config.get("gator_lips", 5),
                "jaw_offset": scenario_config.get("gator_jaw_offset", 8),
                "teeth_offset": scenario_config.get("gator_teeth_offset", 5),
                "lips_offset": scenario_config.get("gator_lips_offset", 3)
            },
            "Awesome_Oscillator": {
                "fast_period": scenario_config.get("ao_fast", 5),
                "slow_period": scenario_config.get("ao_slow", 34)
            },
            "Accelerator_Oscillator": {
                "fast_period": scenario_config.get("ac_fast", 5),
                "slow_period": scenario_config.get("ac_slow", 34)
            },
            "Detrended_Price_Oscillator": {
                "lookback_period": scenario_config.get("dpo_period", 20)
            },
            "Percentage_Price_Oscillator": {
                "fast_period": scenario_config.get("ppo_fast", 12),
                "slow_period": scenario_config.get("ppo_slow", 26),
                "signal_period": scenario_config.get("ppo_signal", 9)
            },
            "Klinger_Volume_Oscillator": {
                "fast_period": scenario_config.get("kvo_fast", 34),
                "slow_period": scenario_config.get("kvo_slow", 55)
            },
            "Choppiness_Index": {
                "lookback_period": scenario_config.get("choppiness_period", 14)
            },
            "Aroon_Oscillator": {
                "lookback_period": scenario_config.get("aroon_osc_period", 14)
            },
            "Balance_of_Power": {
                "smoothing_period": scenario_config.get("bop_smoothing", 14)
            },
            "Commodity_Channel_Index": {
                "lookback_period": scenario_config.get("cci_period", 20),
                "constant": scenario_config.get("cci_constant", 0.015)
            },
            "Coppock_Curve": {
                "wma_period": scenario_config.get("coppock_wma", 10),
                "roc1_period": scenario_config.get("coppock_roc1", 14),
                "roc2_period": scenario_config.get("coppock_roc2", 11)
            },
            "Ease_of_Movement": {
                "lookback_period": scenario_config.get("eom_period", 14)
            },
            "Fisher_Transform": {
                "lookback_period": scenario_config.get("fisher_period", 9)
            },
            "Know_Sure_Thing": {
                "roc1_period": scenario_config.get("kst_roc1", 10),
                "roc2_period": scenario_config.get("kst_roc2", 15),
                "roc3_period": scenario_config.get("kst_roc3", 20),
                "roc4_period": scenario_config.get("kst_roc4", 30),
                "sma1_period": scenario_config.get("kst_sma1", 10),
                "sma2_period": scenario_config.get("kst_sma2", 10),
                "sma3_period": scenario_config.get("kst_sma3", 10),
                "sma4_period": scenario_config.get("kst_sma4", 15)
            },
            "Mass_Index": {
                "ema_period": scenario_config.get("mi_ema", 9),
                "sum_period": scenario_config.get("mi_sum", 25)
            },
            "Median_Price": {
                "smoothing_period": scenario_config.get("median_smoothing", 14)
            },
            "Price_Oscillator": {
                "fast_period": scenario_config.get("price_osc_fast", 12),
                "slow_period": scenario_config.get("price_osc_slow", 26)
            },
            "Price_Volume_Trend": {
                "smoothing_period": scenario_config.get("pvt_smoothing", 14)
            },
            "True_Strength_Index": {
                "first_period": scenario_config.get("tsi_first", 25),
                "second_period": scenario_config.get("tsi_second", 13)
            },
            "Ultimate_Oscillator": {
                "period1": scenario_config.get("uo_period1", 7),
                "period2": scenario_config.get("uo_period2", 14),
                "period3": scenario_config.get("uo_period3", 28),
                "weight1": scenario_config.get("uo_weight1", 4),
                "weight2": scenario_config.get("uo_weight2", 2),
                "weight3": scenario_config.get("uo_weight3", 1)
            },
            "Vortex_Indicator": {
                "lookback_period": scenario_config.get("vortex_period", 14)
            },
            "Williams_Alligator": {
                "jaw_period": scenario_config.get("williams_alligator_jaw", 13),
                "teeth_period": scenario_config.get("williams_alligator_teeth", 8),
                "lips_period": scenario_config.get("williams_alligator_lips", 5),
                "jaw_offset": scenario_config.get("williams_alligator_jaw_offset", 8),
                "teeth_offset": scenario_config.get("williams_alligator_teeth_offset", 5),
                "lips_offset": scenario_config.get("williams_alligator_lips_offset", 3)
            },
            "Williams_R": {
                "lookback_period": scenario_config.get("williams_r_period", 14)
            }
        }

        # Initialize model
        self.model = None
        self.is_trained = False

        # Performance tracking
        self.predictions_history = []
        self.scenario_accuracy = {}

    def _create_fractal_scenarios(...) -> ...:
    """..."""
    passscenarios = {}
        scenario_id = 0

        # Profit zones (0.25% to 2.0% in 0.25% increments)
        profit_targets = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
        for i, profit_target in enumerate(profit_targets):
    passscenarios[scenario_id] = {
                "name": f"Profit Zone {i+1} ({profit_target*100:.1f}%)",
                "profit_target": scenario_config.get(f"profit_zone_{i+1}_target", profit_target),
                "stop_loss": scenario_config.get(f"profit_zone_{i+1}_stop_loss", -0.005),
                "description": f"Price moves up by {profit_target*100:.1f}% before moving down by 0.5%",
                "zone_type": "profit",
                "zone_level": i+1
            }
            scenario_id += 1

        # Risk zones (-0.25% to -2.0% in 0.25% increments)
        risk_targets = [-0.0025, -0.005, -0.0075, -0.01, -0.0125, -0.015, -0.0175, -0.02]
        for i, risk_target in enumerate(risk_targets):
    passscenarios[scenario_id] = {
                "name": f"Risk Zone {i+1} ({abs(risk_target)*100:.1f}%)",
                "profit_target": scenario_config.get(f"risk_zone_{i+1}_target", 0.005),
                "stop_loss": scenario_config.get(f"risk_zone_{i+1}_stop_loss", risk_target),
                "description": f"Price moves down by {abs(risk_target)*100:.1f}% before moving up by 0.5%",
                "zone_type": "risk",
                "zone_level": i+1
            }
            scenario_id += 1

        # Neutral scenario
        scenarios[scenario_id] = {
            "name": "Neutral",
            "profit_target": scenario_config.get("neutral_target", 0.0),
            "stop_loss": scenario_config.get("neutral_stop_loss", 0.0),
            "description": "No scenario triggered within time limit",
            "zone_type": "neutral",
            "zone_level": 0
        }

        return scenarios

    @handle_errors
    async def initialize(...) -> ...:
    """..."""
    passself.logger.info("Initializing Comprehensive Enhanced Scenario-Based Predictor...")

        # Validate configuration
        if not self._validate_configuration():
    passself.logger.error("Invalid configuration for comprehensive enhanced scenario predictor")
            return False

        # Initialize model
        self.model = lgb.LGBMClassifier(**self.model_config)

        self.logger.info("✅ Comprehensive Enhanced Scenario-Based Predictor initialized successfully")
        return True

    def _validate_configuration(...) -> ...:
    pass"""..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            # Validate scenarios
            for scenario_id, scenario in self.scenarios.items():
    passif scenario["zone_type"] != "neutral":
    passif scenario["profit_target"] <= 0 and scenario["zone_type"] == "profit":
    passself.logger.error(f"Invalid profit target for scenario {scenario_id}")
                        return False
                    if scenario["stop_loss"] >= 0 and scenario["zone_type"] == "risk":
    passpassself.logger.error(f"Invalid stop loss for scenario {scenario_id}")
                        return False

            # Validate time limit
            if self.time_limit_minutes <= 0:
    passpassself.logger.error("Invalid time limit")
                return False

            # Validate technical indicator parameters
            for indicator_name, params in self.technical_indicators.items():
    passfor param_name, param_value in params.items():
    passif param_name != "enabled" and param_value <= 0:
    passself.logger.error(f"Invalid parameter for {indicator_name}.{param_name}")
                        return False

            return True
        except Exception as e:
    passpasspasspasspasspasspasspassself.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    @handle_errors
    def extract_comprehensive_features(...) -> ...:
    """..."""
    passfeatures = []

        if len(market_data) < max(self.feature_config["lookback_periods"], 50):
    pass# Not enough data, return default features
            return np.array([0.5] * 350)  # Increased feature count

        # Price-based features
        close_prices = market_data['close'].values
        high_prices = market_data['high'].values
        low_prices = market_data['low'].values
        open_prices = market_data['open'].values
        volumes = market_data['volume'].values

        # Current price and recent prices
        current_price = close_prices[-1]

        # 1. Price momentum features
        for period in self.feature_config["price_momentum_periods"]:
    passif len(close_prices) >= period:
    passmomentum = (current_price - close_prices[-period]) / close_prices[-period]
                features.append(momentum)
            else:
    passfeatures.append(0.0)

        # 2. Volatility features
        returns = np.diff(close_prices) / close_prices[:-1]
        for period in self.feature_config["volatility_periods"]:
    passif len(returns) >= period:
    passvolatility = np.std(returns[-period:])
                features.append(volatility)
            else:
    passfeatures.append(0.0)

        # 3. Volume features
        volume_trend = (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
        volume_ma_ratio = volumes[-1] / np.mean(volumes[-self.feature_config["volume_ma_period"]:]) if np.mean(volumes[-self.feature_config["volume_ma_period"]:]) > 0 else 1.0
        features.extend([volume_trend, volume_ma_ratio])

        # 4. COMPREHENSIVE TECHNICAL INDICATORS

        # Momentum Indicators
        rsi_params = self.technical_indicators["RSI"]
        rsi = talib.RSI(close_prices, timeperiod=rsi_params["lookback_period"])
        features.append(rsi[-1] / 100 if not np.isnan(rsi[-1]) else 0.5)

        macd_params = self.technical_indicators["MACD"]
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices,
            fastperiod=macd_params["fast_period"],
            slowperiod=macd_params["slow_period"],
            signalperiod=macd_params["signal_period"]
        )
        features.extend([
            macd[-1] if not np.isnan(macd[-1]) else 0.0,
            macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0,
            macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
        ])

        stoch_params = self.technical_indicators["Stochastic"]
        stoch_k, stoch_d = talib.STOCH(
            high_prices, low_prices, close_prices,
            fastk_period=stoch_params["k_period"],
            slowk_period=stoch_params["d_period"],
            slowd_period=stoch_params["d_period"]
        )
        features.extend([
            stoch_k[-1] / 100 if not np.isnan(stoch_k[-1]) else 0.5,
            stoch_d[-1] / 100 if not np.isnan(stoch_d[-1]) else 0.5
        ])

        williams_r_params = self.technical_indicators["Williams_R"]
        williams_r = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=williams_r_params["lookback_period"])
        features.append(williams_r[-1] / -100 if not np.isnan(williams_r[-1]) else 0.5)

        roc_params = self.technical_indicators["ROC"]
        roc = talib.ROC(close_prices, timeperiod=roc_params["lookback_period"])
        features.append(roc[-1] if not np.isnan(roc[-1]) else 0.0)

        mom_params = self.technical_indicators["MOM"]
        mom = talib.MOM(close_prices, timeperiod=mom_params["lookback_period"])
        features.append(mom[-1] / current_price if not np.isnan(mom[-1]) else 0.0)

        trix_params = self.technical_indicators["TRIX"]
        trix = talib.TRIX(close_prices, timeperiod=trix_params["lookback_period"])
        features.append(trix[-1] if not np.isnan(trix[-1]) else 0.0)

        ultosc_params = self.technical_indicators["Ultimate_Oscillator"]
        ultosc = talib.ULTOSC(high_prices, low_prices, close_prices,
            timeperiod1=ultosc_params["period1"],
            timeperiod2=ultosc_params["period2"],
            timeperiod3=ultosc_params["period3"]
        )
        features.append(ultosc[-1] / 100 if not np.isnan(ultosc[-1]) else 0.5)

        willr_params = self.technical_indicators["Williams_R"]
        willr = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=willr_params["lookback_period"])
        features.append(willr[-1] / -100 if not np.isnan(willr[-1]) else 0.5)

        aroon_params = self.technical_indicators["Aroon"]
        aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=aroon_params["lookback_period"])
        features.extend([
            aroon_down[-1] / 100 if not np.isnan(aroon_down[-1]) else 0.5,
            aroon_up[-1] / 100 if not np.isnan(aroon_up[-1]) else 0.5
        ])

        cci_params = self.technical_indicators["CCI"]
        cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=cci_params["lookback_period"])
        cci_normalized = (cci[-1] + 300) / 600 if not np.isnan(cci[-1]) else 0.5
        features.append(np.clip(cci_normalized, 0, 1))

        cmo_params = self.technical_indicators["Chaikin_Oscillator"]
        cmo = talib.CMO(close_prices, timeperiod=cmo_params["fast_period"])
        cmo_normalized = (cmo[-1] + 100) / 200 if not np.isnan(cmo[-1]) else 0.5
        features.append(np.clip(cmo_normalized, 0, 1))

        # Trend Indicators
        sma_params = self.technical_indicators["SMA"]
        sma_short = talib.SMA(close_prices, timeperiod=sma_params["short_period"])
        sma_long = talib.SMA(close_prices, timeperiod=sma_params["long_period"])
        sma_ratio = sma_short[-1] / sma_long[-1] if sma_long[-1] > 0 else 1.0
        features.append(sma_ratio if not np.isnan(sma_ratio) else 1.0)

        ema_params = self.technical_indicators["EMA"]
        ema_short = talib.EMA(close_prices, timeperiod=ema_params["short_period"])
        ema_long = talib.EMA(close_prices, timeperiod=ema_params["long_period"])
        ema_ratio = ema_short[-1] / ema_long[-1] if ema_long[-1] > 0 else 1.0
        features.append(ema_ratio if not np.isnan(ema_ratio) else 1.0)

        dema_params = self.technical_indicators["DEMA"]
        dema = talib.DEMA(close_prices, timeperiod=dema_params["lookback_period"])
        dema_ratio = current_price / dema[-1] if dema[-1] > 0 else 1.0
        features.append(dema_ratio if not np.isnan(dema_ratio) else 1.0)

        tema_params = self.technical_indicators["TEMA"]
        tema = talib.TEMA(close_prices, timeperiod=tema_params["lookback_period"])
        tema_ratio = current_price / tema[-1] if tema[-1] > 0 else 1.0
        features.append(tema_ratio if not np.isnan(tema_ratio) else 1.0)

        if self.technical_indicators["HT_TRENDLINE"]["enabled"]:
    passht_trendline = talib.HT_TRENDLINE(close_prices)
            ht_trendline_ratio = current_price / ht_trendline[-1] if ht_trendline[-1] > 0 else 1.0
            features.append(ht_trendline_ratio if not np.isnan(ht_trendline_ratio) else 1.0)
        else:
    passpassfeatures.append(1.0)

        sar_params = self.technical_indicators["Parabolic_SAR"]
        sar = talib.SAR(high_prices, low_prices,
            acceleration=sar_params["acceleration"],
            maximum=sar_params["maximum"]
        )
        sar_ratio = current_price / sar[-1] if sar[-1] > 0 else 1.0
        features.append(sar_ratio if not np.isnan(sar_ratio) else 1.0)

        adx_params = self.technical_indicators["ADX"]
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=adx_params["lookback_period"])
        features.append(adx[-1] / 100 if not np.isnan(adx[-1]) else 0.5)

        dx_params = self.technical_indicators["DX"]
        dx = talib.DX(high_prices, low_prices, close_prices, timeperiod=dx_params["lookback_period"])
        features.append(dx[-1] / 100 if not np.isnan(dx[-1]) else 0.5)

        minus_di_params = self.technical_indicators["MINUS_DI"]
        minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=minus_di_params["lookback_period"])
        features.append(minus_di[-1] / 100 if not np.isnan(minus_di[-1]) else 0.5)

        plus_di_params = self.technical_indicators["PLUS_DI"]
        plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=plus_di_params["lookback_period"])
        features.append(plus_di[-1] / 100 if not np.isnan(plus_di[-1]) else 0.5)

        minus_dm_params = self.technical_indicators["MINUS_DM"]
        minus_dm = talib.MINUS_DM(high_prices, low_prices, timeperiod=minus_dm_params["lookback_period"])
        features.append(minus_dm[-1] / 100 if not np.isnan(minus_dm[-1]) else 0.5)

        plus_dm_params = self.technical_indicators["PLUS_DM"]
        plus_dm = talib.PLUS_DM(high_prices, low_prices, timeperiod=plus_dm_params["lookback_period"])
        features.append(plus_dm[-1] / 100 if not np.isnan(plus_dm[-1]) else 0.5)

        midpoint_params = self.technical_indicators["MIDPOINT"]
        midpoint = talib.MIDPOINT(close_prices, timeperiod=midpoint_params["lookback_period"])
        midpoint_ratio = current_price / midpoint[-1] if midpoint[-1] > 0 else 1.0
        features.append(midpoint_ratio if not np.isnan(midpoint_ratio) else 1.0)

        midprice_params = self.technical_indicators["MIDPRICE"]
        midprice = talib.MIDPRICE(high_prices, low_prices, timeperiod=midprice_params["lookback_period"])
        midprice_ratio = current_price / midprice[-1] if midprice[-1] > 0 else 1.0
        features.append(midprice_ratio if not np.isnan(midprice_ratio) else 1.0)

        t3_params = self.technical_indicators["T3"]
        t3 = talib.T3(close_prices, timeperiod=t3_params["lookback_period"], vfactor=t3_params["volume_factor"])
        t3_ratio = current_price / t3[-1] if t3[-1] > 0 else 1.0
        features.append(t3_ratio if not np.isnan(t3_ratio) else 1.0)

        # Volatility Indicators
        bb_params = self.technical_indicators["Bollinger_Bands"]
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_prices,
            timeperiod=bb_params["lookback_period"],
            nbdevup=bb_params["std_dev"],
            nbdevdn=bb_params["std_dev"]
        )
        bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
        bb_squeeze = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if bb_middle[-1] > 0 else 0.0
        features.extend([
            bb_position if not np.isnan(bb_position) else 0.5,
            bb_squeeze if not np.isnan(bb_squeeze) else 0.0
        ])

        atr_params = self.technical_indicators["ATR"]
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_params["lookback_period"])
        atr_normalized = atr[-1] / current_price if current_price > 0 else 0.0
        features.append(atr_normalized if not np.isnan(atr_normalized) else 0.0)

        if self.technical_indicators["TRANGE"]["enabled"]:
    passtrange = talib.TRANGE(high_prices, low_prices, close_prices)
            trange_normalized = trange[-1] / current_price if current_price > 0 else 0.0
            features.append(trange_normalized if not np.isnan(trange_normalized) else 0.0)
        else:
    passpassfeatures.append(0.0)

        var_params = self.technical_indicators["VAR"]
        var = talib.VAR(close_prices, timeperiod=var_params["lookback_period"])
        var_normalized = var[-1] / (current_price ** 2) if current_price > 0 else 0.0
        features.append(var_normalized if not np.isnan(var_normalized) else 0.0)

        stddev_params = self.technical_indicators["STDDEV"]
        stddev = talib.STDDEV(close_prices, timeperiod=stddev_params["lookback_period"])
        stddev_normalized = stddev[-1] / current_price if current_price > 0 else 0.0
        features.append(stddev_normalized if not np.isnan(stddev_normalized) else 0.0)

        # Volume Indicators
        if self.technical_indicators["OBV"]["enabled"]:
    passobv = talib.OBV(close_prices, volumes)
            obv_normalized = (obv[-1] - obv[-self.technical_indicators["OBV"]["smoothing_period"]]) / obv[-self.technical_indicators["OBV"]["smoothing_period"]] if obv[-self.technical_indicators["OBV"]["smoothing_period"]] > 0 else 0.0
            features.append(obv_normalized if not np.isnan(obv_normalized) else 0.0)
        else:
    passpassfeatures.append(0.0)

        if self.technical_indicators["AD"]["enabled"]:
    passad = talib.AD(high_prices, low_prices, close_prices, volumes)
            ad_normalized = (ad[-1] - ad[-self.technical_indicators["AD"]["smoothing_period"]]) / ad[-self.technical_indicators["AD"]["smoothing_period"]] if ad[-self.technical_indicators["AD"]["smoothing_period"]] > 0 else 0.0
            features.append(ad_normalized if not np.isnan(ad_normalized) else 0.0)
        else:
    passpassfeatures.append(0.0)

        adosc_params = self.technical_indicators["ADOSC"]
        adosc = talib.ADOSC(high_prices, low_prices, close_prices, volumes,
            fastperiod=adosc_params["fast_period"],
            slowperiod=adosc_params["slow_period"]
        )
        adosc_normalized = adosc[-1] / 1000 if not np.isnan(adosc[-1]) else 0.0
        features.append(adosc_normalized)

        mfi_params = self.technical_indicators["MFI"]
        mfi = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=mfi_params["lookback_period"])
        features.append(mfi[-1] / 100 if not np.isnan(mfi[-1]) else 0.5)

        # Cycle Indicators
        if self.technical_indicators["HT_DCPERIOD"]["enabled"]:
    passht_dcperiod = talib.HT_DCPERIOD(close_prices)
            features.append(ht_dcperiod[-1] / 100 if not np.isnan(ht_dcperiod[-1]) else 0.5)
        else:
    passpassfeatures.append(0.5)

        if self.technical_indicators["HT_DCPHASE"]["enabled"]:
    passht_dcphase = talib.HT_DCPHASE(close_prices)
            features.append(ht_dcphase[-1] / 100 if not np.isnan(ht_dcphase[-1]) else 0.5)
        else:
    passpassfeatures.append(0.5)

        if self.technical_indicators["HT_PHASOR"]["enabled"]:
    passht_phasor_inphase, ht_phasor_quadrature = talib.HT_PHASOR(close_prices)
            features.extend([
                ht_phasor_inphase[-1] / 100 if not np.isnan(ht_phasor_inphase[-1]) else 0.5,
                ht_phasor_quadrature[-1] / 100 if not np.isnan(ht_phasor_quadrature[-1]) else 0.5
            ])
        else:
    passpassfeatures.extend([0.5, 0.5])

        if self.technical_indicators["HT_SINE"]["enabled"]:
    passht_sine, ht_leadsine = talib.HT_SINE(close_prices)
            features.extend([
                ht_sine[-1] if not np.isnan(ht_sine[-1]) else 0.0,
                ht_leadsine[-1] if not np.isnan(ht_leadsine[-1]) else 0.0
            ])
        else:
    passpassfeatures.extend([0.0, 0.0])

        if self.technical_indicators["HT_TRENDMODE"]["enabled"]:
    passht_trendmode = talib.HT_TRENDMODE(close_prices)
            features.append(ht_trendmode[-1] if not np.isnan(ht_trendmode[-1]) else 0.0)
        else:
    passpassfeatures.append(0.0)

        # Math Transform
        linearreg_params = self.technical_indicators["LINEARREG"]
        linearreg = talib.LINEARREG(close_prices, timeperiod=linearreg_params["lookback_period"])
        linearreg_ratio = current_price / linearreg[-1] if linearreg[-1] > 0 else 1.0
        features.append(linearreg_ratio if not np.isnan(linearreg_ratio) else 1.0)

        tsf_params = self.technical_indicators["TSF"]
        tsf = talib.TSF(close_prices, timeperiod=tsf_params["lookback_period"])
        tsf_ratio = current_price / tsf[-1] if tsf[-1] > 0 else 1.0
        features.append(tsf_ratio if not np.isnan(tsf_ratio) else 1.0)

        stochrsi_params = self.technical_indicators["STOCHRSI"]
        stochrsi_fastk, stochrsi_fastd = talib.STOCHRSI(close_prices,
            fastk_period=stochrsi_params["fastk_period"],
            fastd_period=stochrsi_params["fastd_period"]
        )
        features.extend([
            stochrsi_fastk[-1] / 100 if not np.isnan(stochrsi_fastk[-1]) else 0.5,
            stochrsi_fastd[-1] / 100 if not np.isnan(stochrsi_fastd[-1]) else 0.5
        ])

        # Additional price-based features
        price_range = (high_prices[-1] - low_prices[-1]) / current_price
        upper_shadow = (high_prices[-1] - current_price) / current_price
        lower_shadow = (current_price - low_prices[-1]) / current_price
        body_size = abs(close_prices[-1] - open_prices[-1]) / current_price

        features.extend([price_range, upper_shadow, lower_shadow, body_size])

        # Latest return
        latest_return = (current_price - close_prices[-2]) / close_prices[-2] if len(close_prices) > 1 else 0.0
        features.append(latest_return)

        # Price acceleration (second derivative)
        if len(close_prices) >= 3:
    passreturn_1 = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
            return_2 = (close_prices[-2] - close_prices[-3]) / close_prices[-3]
            acceleration = return_1 - return_2
            features.append(acceleration)
        else:
    passfeatures.append(0.0)

        return np.array(features)

    def get_comprehensive_configuration_summary(...) -> ...:
    """..."""
    passreturn {
            "scenarios": self.scenarios,
            "time_limit_minutes": self.time_limit_minutes,
            "model_config": self.model_config,
            "technical_indicators": self.technical_indicators,
            "feature_config": self.feature_config,
            "is_trained": self.is_trained,
            "model_performance": self.model_performance,
            "feature_importance": self.feature_importance,
            "n_scenarios": len(self.scenarios),
            "n_features": 350  # Comprehensive feature count
        }