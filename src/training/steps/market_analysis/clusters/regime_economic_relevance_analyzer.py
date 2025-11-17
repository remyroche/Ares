"""
Analyseur de Pertinence Économique des Régimes

Ce module fournit une analyse complète de la pertinence économique des régimes de marché.
Il évalue si la classification correcte des régimes se traduit par de meilleures
performances de trading de manière stable et actionnable.

L'analyse répond à la question clé : "Does being right about regimes translate 
into better P&L in a stable, actionable way?"
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend non-interactif pour éviter les problèmes sur macOS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
from scipy import stats
from sklearn.utils import resample

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success,
        tprint_debug,
        tprint_timer,
        tprint_logged
    )
except ImportError:
    # Fallback basic logging if tprint is not available
    print("Warning: 'tprint' utilities not found. Using standard logging.")
    logging.basicConfig(level=logging.INFO)
    tprint_info = logging.info
    tprint_warning = logging.warning
    tprint_error = logging.error
    tprint_success = logging.info
    tprint_debug = logging.debug
    tprint_timer = lambda x: (lambda y: (lambda: y))(None)  # No-op timer
    tprint_logged = lambda **kwargs: lambda f: f  # No-op decorator

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Métriques de performance pour une stratégie de trading.
    
    Attributes:
        cagr: Compound Annual Growth Rate
        sharpe_ratio: Ratio de Sharpe annualisé
        nw_sharpe_ratio: Ratio de Sharpe ajusté Newey-West (robuste à l'autocorrélation)
        max_drawdown: Drawdown maximum
        max_drawdown_duration: Durée maximale de drawdown (périodes)
        avg_drawdown_duration: Durée moyenne des drawdowns (périodes)
        volatility: Volatilité annualisée
        turnover: Taux de rotation du portefeuille
        hit_rate: Taux de réussite (proportion de trades positifs)
        total_return: Rendement total sur la période
        calmar_ratio: Ratio Calmar (rendement / drawdown max)
        sortino_ratio: Ratio de Sortino (rendement / volatilité négative)
        var_95: Value-at-Risk à 95%
        cvar_95: Conditional VaR (Expected Shortfall) à 95%
    """
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    nw_sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: float = 0.0
    avg_drawdown_duration: float = 0.0
    volatility: float = 0.0
    turnover: float = 0.0
    hit_rate: float = 0.0
    total_return: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit les métriques en dictionnaire."""
        return {
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'nw_sharpe_ratio': self.nw_sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_drawdown_duration': self.avg_drawdown_duration,
            'volatility': self.volatility,
            'turnover': self.turnover,
            'hit_rate': self.hit_rate,
            'total_return': self.total_return,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95
        }


@dataclass
class StrategyResults:
    """
    Résultats d'évaluation pour une stratégie de trading.
    
    Attributes:
        name: Nom de la stratégie
        positions: Série temporelle des positions
        returns: Série temporelle des rendements
        metrics: Métriques de performance calculées
        benchmark_returns: Rendements du benchmark (buy & hold)
    """
    name: str
    positions: pd.Series
    returns: pd.Series
    metrics: PerformanceMetrics
    benchmark_returns: Optional[pd.Series] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les résultats en dictionnaire."""
        result = {
            'name': self.name,
            'metrics': self.metrics.to_dict(),
            'positions_summary': {
                'mean': float(self.positions.mean()),
                'std': float(self.positions.std()),
                'min': float(self.positions.min()),
                'max': float(self.positions.max()),
                'unique_values': len(self.positions.unique())
            },
            'returns_summary': {
                'mean': float(self.returns.mean()),
                'std': float(self.returns.std()),
                'skew': self._safe_float(self.returns.skew()),
                'kurt': self._safe_float(self.returns.kurtosis()),
                'min': float(self.returns.min()),
                'max': float(self.returns.max())
            }
        }
        
        if self.benchmark_returns is not None:
            result['benchmark_summary'] = {
                'mean': float(self.benchmark_returns.mean()),
                'std': float(self.benchmark_returns.std()),
                'total_return': self._safe_return_calc(self.benchmark_returns)
            }
        
        return result
    
    def _safe_float(self, value) -> float:
        """Conversion sécurisée en float."""
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_return_calc(self, returns: pd.Series) -> float:
        """Calcul sécurisé du rendement total."""
        try:
            if len(returns) == 0:
                return 0.0
            return float((1 + returns).prod() - 1)
        except (ValueError, TypeError):
            return 0.0


class RegimeEconomicRelevanceAnalyzer:
    """
    Analyseur de pertinence économique des régimes de marché.
    
    Cette classe évalue si la classification correcte des régimes se traduit
    par de meilleures performances de trading de manière stable et actionnable.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 trading_days_per_year: int = 252,
                 transaction_cost: float = 0.001,
                 significance_tests: bool = True,
                 n_permutations: int = 1000,
                 block_size: Optional[int] = None,
                 random_state: Optional[int] = None):
        """
        Initialise l'analyseur de pertinence économique.
        
        Args:
            risk_free_rate: Taux sans risque annualisé (défaut: 2%)
            trading_days_per_year: Nombre de jours de trading par an (défaut: 252)
            transaction_cost: Coût de transaction par trade (défaut: 0.1%)
            significance_tests: Activer les tests de signification (défaut: True)
            n_permutations: Nombre de permutations pour le test (défaut: 1000)
            block_size: Taille des blocs pour le test de permutation (auto si None)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.transaction_cost = transaction_cost
        self.significance_tests = significance_tests
        self.n_permutations = n_permutations
        self.block_size = block_size or 10  # Valeur par défaut si None
        self.rng = np.random.default_rng(random_state)
        
        tprint_info("🔧 Initialisation de RegimeEconomicRelevanceAnalyzer")
        tprint_info(f"   • Taux sans risque: {risk_free_rate:.1%}")
        tprint_info(f"   • Jours de trading/an: {trading_days_per_year}")
        tprint_info(f"   • Coût de transaction: {transaction_cost:.2%}")
        tprint_info(f"   • Tests de signification: {significance_tests}")
        
        # Mapping des régimes vers positions (basé sur caractéristiques économiques)
        self.regime_position_mapping = {
            # Bull/Expansion → Long (+1.0)
            'bull': 1.0,
            'expansion': 1.0,
            'trending': 0.8,
            'uptrend': 0.9,
            
            # Neutral/Range → Small long/flat (+0.3 à 0.5)
            'neutral': 0.4,
            'range': 0.3,
            'sideways': 0.3,
            'stable': 0.5,
            
            # High Vol/Crisis → Short or flat (-0.5 à 0)
            'bear': -0.3,
            'crisis': -0.5,
            'volatile': 0.0,
            'downtrend': -0.4,
            'correction': -0.2
        }
        
        # Mapping par ID de régime numérique (si les régimes sont numérotés)
        self.numeric_regime_mapping = {}
        
    def _build_dynamic_numeric_mapping(
        self,
        regime_labels: pd.Series,
        returns: pd.Series,
    ) -> Dict[int, float]:
        df = pd.DataFrame({"regime": regime_labels, "ret": returns}).dropna()
        if df.empty:
            return {}

        # Use 1-step forward returns as proxy for short-horizon regime alpha
        df["ret_fwd"] = df["ret"].shift(-1)
        df = df.dropna()
        if df.empty:
            return {}

        grouped = df.groupby("regime")["ret_fwd"]
        regime_mean = grouped.mean()
        if regime_mean.empty:
            return {}

        regime_std = grouped.std().replace(0.0, np.nan)
        market_mean = float(df["ret_fwd"].mean())

        # Alpha vs market and per-regime Sharpe (un-annualized, short-horizon)
        alphas = (regime_mean - market_mean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sharpe_raw = (regime_mean / (regime_std + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if alphas.empty:
            return {}

        max_abs_alpha = float(np.abs(alphas).max())
        max_abs_sharpe = float(np.abs(sharpe_raw).max())

        if max_abs_alpha == 0.0 and max_abs_sharpe == 0.0:
            # No discernible signal -> flat everywhere
            return {int(rid): 0.0 for rid in alphas.index}

        mapping: Dict[int, float] = {}
        deadzone = 0.15

        for rid in alphas.index:
            rid_int = int(rid)

            # Noise / unlabeled regime stays flat
            if rid_int == -1:
                mapping[rid_int] = 0.0
                continue

            alpha_val = float(alphas.loc[rid])
            sharpe_val = float(sharpe_raw.loc[rid])

            score_alpha = alpha_val / max_abs_alpha if max_abs_alpha > 0.0 else 0.0
            score_sharpe = sharpe_val / max_abs_sharpe if max_abs_sharpe > 0.0 else 0.0
            raw_score = 0.7 * score_alpha + 0.3 * score_sharpe

            # Continuous, sign-aware weighting with a small deadzone
            if abs(raw_score) < deadzone:
                pos = 0.0
            else:
                pos = float(np.clip(raw_score, -1.0, 1.0))

            mapping[rid_int] = pos

        tprint_info(
            f"   • Dynamic regime mapping (sign-aware, weighted) built for {len(mapping)} regimes "
            f"(deadzone={deadzone:.2f})"
        )
        return mapping
        
    def convert_regimes_to_positions(self, 
                                   regime_labels: Union[pd.Series, np.ndarray],
                                   regime_types: Optional[Dict[int, str]] = None,
                                   custom_mapping: Optional[Dict[Union[int, str], float]] = None) -> pd.Series:
        """
        Convertit les étiquettes de régimes en positions de trading.
        
        Args:
            regime_labels: Étiquettes des régimes (numériques ou textuelles)
            regime_types: Dictionnaire mapping ID de régime → type de régime
            custom_mapping: Mapping personnalisé régime → position
            
        Returns:
            Série temporelle des positions de trading
        """
        tprint_info("🔄 Conversion des régimes en positions de trading")
        
        # Conversion en Series si nécessaire
        if isinstance(regime_labels, np.ndarray):
            regime_labels = pd.Series(regime_labels)
        
        # Utiliser le mapping personnalisé si fourni
        if custom_mapping:
            mapping = custom_mapping
            tprint_info("   • Utilisation du mapping personnalisé")
        else:
            # Déterminer le mapping approprié
            if regime_types:
                # Mapping par type de régime
                mapping = {}
                for regime_id, regime_type in regime_types.items():
                    position = self.regime_position_mapping.get(
                        regime_type.lower(), 0.0
                    )
                    mapping[regime_id] = position
                tprint_info("   • Mapping par type de régime")
            else:
                # Mapping par ID numérique (si disponible)
                if self.numeric_regime_mapping:
                    mapping = self.numeric_regime_mapping
                    tprint_info("   • Mapping numérique pré-configuré")
                else:
                    # Mapping par défaut basé sur l'ordre des régimes
                    unique_regimes = sorted(regime_labels.unique())
                    n_regimes = len(unique_regimes)
                    
                    mapping = {}
                    for i, regime_id in enumerate(unique_regimes):
                        # Distribution symétrique autour de 0
                        if regime_id == -1:  # Noise
                            mapping[regime_id] = 0.0
                        else:
                            # Mapping linéaire: -1 → -0.5, 0 → 0.0, 1 → 0.5, etc.
                            normalized_pos = (i - n_regimes/2) / (n_regimes/2)
                            mapping[regime_id] = np.clip(normalized_pos, -1.0, 1.0)
                    
                    tprint_info(f"   • Mapping par défaut linéaire pour {n_regimes} régimes")
        
        # Appliquer le mapping
        positions = regime_labels.map(mapping).fillna(0.0)
        
        # Statistiques du mapping
        tprint_info(f"   • Positions: min={positions.min():.2f}, max={positions.max():.2f}, mean={positions.mean():.2f}")
        longs_forts = len(positions[positions > 0.5])
        longs_faibles = len(positions[(positions > 0) & (positions <= 0.5)])
        neutres = len(positions[positions == 0])
        courts = len(positions[positions < 0])
        tprint_info(f"   • Distribution: {longs_forts} longs forts, "
                   f"{longs_faibles} longs faibles, "
                   f"{neutres} neutres, "
                   f"{courts} courts")
        
        return positions
    
    def evaluate_strategies(self,
                          prices: pd.Series,
                          regime_labels: Union[pd.Series, np.ndarray],
                          regime_types: Optional[Dict[int, str]] = None,
                          predicted_regimes: Optional[Union[pd.Series, np.ndarray]] = None,
                          custom_mapping: Optional[Dict[Union[int, str], float]] = None,
                          returns_input: bool = False,
                          use_dynamic_mapping: bool = False) -> Dict[str, StrategyResults]:
        """
        Évalue les trois stratégies : prédite, réelle, et buy & hold.
        
        Args:
            prices: Série des prix (close)
            regime_labels: Étiquettes réelles des régimes
            regime_types: Dictionnaire mapping ID de régime → type de régime
            predicted_regimes: Étiquettes prédites des régimes (optionnel)
            custom_mapping: Mapping personnalisé régime → position
            
        Returns:
            Dictionnaire des résultats par stratégie
        """
        tprint_info("📊 Évaluation des stratégies de trading")
        
        # Calculer les rendements (ou utiliser directement si déjà fournis)
        if returns_input:
            returns = prices.fillna(0)
        else:
            returns = prices.pct_change().fillna(0)
        
        # Aligner les étiquettes de régime avec l'index des prix
        if isinstance(regime_labels, np.ndarray):
            regime_labels_series = pd.Series(regime_labels, index=prices.index)
        elif isinstance(regime_labels, pd.Series):
            regime_labels_series = regime_labels
        else:
            regime_labels_series = pd.Series(regime_labels, index=prices.index)
        
        if not regime_labels_series.index.equals(prices.index):
            regime_labels_series = pd.Series(regime_labels_series.values, index=prices.index)
        
        dynamic_regime_types = regime_types
        dynamic_custom_mapping = custom_mapping
        
        if use_dynamic_mapping and custom_mapping is None:
            try:
                dynamic_mapping = self._build_dynamic_numeric_mapping(regime_labels_series, returns)
                if dynamic_mapping:
                    self.numeric_regime_mapping = dynamic_mapping
                    dynamic_regime_types = None
                    dynamic_custom_mapping = None
            except Exception as e:
                tprint_warning(f"   • Dynamic regime mapping failed, falling back to default mapping: {e}")
        
        # Initialiser les résultats
        strategies = {}
        
        # 1. Stratégie Buy & Hold (benchmark)
        tprint_info("   • Évaluation de la stratégie Buy & Hold")
        buy_hold_positions = pd.Series(1.0, index=prices.index)
        buy_hold_returns = returns.copy()
        buy_hold_metrics = self.calculate_performance_metrics(buy_hold_returns, buy_hold_positions)
        
        strategies['buy_hold'] = StrategyResults(
            name='Buy & Hold',
            positions=buy_hold_positions,
            returns=buy_hold_returns,
            metrics=buy_hold_metrics
        )
        
        # 2. Stratégie basée sur les régimes réels
        tprint_info("   • Évaluation de la stratégie basée sur les régimes réels")
        real_positions = self.convert_regimes_to_positions(
            regime_labels_series, dynamic_regime_types, dynamic_custom_mapping
        )
        real_returns = self._calculate_strategy_returns(returns, real_positions)
        real_metrics = self.calculate_performance_metrics(real_returns, real_positions)
        
        strategies['real_regime'] = StrategyResults(
            name='Régimes Réels',
            positions=real_positions,
            returns=real_returns,
            metrics=real_metrics,
            benchmark_returns=buy_hold_returns
        )
        
        # 3. Stratégie basée sur les régimes prédits (si disponible)
        if predicted_regimes is not None:
            tprint_info("   • Évaluation de la stratégie basée sur les régimes prédits")
            pred_positions = self.convert_regimes_to_positions(
                predicted_regimes, dynamic_regime_types, dynamic_custom_mapping
            )
            pred_returns = self._calculate_strategy_returns(returns, pred_positions)
            pred_metrics = self.calculate_performance_metrics(pred_returns, pred_positions)
            
            strategies['predicted_regime'] = StrategyResults(
                name='Régimes Prédits',
                positions=pred_positions,
                returns=pred_returns,
                metrics=pred_metrics,
                benchmark_returns=buy_hold_returns
            )
        else:
            tprint_warning("   • Régimes prédits non fournis - stratégie omise")
        
        # Afficher le résumé des performances
        self._print_strategy_summary(strategies)
        
        return strategies
    
    def _calculate_strategy_returns(self, 
                                 market_returns: pd.Series, 
                                 positions: pd.Series) -> pd.Series:
        """
        Calcule les rendements d'une stratégie donnée.
        
        Args:
            market_returns: Rendements du marché
            positions: Positions de la stratégie
            
        Returns:
            Rendements de la stratégie (avec coûts de transaction)
        """
        # Aligner les séries
        if len(positions) != len(market_returns):
            positions = positions.reindex(market_returns.index, method='ffill').fillna(0.0)
        
        # Calculer les changements de position (turnover)
        position_changes = positions.diff().abs()
        
        # Appliquer les coûts de transaction
        transaction_costs = position_changes.fillna(0).astype(float) * self.transaction_cost
        
        # Calculer les rendements de la stratégie
        strategy_returns = (positions.shift(1) * market_returns) - transaction_costs
        strategy_returns = strategy_returns.fillna(0.0)
        
        return strategy_returns
    
    def calculate_performance_metrics(self, 
                                   returns: pd.Series, 
                                   positions: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calcule les métriques de performance d'une stratégie.
        
        Args:
            returns: Série des rendements de la stratégie
            positions: Série des positions (pour le turnover)
            
        Returns:
            Objet PerformanceMetrics avec toutes les métriques
        """
        # Rendement total
        if len(returns) == 0:
            return PerformanceMetrics()
        
        total_return = self._safe_return_calc(returns)
        
        # CAGR (Compound Annual Growth Rate)
        n_years = len(returns) / self.trading_days_per_year
        cagr = self._safe_pow_calc(1 + total_return, 1 / n_years) - 1 if n_years > 0 else 0.0
        
        # Volatilité annualisée
        volatility = float(returns.std() * np.sqrt(self.trading_days_per_year)) if len(returns) > 0 else 0.0
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        sharpe_ratio = float(excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year)) if len(returns) > 0 and returns.std() > 0 else 0.0
        
        # Maximum Drawdown and durations
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
        # Drawdown durations
        dd = drawdown.values
        durations = []
        cur = 0
        for v in dd:
            if v < 0:
                cur += 1
            elif cur > 0:
                durations.append(cur)
                cur = 0
        if cur > 0:
            durations.append(cur)
        max_dd_duration = float(max(durations)) if durations else 0.0
        avg_dd_duration = float(np.mean(durations)) if durations else 0.0
        
        # Calmar Ratio
        calmar_ratio = float(cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_std = float(negative_returns.std() * np.sqrt(self.trading_days_per_year)) if len(negative_returns) > 0 else 0.0
        sortino_ratio = float((cagr - self.risk_free_rate) / downside_std) if downside_std > 0 else 0.0
        
        # Newey-West adjusted Sharpe
        nw_sharpe_ratio = self._newey_west_sharpe(returns)
        
        # Tail risk metrics
        try:
            var_95 = float(np.percentile(returns, 5))
            cvar_95 = float(returns[returns <= var_95].mean()) if np.sum(returns <= var_95) > 0 else var_95
        except Exception:
            var_95, cvar_95 = 0.0, 0.0
        
        # Hit Rate
        hit_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0
        
        # Turnover (si positions fournies)
        turnover = 0.0
        if positions is not None:
            position_changes = positions.diff().abs()
            turnover = float(position_changes.mean()) if len(position_changes) > 0 else 0.0
        
        return PerformanceMetrics(
            cagr=cagr,
            sharpe_ratio=sharpe_ratio,
            nw_sharpe_ratio=nw_sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown_duration=avg_dd_duration,
            volatility=volatility,
            turnover=turnover,
            hit_rate=hit_rate,
            total_return=total_return,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _safe_return_calc(self, returns: pd.Series) -> float:
        """Calcul sécurisé du rendement total."""
        try:
            if len(returns) == 0:
                return 0.0
            return float((1 + returns).prod() - 1)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_pow_calc(self, base: float, exponent: float) -> float:
        """Calcul sécurisé de puissance."""
        try:
            if base <= 0:
                return 0.0
            return float(base ** exponent)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
    
    def perform_significance_test(self, 
                               strategies: Dict[str, StrategyResults],
                               test_method: str = 'block_permutation',
                               market_returns: Optional[pd.Series] = None,
                               positions_by_strategy: Optional[Dict[str, pd.Series]] = None) -> Dict[str, Any]:
        """
        Implémente le test de permutation par blocs pour évaluer la signification.
        
        Args:
            strategies: Résultats des stratégies à tester
            test_method: Méthode de test ('block_permutation', 'bootstrap')
            
        Returns:
            Résultats des tests de signification
        """
        if not self.significance_tests:
            tprint_info("⚠️ Tests de signification désactivés")
            return {}
        
        tprint_info(f"🔬 Test de signification: {test_method}")
        
        results = {}
        
        # Extraire les rendements des stratégies
        strategy_returns = {}
        for name, strategy in strategies.items():
            if name != 'buy_hold':  # Exclure le benchmark
                strategy_returns[name] = strategy.returns.values
        
        if not strategy_returns:
            tprint_warning("   • Aucune stratégie à tester")
            return {}
        
        # S'assurer que block_size est un entier
        block_size = int(self.block_size) if self.block_size is not None else 10
        
        tprint_info(f"   • Taille des blocs: {block_size}")
        tprint_info(f"   • Nombre de permutations: {self.n_permutations}")
        
        # Effectuer les tests
        if test_method == 'block_permutation':
            if market_returns is not None and positions_by_strategy is not None:
                results = self._block_permutation_positions_test(market_returns, positions_by_strategy, block_size)
            else:
                results = self._block_permutation_test(strategy_returns, block_size)
        elif test_method == 'bootstrap':
            results = self._bootstrap_test(strategy_returns)
        else:
            tprint_error(f"   • Méthode de test inconnue: {test_method}")
            return {}
        
        # Afficher les résultats
        self._print_significance_results(results)
        
        return results
    
    def _block_permutation_test(self, strategy_returns: Dict[str, np.ndarray], block_size: int) -> Dict[str, Any]:
        """
        Test de permutation par blocs pour préserver la structure temporelle.
        """
        tprint_debug("   • Exécution du test de permutation par blocs")
        
        # Statistiques observées
        observed_stats = {}
        for name, returns in strategy_returns.items():
            observed_stats[name] = {
                'mean': float(np.mean(returns)),
                'sharpe': self._calculate_sharpe(returns),
                'total_return': float(np.sum(returns))
            }
        
        # Distribution sous l'hypothèse nulle par permutation
        null_distributions = {name: {metric: [] for metric in ['mean', 'sharpe', 'total_return']} 
                            for name in strategy_returns.keys()}
        
        # Générer les permutations
        for i in range(self.n_permutations):
            if i % 100 == 0 and i > 0:
                tprint_debug(f"     • Permutation {i}/{self.n_permutations}")
            
            # Permutation par blocs pour une stratégie de référence
            reference_name = list(strategy_returns.keys())[0]
            reference_returns = strategy_returns[reference_name]
            permuted_indices = self._generate_block_permutation(len(reference_returns), block_size)
            
            # Appliquer la même permutation à toutes les stratégies
            for name, returns in strategy_returns.items():
                permuted_returns = returns[permuted_indices]
                
                null_distributions[name]['mean'].append(float(np.mean(permuted_returns)))
                null_distributions[name]['sharpe'].append(float(self._calculate_sharpe(permuted_returns)))
                null_distributions[name]['total_return'].append(float(np.prod(1.0 + permuted_returns) - 1.0))
        
        # Calculer les p-values
        p_values = {}
        p_values_two_sided = {}
        for name in strategy_returns.keys():
            p_values[name] = {}
            p_values_two_sided[name] = {}
            for metric in ['mean', 'sharpe', 'total_return']:
                observed = observed_stats[name][metric]
                null_dist = np.array(null_distributions[name][metric])
                
                p_upper = (np.sum(null_dist >= observed) + 1) / (len(null_dist) + 1)
                p_lower = (np.sum(null_dist <= observed) + 1) / (len(null_dist) + 1)
                p_values[name][metric] = p_upper
                p_values_two_sided[name][metric] = min(1.0, 2.0 * min(p_upper, p_lower))
        
        # Assembler les résultats
        results = {
            'method': 'block_permutation',
            'block_size': block_size,
            'n_permutations': self.n_permutations,
            'observed_stats': observed_stats,
            'p_values': p_values,
            'p_values_two_sided': p_values_two_sided,
            'null_distributions': null_distributions
        }
        
        return results
    
    def _block_permutation_positions_test(self, market_returns: pd.Series, positions_by_strategy: Dict[str, pd.Series], block_size: int) -> Dict[str, Any]:
        """Block-permutation on positions, recomputing strategy returns vs market.
        This preserves market structure and tests regime timing alignment."""
        # Align inputs
        mr = market_returns.dropna()
        pos_aligned = {}
        for name, pos in positions_by_strategy.items():
            ps = pos.reindex(mr.index, method='ffill').fillna(0.0)
            pos_aligned[name] = ps
        
        # Observed stats
        observed_stats = {}
        for name, ps in pos_aligned.items():
            sr = self._calculate_strategy_returns(mr, ps)
            observed_stats[name] = {
                'mean': float(sr.mean()),
                'sharpe': self._calculate_sharpe(sr.values),
                'total_return': float((1.0 + sr).prod() - 1.0)
            }
        
        # Null distributions
        null_distributions = {name: {metric: [] for metric in ['mean','sharpe','total_return']} for name in pos_aligned.keys()}
        n = len(mr)
        for i in range(self.n_permutations):
            perm_idx = self._generate_block_permutation(n, block_size)
            for name, ps in pos_aligned.items():
                perm_ps = pd.Series(ps.values[perm_idx], index=mr.index)
                sr = self._calculate_strategy_returns(mr, perm_ps)
                null_distributions[name]['mean'].append(float(sr.mean()))
                null_distributions[name]['sharpe'].append(float(self._calculate_sharpe(sr.values)))
                null_distributions[name]['total_return'].append(float((1.0 + sr).prod() - 1.0))
        
        # P-values (one- and two-sided)
        p_values = {}
        p_values_two_sided = {}
        for name in pos_aligned.keys():
            p_values[name] = {}
            p_values_two_sided[name] = {}
            for metric in ['mean','sharpe','total_return']:
                obs = observed_stats[name][metric]
                dist = np.array(null_distributions[name][metric])
                p_up = (np.sum(dist >= obs) + 1) / (len(dist) + 1)
                p_lo = (np.sum(dist <= obs) + 1) / (len(dist) + 1)
                p_values[name][metric] = p_up
                p_values_two_sided[name][metric] = float(min(1.0, 2.0 * min(p_up, p_lo)))
        
        return {
            'method': 'block_permutation_positions',
            'block_size': block_size,
            'n_permutations': self.n_permutations,
            'observed_stats': observed_stats,
            'p_values': p_values,
            'p_values_two_sided': p_values_two_sided,
            'null_distributions': null_distributions
        }
    
    def _bootstrap_test(self, strategy_returns: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Test bootstrap pour évaluer la robustesse des performances.
        """
        tprint_debug("   • Exécution du test bootstrap")
        
        # Statistiques observées
        observed_stats = {}
        for name, returns in strategy_returns.items():
            observed_stats[name] = {
                'mean': float(np.mean(returns)),
                'sharpe': self._calculate_sharpe(returns),
                'total_return': float(np.sum(returns))
            }
        
        # Distribution bootstrap
        bootstrap_distributions = {name: {metric: [] for metric in ['mean', 'sharpe', 'total_return']} 
                               for name in strategy_returns.keys()}
        
        n = len(next(iter(strategy_returns.values())))
        bsize = int(self.block_size) if self.block_size is not None else 10
        
        # Générer les échantillons bootstrap (Moving Block Bootstrap)
        for i in range(self.n_permutations):
            if i % 100 == 0 and i > 0:
                tprint_debug(f"     • Bootstrap {i}/{self.n_permutations}")
            
            for name, returns in strategy_returns.items():
                indices = []
                while len(indices) < n:
                    start = int(self.rng.integers(0, max(1, n - bsize + 1)))
                    end = min(start + bsize, n)
                    indices.extend(range(start, end))
                indices = np.array(indices[:n])
                bootstrap_sample = returns[indices]
                
                bootstrap_distributions[name]['mean'].append(float(np.mean(bootstrap_sample)))
                bootstrap_distributions[name]['sharpe'].append(float(self._calculate_sharpe(bootstrap_sample)))
                bootstrap_distributions[name]['total_return'].append(float(np.prod(1.0 + bootstrap_sample) - 1.0))
        
        # Calculer les intervalles de confiance et p-values
        results = {
            'method': 'bootstrap',
            'n_permutations': self.n_permutations,
            'observed_stats': observed_stats,
            'confidence_intervals': {},
            'bootstrap_distributions': bootstrap_distributions
        }
        
        for name in strategy_returns.keys():
            results['confidence_intervals'][name] = {}
            for metric in ['mean', 'sharpe', 'total_return']:
                dist = bootstrap_distributions[name][metric]
                observed = observed_stats[name][metric]
                
                # Intervalles de confiance 95%
                ci_lower = np.percentile(dist, 2.5)
                ci_upper = np.percentile(dist, 97.5)
                
                p_upper = (np.sum(np.array(dist) >= observed) + 1) / (len(dist) + 1)
                p_lower = (np.sum(np.array(dist) <= observed) + 1) / (len(dist) + 1)
                p_value = p_upper
                p_value_two_sided = min(1.0, 2.0 * min(p_upper, p_lower))
                
                results['confidence_intervals'][name][metric] = {
                    'ci_95_lower': float(ci_lower),
                    'ci_95_upper': float(ci_upper),
                    'p_value': p_value,
                    'p_value_two_sided': p_value_two_sided,
                    'observed': float(observed)
                }
    def _generate_block_permutation(self, n: int, block_size: int) -> np.ndarray:
        """
        Génère une permutation par blocs pour préserver la structure temporelle.
        """
        n_blocks = n // block_size
        if n % block_size != 0:
            n_blocks += 1
        
        blocks = []
        for i in range(n_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, n)
            blocks.append(np.arange(start, end))
        
        order = self.rng.permutation(n_blocks)
        permuted_indices = np.concatenate([blocks[i] for i in order])
        
        if len(permuted_indices) > n:
            permuted_indices = permuted_indices[:n]
        
        return permuted_indices
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sharpe pour un tableau de rendements."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        return float(np.mean(excess_returns) / np.std(returns) * np.sqrt(self.trading_days_per_year))
    
    def _newey_west_sharpe(self, returns: pd.Series, max_lag: Optional[int] = None) -> float:
        """Sharpe ajusté Newey-West (HAC) pour returns journaliers."""
        try:
            r = returns.values if isinstance(returns, pd.Series) else np.asarray(returns)
            if r.size < 2:
                return 0.0
            ex = r - self.risk_free_rate / self.trading_days_per_year
            mu = np.mean(ex)
            n = len(ex)
            if max_lag is None:
                max_lag = int(1.5 * np.sqrt(n))
            gamma0 = np.var(ex, ddof=0)
            s = gamma0
            for lag in range(1, max_lag + 1):
                w = 1 - lag / (max_lag + 1)
                cov = np.cov(ex[:-lag], ex[lag:], ddof=0)[0, 1]
                s += 2 * w * cov
            se = np.sqrt(s / n)
            if se == 0:
                return 0.0
            sharpe_nw = (mu / se) * np.sqrt(self.trading_days_per_year)
            return float(sharpe_nw)
        except Exception:
            return 0.0
    
    def _transaction_cost_sensitivity(self, market_returns: pd.Series, positions_by_strategy: Dict[str, pd.Series], cost_grid: Optional[List[float]] = None) -> Dict[str, Any]:
        """Evaluate performance vs different proportional transaction costs."""
        if cost_grid is None:
            cost_grid = [0.0, 0.0005, 0.001, 0.002]
        results = {}
        base_cost = self.transaction_cost
        for c in cost_grid:
            self.transaction_cost = c
            perf = {}
            for name, ps in positions_by_strategy.items():
                sr = self._calculate_strategy_returns(market_returns, ps)
                metrics = self.calculate_performance_metrics(sr, ps)
                perf[name] = {
                    'total_return': metrics.total_return,
                    'sharpe': metrics.sharpe_ratio,
                    'nw_sharpe': metrics.nw_sharpe_ratio
                }
            results[c] = perf
        self.transaction_cost = base_cost
        return results
    
    def generate_economic_report(self,
                              strategies: Dict[str, StrategyResults],
                              significance_results: Optional[Dict[str, Any]] = None,
                              output_dir: str = "outcomes",
                              report_prefix: Optional[str] = None) -> str:
        """
        Crée un rapport détaillé sur la pertinence économique des régimes.

        Args:
            strategies: Résultats des stratégies évaluées
            significance_results: Résultats des tests de signification
            output_dir: Répertoire de sortie
            report_prefix: Optional prefix for the report filename (default: 'regime_economic_relevance_report')

        Returns:
            Chemin du rapport généré
        """
        tprint_info("📝 Génération du rapport économique")

        # Créer le répertoire de sortie
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Générer le nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = report_prefix if report_prefix else "regime_economic_relevance_report"
        report_filename = f"{prefix}_{timestamp}.md"
        report_path = output_path / report_filename
        
        # Construire le contenu du rapport
        md_content = self._build_economic_report_content(strategies, significance_results)
        
        # Écrire le fichier MD
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # Écrire le JSON avec le même timestamp
        json_filename = f"{prefix}_{timestamp}.json"
        json_path = output_path / json_filename
        save_data = {
            'metadata': {
                'timestamp': timestamp,
                'analysis_type': 'regime_economic_relevance',
                'configuration': {
                    'risk_free_rate': self.risk_free_rate,
                    'trading_days_per_year': self.trading_days_per_year,
                    'transaction_cost': self.transaction_cost,
                    'significance_tests': self.significance_tests,
                    'n_permutations': self.n_permutations,
                    'block_size': self.block_size
                }
            },
            'strategies': {name: strategy.to_dict() for name, strategy in strategies.items()},
            'significance_results': significance_results or {}
        }
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(save_data, jf, indent=2, ensure_ascii=False)
        
        # Générer les graphiques
        self._generate_performance_charts(strategies, output_path, timestamp)
        
        absolute_report_path = str(report_path.resolve())
        tprint_success(f"✅ Rapport généré: {absolute_report_path}")
        
        return absolute_report_path
    
    def _build_economic_report_content(self, 
                                    strategies: Dict[str, StrategyResults],
                                    significance_results: Optional[Dict[str, Any]]) -> str:
        """Construit le contenu du rapport markdown."""
        
        # En-tête du rapport
        md = """# Rapport de Pertinence Économique des Régimes

**Date:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
**Analyse:** Does being right about regimes translate into better P&L in a stable, actionable way?

---

## Résumé Exécutif

"""
        
        # Extraire les métriques clés
        if 'buy_hold' in strategies:
            bh_metrics = strategies['buy_hold'].metrics
            md += f"**Benchmark (Buy & Hold):**\n"
            md += f"- Rendement total: {bh_metrics.total_return:.2%}\n"
            md += f"- Sharpe Ratio: {bh_metrics.sharpe_ratio:.2f}\n"
            md += f"- Maximum Drawdown: {bh_metrics.max_drawdown:.2%}\n\n"
        
        if 'real_regime' in strategies:
            real_metrics = strategies['real_regime'].metrics
            bh_metrics = strategies['buy_hold'].metrics
            
            # Calculer l'outperformance
            excess_return = real_metrics.total_return - bh_metrics.total_return
            excess_sharpe = real_metrics.sharpe_ratio - bh_metrics.sharpe_ratio
            
            md += f"**Stratégie Basée sur les Régimes:**\n"
            md += f"- Rendement total: {real_metrics.total_return:.2%} ({excess_return:+.2%} vs benchmark)\n"
            md += f"- Sharpe Ratio: {real_metrics.sharpe_ratio:.2f} ({excess_sharpe:+.2f} vs benchmark)\n"
            md += f"- Maximum Drawdown: {real_metrics.max_drawdown:.2%}\n"
            md += f"- Turnover: {real_metrics.turnover:.4f}\n\n"
        
        if 'predicted_regime' in strategies:
            pred_metrics = strategies['predicted_regime'].metrics
            bh_metrics = strategies['buy_hold'].metrics
            
            # Calculer l'outperformance
            excess_return = pred_metrics.total_return - bh_metrics.total_return
            excess_sharpe = pred_metrics.sharpe_ratio - bh_metrics.sharpe_ratio
            
            md += f"**Stratégie Basée sur les Régimes Prédits:**\n"
            md += f"- Rendement total: {pred_metrics.total_return:.2%} ({excess_return:+.2%} vs benchmark)\n"
            md += f"- Sharpe Ratio: {pred_metrics.sharpe_ratio:.2f} ({excess_sharpe:+.2f} vs benchmark)\n"
            md += f"- Maximum Drawdown: {pred_metrics.max_drawdown:.2%}\n"
            md += f"- Turnover: {pred_metrics.turnover:.4f}\n\n"
        
        # Conclusion sur la pertinence économique
        md += """## Conclusion sur la Pertinence Économique

"""
        
        if 'real_regime' in strategies:
            real_metrics = strategies['real_regime'].metrics
            bh_metrics = strategies['buy_hold'].metrics
            
            if real_metrics.sharpe_ratio > bh_metrics.sharpe_ratio * 1.1:  # 10% better
                md += "✅ **OUI** - La connaissance correcte des régimes se traduit par une meilleure performance économique.\n\n"
                md += "Les régimes identifiés ont une valeur économique actionnable et peuvent être utilisés "
                md += "pour améliorer les décisions de trading.\n\n"
            elif real_metrics.sharpe_ratio > bh_metrics.sharpe_ratio:
                md += "⚠️ **PARTIELLEMENT** - La connaissance des régimes apporte un bénéfice modeste.\n\n"
                md += "L'amélioration de performance est limitée et pourrait ne pas justifier "
                md += "la complexité ajoutée au système de trading.\n\n"
            else:
                md += "❌ **NON** - La connaissance des régimes n'améliore pas la performance.\n\n"
                md += "Les régimes identifiés n'ont pas de valeur économique actionnable dans leur "
                md += "forme actuelle.\n\n"
        
        # Section de performance détaillée
        md += """---

## Analyse Détaillée des Performances

### Tableau Comparatif

| Stratégie | Rendement Total | CAGR | Sharpe | Volatilité | Max DD | Calmar | Turnover |
|------------|-----------------|-------|---------|-------------|---------|---------|----------|
"""
        
        for name, strategy in strategies.items():
            metrics = strategy.metrics
            md += f"| {strategy.name} | {metrics.total_return:.2%} | {metrics.cagr:.2%} | {metrics.sharpe_ratio:.2f} | "
            md += f"{metrics.volatility:.2%} | {metrics.max_drawdown:.2%} | {metrics.calmar_ratio:.2f} | {metrics.turnover:.4f} |\n"
        
        # Résultats des tests de signification
        if significance_results:
            md += """---

## Tests de Signification

"""
            
            if 'p_values' in significance_results:
                md += "### Test de Permutation par Blocs\n\n"
                md += "**P-values (unilatérales):**\n\n"
                
                p_values = significance_results['p_values']
                for strategy_name, metrics in p_values.items():
                    md += f"**{strategy_name}:**\n"
                    for metric, p_value in metrics.items():
                        significance = "significatif" if p_value < 0.05 else "non significatif"
                        md += f"- {metric}: {p_value:.3f} ({significance})\n"
                    md += "\n"
            
            elif 'confidence_intervals' in significance_results:
                md += "### Test Bootstrap\n\n"
                md += "**Intervalles de confiance 95%:**\n\n"
                
                ci = significance_results['confidence_intervals']
                for strategy_name, metrics in ci.items():
                    md += f"**{strategy_name}:**\n"
                    for metric, values in metrics.items():
                        observed = values.get('observed', 0)
                        lower = values['ci_95_lower']
                        upper = values['ci_95_upper']
                        p_value = values['p_value']
                        significance = "significatif" if p_value < 0.05 else "non significatif"
                        
                        md += f"- {metric}: {observed:.4f} [{lower:.4f}, {upper:.4f}] (p={p_value:.3f}, {significance})\n"
                    md += "\n"
        
        # Recommandations
        md += """---

## Recommandations

"""
        
        if 'real_regime' in strategies and 'buy_hold' in strategies:
            real_metrics = strategies['real_regime'].metrics
            bh_metrics = strategies['buy_hold'].metrics
            
            if real_metrics.sharpe_ratio > bh_metrics.sharpe_ratio * 1.2:  # 20% better
                md += "### Forte Recommandation\n\n"
                md += "Les régimes identifiés ont une forte valeur économique. Recommandé:\n"
                md += "- Intégrer les régimes dans le système de trading live\n"
                md += "- Développer des stratégies spécifiques par régime\n"
                md += "- Monitorer la stabilité des performances sur le long terme\n\n"
            elif real_metrics.sharpe_ratio > bh_metrics.sharpe_ratio:
                md += "### Recommandation Modérée\n\n"
                md += "Les régimes ont une valeur économique limitée. Considérer:\n"
                md += "- Tests supplémentaires avant déploiement\n"
                md += "- Optimisation des mappings régime→position\n"
                md += "- Analyse des coûts de transaction\n\n"
            else:
                md += "### Pas de Recommandation\n\n"
                md += "Les régimes n'apportent pas de valeur économique. Suggéré:\n"
                md += "- Revoir la méthodologie de détection des régimes\n"
                md += "- Tester des approches alternatives\n"
                md += "- Se concentrer sur d'autres facteurs alpha\n\n"
        
        # Méta-informations
        md += """---

## Méta-informations

**Configuration de l'analyse:**
- Taux sans risque: """ + f"{self.risk_free_rate:.1%}" + """
- Jours de trading/an: """ + str(self.trading_days_per_year) + """
- Coût de transaction: """ + f"{self.transaction_cost:.2%}" + """
- Tests de signification: """ + ("Activés" if self.significance_tests else "Désactivés") + """

"""
        
        if significance_results:
            md += f"**Tests de signification:**\n"
            md += f"- Méthode: {significance_results.get('method', 'N/A')}\n"
            if 'n_permutations' in significance_results:
                md += f"- Nombre de permutations: {significance_results['n_permutations']}\n"
            if 'block_size' in significance_results:
                md += f"- Taille des blocs: {significance_results['block_size']}\n"
            md += "\n"
        
        md += """---

*Ce rapport a été généré automatiquement par RegimeEconomicRelevanceAnalyzer.*
"""
        
        return md
    
    def _generate_performance_charts(self,
                                 strategies: Dict[str, StrategyResults],
                                 output_path: Path,
                                 timestamp: str):
        """Génère les graphiques de performance."""
        
        tprint_info("   • Génération des graphiques de performance")
        
        try:
            # Configuration du style
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('default')
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Analyse de Performance des Stratégies', fontsize=16)
            
            # 1. Performance cumulée
            ax1 = axes[0, 0]
            for name, strategy in strategies.items():
                cumulative = (1 + strategy.returns).cumprod()
                ax1.plot(cumulative.index, cumulative, label=strategy.name, linewidth=2)
            
            ax1.set_title('Performance Cumulée')
            ax1.set_ylabel('Rendement Cumulé')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdown
            ax2 = axes[0, 1]
            for name, strategy in strategies.items():
                cumulative = (1 + strategy.returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, label=strategy.name)
            
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Comparaison des métriques
            ax3 = axes[1, 0]
            metrics_comparison = []
            strategy_names = []
            
            for name, strategy in strategies.items():
                metrics_comparison.append([
                    strategy.metrics.sharpe_ratio,
                    strategy.metrics.calmar_ratio,
                    strategy.metrics.sortino_ratio
                ])
                strategy_names.append(strategy.name)
            
            metrics_comparison = np.array(metrics_comparison)
            x = np.arange(len(strategy_names))
            width = 0.25
            
            ax3.bar(x - width, metrics_comparison[:, 0], width, label='Sharpe')
            ax3.bar(x, metrics_comparison[:, 1], width, label='Calmar')
            ax3.bar(x + width, metrics_comparison[:, 2], width, label='Sortino')
            
            ax3.set_title('Comparaison des Ratios de Performance')
            ax3.set_ylabel('Ratio')
            ax3.set_xticks(x)
            ax3.set_xticklabels(strategy_names)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Distribution des rendements
            ax4 = axes[1, 1]
            for name, strategy in strategies.items():
                if name == 'buy_hold':  # Focus sur les stratégies de régime
                    continue
                ax4.hist(strategy.returns, bins=50, alpha=0.6, label=strategy.name, density=True)
            
            ax4.set_title('Distribution des Rendements')
            ax4.set_xlabel('Rendement')
            ax4.set_ylabel('Densité')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Ajuster et sauvegarder
            plt.tight_layout()
            chart_filename = f"regime_performance_charts_{timestamp}.png"
            chart_path = output_path / chart_filename
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            tprint_success(f"   • Graphiques sauvegardés: {chart_path}")
            
        except Exception as e:
            tprint_warning(f"   • Impossible de générer les graphiques: {str(e)}")
            tprint_warning("   • Poursuite de l'analyse sans graphiques")
    
    def _print_strategy_summary(self, strategies: Dict[str, StrategyResults]):
        """Affiche un résumé des performances des stratégies."""
        
        tprint_info("\n📈 Résumé des Performances:")
        tprint_info("=" * 80)
        tprint_info(f"{'Stratégie':<20} {'Rendement':<12} {'Sharpe':<8} {'Max DD':<10} {'Turnover':<10}")
        tprint_info("=" * 80)
        
        for name, strategy in strategies.items():
            metrics = strategy.metrics
            tprint_info(f"{strategy.name:<20} {metrics.total_return:<12.2%} {metrics.sharpe_ratio:<8.2f} "
                       f"{metrics.max_drawdown:<10.2%} {metrics.turnover:<10.4f}")
        
        tprint_info("=" * 80)
    
    def _print_significance_results(self, results: Optional[Dict[str, Any]]):
        """Affiche les résultats des tests de signification."""
        
        if not results:
            tprint_info("\n🔬 Aucun résultat de test de signification disponible")
            return
        
        tprint_info("\n🔬 Résultats des Tests de Signification:")
        tprint_info("=" * 60)
        
        if 'p_values' in results:
            tprint_info("Test de Permutation par Blocs:")
            for strategy_name, metrics in results['p_values'].items():
                tprint_info(f"\n{strategy_name}:")
                for metric, p_value in metrics.items():
                    significance = "✅" if p_value < 0.05 else "❌"
                    tprint_info(f"  {metric}: {p_value:.4f} {significance}")
        
        elif 'confidence_intervals' in results:
            tprint_info("Test Bootstrap:")
            for strategy_name, metrics in results['confidence_intervals'].items():
                tprint_info(f"\n{strategy_name}:")
                for metric, values in metrics.items():
                    observed = values.get('observed', 0)
                    p_value = values['p_value']
                    significance = "✅" if p_value < 0.05 else "❌"
                    tprint_info(f"  {metric}: {observed:.4f} (p={p_value:.4f}) {significance}")
        
        tprint_info("=" * 60)
    
    def save_results(self,
                   strategies: Dict[str, StrategyResults],
                   significance_results: Optional[Dict[str, Any]] = None,
                   output_dir: str = "outcomes",
                   json_prefix: Optional[str] = None) -> str:
        """
        Sauvegarde les résultats complets dans outcomes/ avec timestamp.

        Args:
            strategies: Résultats des stratégies évaluées
            significance_results: Résultats des tests de signification
            output_dir: Répertoire de sortie
            json_prefix: Optional prefix for the JSON filename (default: 'regime_economic_analysis')

        Returns:
            Chemin du fichier JSON sauvegardé
        """
        tprint_info("💾 Sauvegarde des résultats complets")

        # Créer le répertoire de sortie
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Générer le nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = json_prefix if json_prefix else "regime_economic_analysis"
        json_filename = f"{prefix}_{timestamp}.json"
        json_path = output_path / json_filename
        
        # Préparer les données à sauvegarder
        save_data = {
            'metadata': {
                'timestamp': timestamp,
                'analysis_type': 'regime_economic_relevance',
                'configuration': {
                    'risk_free_rate': self.risk_free_rate,
                    'trading_days_per_year': self.trading_days_per_year,
                    'transaction_cost': self.transaction_cost,
                    'significance_tests': self.significance_tests,
                    'n_permutations': self.n_permutations,
                    'block_size': self.block_size
                }
            },
            'strategies': {name: strategy.to_dict() for name, strategy in strategies.items()},
            'significance_results': significance_results or {}
        }
        
        # Sauvegarder en JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        absolute_json_path = str(json_path.resolve())
        tprint_success(f"✅ Résultats sauvegardés: {absolute_json_path}")
        
        return absolute_json_path


def create_regime_economic_relevance_analyzer(
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
    transaction_cost: float = 0.001,
    significance_tests: bool = True,
    n_permutations: int = 1000,
    block_size: Optional[int] = None,
    random_state: Optional[int] = None
) -> RegimeEconomicRelevanceAnalyzer:
    """
    Fonction usine pour créer un analyseur de pertinence économique des régimes.
    
    Args:
        risk_free_rate: Taux sans risque annualisé
        trading_days_per_year: Nombre de jours de trading par an
        transaction_cost: Coût de transaction par trade
        significance_tests: Activer les tests de signification
        n_permutations: Nombre de permutations pour les tests
        block_size: Taille des blocs pour le test de permutation
        
    Returns:
        Instance de RegimeEconomicRelevanceAnalyzer configurée
    """
    return RegimeEconomicRelevanceAnalyzer(
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
        transaction_cost=transaction_cost,
        significance_tests=significance_tests,
        n_permutations=n_permutations,
        block_size=block_size,
        random_state=random_state
    )