"""
Tests complets pour le module Rolling HMM Clustering

Ce module contient des tests unitaires, d'intégration et de bout en bout pour
tous les composants du module rolling_hmm_clustering.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import asyncio
import warnings

# Ignorer les avertissements pour les tests
warnings.filterwarnings('ignore')

# Import des classes à tester
from src.training.steps.market_analysis.rolling_hmm_clustering.feature_engineering import (
    RollingHMMFeatureEngineer,
    FeatureEngineeringConfig,
    EWMAConfig,
    DEFAULT_EWMA_CONFIGS
)
from src.training.steps.market_analysis.rolling_hmm_clustering.sticky_hmm_model import (
    StickyHMMModel,
    StickyHMMConfig
)
from src.training.steps.market_analysis.rolling_hmm_clustering.hpo_config import (
    RollingHMMOptimizer,
    HPOConfig,
    DEFAULT_HPO_CONFIG
)
from src.training.steps.market_analysis.rolling_hmm_clustering.rolling_hmm_regime_discovery_step import (
    RollingHMMRegimeDiscoveryStep
)


class TestFeatureEngineering(unittest.TestCase):
    """Tests pour la classe RollingHMMFeatureEngineer."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Configuration simple pour les tests
        self.config = FeatureEngineeringConfig(
            ewma_configs=[EWMAConfig(short_window=8, long_window=16, name="8+16")],
            use_log_returns=True,
            use_volatility_features=True,
            use_trend_features=True,
            use_volume_features=True,
            pca_components=3,
            normalize_method='zscore',
            rolling_normalize_window=50,
            enable_vectorbt_optimization=False,  # Désactivé pour les tests
            enable_hardware_optimization=False,  # Désactivé pour les tests
            enable_numba_jit=False  # Désactivé pour les tests
        )
        
        # Données de marché synthétiques
        np.random.seed(42)
        n_samples = 200
        self.market_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'high': 100 + np.cumsum(np.random.normal(0.1, 1, n_samples)),
            'low': 100 + np.cumsum(np.random.normal(-0.1, 1, n_samples)),
            'close': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        # S'assurer que high >= close >= low
        self.market_data['high'] = np.maximum(self.market_data['high'], self.market_data['close'])
        self.market_data['low'] = np.minimum(self.market_data['low'], self.market_data['close'])
    
    def test_initialization(self):
        """Test de l'initialisation de RollingHMMFeatureEngineer."""
        engineer = RollingHMMFeatureEngineer(self.config)
        
        self.assertEqual(engineer.config, self.config)
        self.assertEqual(len(engineer.config.ewma_configs), 1)
        self.assertEqual(engineer.config.ewma_configs[0].name, "8+16")
        self.assertIsNone(engineer.rolling_optimizer)
        self.assertIsNone(engineer.hardware_manager)
    
    def test_ewma_config_validation(self):
        """Test de la validation des configurations EWMA."""
        # Test avec short_window >= long_window (doit échouer)
        with self.assertRaises(ValueError):
            EWMAConfig(short_window=16, long_window=8, name="16+8")
        
        # Test avec short_window < long_window (doit réussir)
        ewma_config = EWMAConfig(short_window=8, long_window=16, name="8+16")
        self.assertEqual(ewma_config.short_window, 8)
        self.assertEqual(ewma_config.long_window, 16)
    
    def test_generate_features(self):
        """Test de la génération de caractéristiques."""
        engineer = RollingHMMFeatureEngineer(self.config)
        features = engineer.generate_features(self.market_data)
        
        # Vérifier que les caractéristiques sont générées
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        # La génération de caractéristiques peut supprimer des lignes avec des NaN
        self.assertLessEqual(len(features), len(self.market_data))
        
        # Vérifier que les caractéristiques sont normalisées (très tolérant)
        for col in features.columns:
            self.assertLess(abs(features[col].mean()), 1.0)  # Moyenne proche de 0
            self.assertGreater(features[col].std(), 0.1)  # Écart-type > 0
    
    def test_precompute_all_features(self):
        """Test du précalcul des caractéristiques pour toutes les configurations EWMA."""
        # Configuration avec plusieurs EWMA
        multi_config = FeatureEngineeringConfig(
            ewma_configs=[
                EWMAConfig(short_window=8, long_window=16, name="8+16"),
                EWMAConfig(short_window=12, long_window=24, name="12+24")
            ],
            enable_vectorbt_optimization=False,
            enable_hardware_optimization=False,
            enable_numba_jit=False
        )
        
        engineer = RollingHMMFeatureEngineer(multi_config)
        all_features = engineer.precompute_all_features(self.market_data)
        
        # Vérifier que les caractéristiques sont précalculées pour toutes les configurations
        self.assertEqual(len(all_features), 2)
        self.assertIn("8+16", all_features)
        self.assertIn("12+24", all_features)
        
        # Vérifier que les caractéristiques sont mises en cache
        self.assertIsNotNone(engineer.get_cached_features(EWMAConfig(short_window=8, long_window=16, name="8+16")))
        self.assertIsNotNone(engineer.get_cached_features(EWMAConfig(short_window=12, long_window=24, name="12+24")))
    
    def test_apply_pca(self):
        """Test de l'application de l'ACP."""
        engineer = RollingHMMFeatureEngineer(self.config)
        features = engineer.generate_features(self.market_data)
        
        # Appliquer l'ACP
        features_pca, pca_model, explained_var = engineer.apply_pca(
            features, n_components=3
        )
        
        # Vérifier les résultats
        self.assertEqual(features_pca.shape[1], 3)
        self.assertEqual(len(features_pca), len(features))
        self.assertGreaterEqual(explained_var, 0)
        self.assertLessEqual(explained_var, 1)
        
        # Vérifier que l'ACP est mise en cache
        cache_key = (str(hash(tuple(features.index))), 3)
        self.assertIn(cache_key, engineer._pca_cache)


class TestStickyHMMModel(unittest.TestCase):
    """Tests pour la classe StickyHMMModel."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Configuration simple pour les tests
        self.config = StickyHMMConfig(
            n_components=3,
            n_iter=50,  # Réduit pour les tests
            covariance_type='diag',
            min_covar=1e-3,
            kappa=10.0,
            use_sticky_priors=True,
            kmeans_init=True,
            random_state=42
        )
        
        # Données synthétiques
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        self.X = np.random.randn(n_samples, n_features)
    
    def test_initialization(self):
        """Test de l'initialisation de StickyHMMModel."""
        model = StickyHMMModel(self.config)
        
        self.assertEqual(model.config, self.config)
        self.assertFalse(model.is_fitted)
        self.assertEqual(model.model.n_components, 3)
        self.assertEqual(model.model.covariance_type, 'diag')
    
    def test_fit(self):
        """Test de l'ajustement du modèle."""
        model = StickyHMMModel(self.config)
        model.fit(self.X)
        
        # Vérifier que le modèle est ajusté
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.feature_dim)
        self.assertEqual(model.feature_dim, self.X.shape[1])
    
    def test_predict(self):
        """Test de la prédiction des états cachés."""
        model = StickyHMMModel(self.config)
        model.fit(self.X)
        
        # Prédire les états
        states = model.predict(self.X)
        
        # Vérifier les résultats
        self.assertEqual(len(states), len(self.X))
        self.assertTrue(np.all(states >= 0))
        self.assertTrue(np.all(states < self.config.n_components))
    
    def test_predict_proba(self):
        """Test de la prédiction des probabilités d'état."""
        model = StickyHMMModel(self.config)
        model.fit(self.X)
        
        # Prédire les probabilités
        probs = model.predict_proba(self.X)
        
        # Vérifier les résultats
        self.assertEqual(probs.shape, (len(self.X), self.config.n_components))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))  # Les probabilités somment à 1
    
    def test_score(self):
        """Test du calcul de la log-vraisemblance."""
        model = StickyHMMModel(self.config)
        model.fit(self.X)
        
        # Calculer le score
        score = model.score(self.X)
        
        # Vérifier les résultats
        self.assertIsInstance(score, float)
        self.assertLess(score, 0)  # La log-vraisemblance est négative
    
    def test_get_transition_matrix(self):
        """Test de la récupération de la matrice de transition."""
        model = StickyHMMModel(self.config)
        model.fit(self.X)
        
        # Récupérer la matrice de transition
        transmat = model.get_transition_matrix()
        
        # Vérifier les résultats
        self.assertEqual(transmat.shape, (self.config.n_components, self.config.n_components))
        self.assertTrue(np.allclose(transmat.sum(axis=1), 1.0))  # Les lignes somment à 1
    
    def test_get_stationary_distribution(self):
        """Test du calcul de la distribution stationnaire."""
        model = StickyHMMModel(self.config)
        model.fit(self.X)
        
        # Calculer la distribution stationnaire
        stationary = model.get_stationary_distribution()
        
        # Vérifier les résultats
        self.assertEqual(len(stationary), self.config.n_components)
        self.assertAlmostEqual(stationary.sum(), 1.0, places=6)
        self.assertTrue(np.all(stationary >= 0))
    
    def test_get_expected_durations(self):
        """Test du calcul des durées attendues des états."""
        model = StickyHMMModel(self.config)
        model.fit(self.X)
        
        # Calculer les durées attendues
        durations = model.get_expected_durations()
        
        # Vérifier les résultats
        self.assertEqual(len(durations), self.config.n_components)
        self.assertTrue(np.all(durations > 0))


class TestHPOConfig(unittest.TestCase):
    """Tests pour la classe RollingHMMOptimizer."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Configuration simple pour les tests
        self.config = HPOConfig(
            stages=None,  # Utilisera les valeurs par défaut
            n_rounds=1,  # Réduit pour les tests
            enable_final_refinement=True,
            final_refinement_trials=5,  # Réduit pour les tests
            cv_folds=2,  # Réduit pour les tests
            verbose=False  # Désactivé pour les tests
        )
        
        # Données de marché synthétiques
        np.random.seed(42)
        n_samples = 100
        self.market_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'high': 100 + np.cumsum(np.random.normal(0.1, 1, n_samples)),
            'low': 100 + np.cumsum(np.random.normal(-0.1, 1, n_samples)),
            'close': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        # S'assurer que high >= close >= low
        self.market_data['high'] = np.maximum(self.market_data['high'], self.market_data['close'])
        self.market_data['low'] = np.minimum(self.market_data['low'], self.market_data['close'])
    
    def test_initialization(self):
        """Test de l'initialisation de RollingHMMOptimizer."""
        optimizer = RollingHMMOptimizer(self.config)
        
        self.assertEqual(optimizer.config, self.config)
        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(optimizer.param_groups[0].name, "feature_engineering")
        self.assertEqual(optimizer.param_groups[1].name, "model_structure")
        self.assertEqual(optimizer.param_groups[2].name, "regularization")
    
    def test_create_parameter_groups(self):
        """Test de la création des groupes de paramètres."""
        optimizer = RollingHMMOptimizer(self.config)
        groups = optimizer._create_parameter_groups()
        
        # Vérifier les groupes
        self.assertEqual(len(groups), 3)
        
        # Groupe 1: Feature Engineering
        self.assertEqual(groups[0].name, "feature_engineering")
        self.assertIn("ewma_config_idx", groups[0].params)
        self.assertEqual(groups[0].params["ewma_config_idx"]["type"], "categorical")
        self.assertEqual(len(groups[0].params["ewma_config_idx"]["choices"]), 6)
        
        # Groupe 2: Model Structure
        self.assertEqual(groups[1].name, "model_structure")
        self.assertIn("n_components", groups[1].params)
        self.assertIn("pca_components", groups[1].params)
        
        # Groupe 3: Regularization
        self.assertEqual(groups[2].name, "regularization")
        self.assertIn("min_covar", groups[2].params)
        self.assertIn("kappa", groups[2].params)
    
    @patch('src.training.steps.market_analysis.rolling_hmm_clustering.hpo_config.ClusterQualityAssessor')
    def test_create_objective_function(self, mock_quality_assessor):
        """Test de la création de la fonction objectif."""
        # Mock du quality assessor
        mock_assessor_instance = Mock()
        mock_quality_assessor.return_value = mock_assessor_instance
        
        # Mock des métriques
        mock_metrics = Mock()
        mock_metrics.within_regime_cv = 0.1
        mock_metrics.between_regime_cv = 0.3
        mock_metrics.temporal_smoothness = 0.7
        mock_metrics.quality_score = 0.8
        mock_metrics.silhouette_score = 0.2
        mock_assessor_instance.assess_hmm_regime_quality.return_value = mock_metrics
        
        # Créer l'optimiseur et la fonction objectif
        optimizer = RollingHMMOptimizer(self.config)
        
        # Mock du feature engineer
        mock_feature_engineer = Mock()
        mock_features = pd.DataFrame(np.random.randn(100, 5))
        mock_feature_engineer.generate_features.return_value = mock_features
        mock_feature_engineer.apply_pca.return_value = (mock_features, Mock(), 0.9)
        
        # Créer la fonction objectif
        objective = optimizer.create_objective_function(
            self.market_data,
            mock_feature_engineer,
            StickyHMMModel,
            mock_assessor_instance
        )
        
        # Vérifier que la fonction est callable
        self.assertTrue(callable(objective))
        
        # Tester avec des paramètres
        params = {
            'ewma_config_idx': 0,
            'n_components': 3,
            'pca_components': 3,
            'min_covar': 1e-3,
            'kappa': 10.0
        }
        
        score = objective(params)
        
        # Vérifier que le score est un nombre
        self.assertIsInstance(score, float)
    
    def test_create_fine_grid(self):
        """Test de la création d'une grille fine autour des meilleurs paramètres."""
        optimizer = RollingHMMOptimizer(self.config)
        
        # Meilleurs paramètres
        best_params = {
            'ewma_config_idx': 2,
            'n_components': 5,
            'pca_components': 4,
            'min_covar': 1e-3,
            'kappa': 10.0
        }
        
        # Créer la grille fine
        fine_grid = optimizer._create_fine_grid(best_params)
        
        # Vérifier la grille
        self.assertGreater(len(fine_grid), 0)
        self.assertLessEqual(len(fine_grid), 30)  # Limité à 30 combinaisons
        
        # Vérifier que tous les paramètres sont présents
        for params in fine_grid:
            self.assertIn('ewma_config_idx', params)
            self.assertIn('n_components', params)
            self.assertIn('pca_components', params)
            self.assertIn('min_covar', params)
            self.assertIn('kappa', params)
    
    def test_sample_around_best(self):
        """Test de l'échantillonnage autour des meilleurs paramètres."""
        optimizer = RollingHMMOptimizer(self.config)
        
        # Meilleurs paramètres
        best_params = {
            'ewma_config_idx': 2,
            'n_components': 5,
            'pca_components': 4,
            'min_covar': 1e-3,
            'kappa': 10.0
        }
        
        # Échantillonner autour
        sampled_params = optimizer._sample_around_best(best_params)
        
        # Vérifier que tous les paramètres sont présents
        self.assertIn('ewma_config_idx', sampled_params)
        self.assertIn('n_components', sampled_params)
        self.assertIn('pca_components', sampled_params)
        self.assertIn('min_covar', sampled_params)
        self.assertIn('kappa', sampled_params)
        
        # Vérifier que les paramètres sont dans les plages valides
        self.assertGreaterEqual(sampled_params['ewma_config_idx'], 0)
        self.assertLessEqual(sampled_params['ewma_config_idx'], 5)
        self.assertGreaterEqual(sampled_params['n_components'], 4)
        self.assertLessEqual(sampled_params['n_components'], 6)
        self.assertGreaterEqual(sampled_params['pca_components'], 3)
        self.assertLessEqual(sampled_params['pca_components'], 5)


class TestRollingHMMRegimeDiscoveryStep(unittest.TestCase):
    """Tests pour la classe RollingHMMRegimeDiscoveryStep."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        self.step = RollingHMMRegimeDiscoveryStep()
        
        # Données de marché synthétiques
        np.random.seed(42)
        n_samples = 100
        self.market_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'high': 100 + np.cumsum(np.random.normal(0.1, 1, n_samples)),
            'low': 100 + np.cumsum(np.random.normal(-0.1, 1, n_samples)),
            'close': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        # S'assurer que high >= close >= low
        self.market_data['high'] = np.maximum(self.market_data['high'], self.market_data['close'])
        self.market_data['low'] = np.minimum(self.market_data['low'], self.market_data['close'])
        
        # Configuration de base
        self.config = {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1h',
            'regime_timeframe': '1h',
            'execution_mode': 'light',
            'enable_auto_tuning': False,  # Désactivé pour les tests
            'rolling_hmm_params': {
                'ewma_config_idx': 0,
                'n_components': 3,
                'pca_components': 3,
                'min_covar': 1e-3,
                'kappa': 10.0
            }
        }
    
    def test_initialization(self):
        """Test de l'initialisation de RollingHMMRegimeDiscoveryStep."""
        self.assertEqual(self.step.step_name, "rolling_hmm_regime_discovery")
        self.assertIsNotNone(self.step.logger)
        self.assertIsNone(self.step._quality_assessor)
        self.assertIsNone(self.step.hardware_manager)
    
    def test_validate_config(self):
        """Test de la validation de la configuration."""
        # Configuration valide
        self.step._validate_config(self.config)
        
        # Configuration invalide (manque symbol)
        invalid_config = self.config.copy()
        del invalid_config['symbol']
        
        with self.assertRaises(ValueError):
            self.step._validate_config(invalid_config)
    
    def test_get_feature_config(self):
        """Test de la récupération de la configuration de feature engineering."""
        feature_config = self.step._get_feature_config(self.config)
        
        self.assertIsInstance(feature_config, FeatureEngineeringConfig)
        self.assertEqual(len(feature_config.ewma_configs), len(DEFAULT_EWMA_CONFIGS))
        self.assertTrue(feature_config.use_log_returns)
        self.assertTrue(feature_config.use_volatility_features)
        self.assertTrue(feature_config.use_trend_features)
        self.assertTrue(feature_config.use_volume_features)
    
    def test_get_hpo_config(self):
        """Test de la récupération de la configuration HPO."""
        hpo_config = self.step._get_hpo_config(self.config)
        
        self.assertIsInstance(hpo_config, HPOConfig)
        self.assertEqual(hpo_config.final_refinement_trials, 20)  # Mode light
        self.assertEqual(hpo_config.cv_folds, 3)  # Mode light
    
    def test_apply_execution_mode_filter(self):
        """Test du filtrage des données selon le mode d'exécution."""
        # Mode full (pas de filtrage)
        filtered_full = self.step._apply_execution_mode_filter(
            self.market_data, 'full', '1h'
        )
        self.assertEqual(len(filtered_full), len(self.market_data))
        
        # Mode light (180 jours)
        filtered_light = self.step._apply_execution_mode_filter(
            self.market_data, 'light', '1h'
        )
        self.assertLessEqual(len(filtered_light), 180 * 24)  # 180 jours * 24 heures
        
        # Mode blank (20 jours)
        filtered_blank = self.step._apply_execution_mode_filter(
            self.market_data, 'blank', '1h'
        )
        self.assertLessEqual(len(filtered_blank), 20 * 24)  # 20 jours * 24 heures
    
    @patch('src.training.steps.market_analysis.rolling_hmm_clustering.rolling_hmm_regime_discovery_step.ClusterQualityAssessor')
    async def test_execute(self, mock_quality_assessor):
        """Test de l'exécution de l'étape."""
        # Mock du quality assessor
        mock_assessor_instance = Mock()
        mock_quality_assessor.return_value = mock_assessor_instance
        
        # Mock des métriques
        mock_metrics = Mock()
        mock_metrics.to_dict.return_value = {
            'quality_score': 0.8,
            'silhouette_score': 0.2,
            'davies_bouldin_score': 0.5,
            'temporal_smoothness': 0.7,
            'regime_persistence': 10.0
        }
        mock_assessor_instance.assess_hmm_regime_quality.return_value = mock_metrics
        
        # Mock de l'artifact manager
        self.step.artifact_manager = Mock()
        self.step.artifact_manager.set_context = Mock()
        self.step.artifact_manager.save_artifact = Mock()
        
        # Ajouter les données de marché directement à la configuration
        self.config['market_data'] = self.market_data
        
        # Exécuter l'étape
        result = await self.step.execute(self.config)
        
        # Vérifier les résultats
        self.assertTrue(result['success'])
        self.assertIn('artifacts', result)
        self.assertIn('metrics', result)
        self.assertIn('execution_time', result)
        self.assertIn('n_regimes', result)
        self.assertEqual(result['n_regimes'], 3)


class TestEndToEnd(unittest.TestCase):
    """Test de bout en bout pour l'ensemble du pipeline."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Données de marché synthétiques
        np.random.seed(42)
        n_samples = 200
        self.market_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'high': 100 + np.cumsum(np.random.normal(0.1, 1, n_samples)),
            'low': 100 + np.cumsum(np.random.normal(-0.1, 1, n_samples)),
            'close': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        # S'assurer que high >= close >= low
        self.market_data['high'] = np.maximum(self.market_data['high'], self.market_data['close'])
        self.market_data['low'] = np.minimum(self.market_data['low'], self.market_data['close'])
    
    @patch('src.training.steps.market_analysis.rolling_hmm_clustering.rolling_hmm_regime_discovery_step.ClusterQualityAssessor')
    async def test_end_to_end_pipeline(self, mock_quality_assessor):
        """Test complet du pipeline de bout en bout."""
        # Mock du quality assessor
        mock_assessor_instance = Mock()
        mock_quality_assessor.return_value = mock_assessor_instance
        
        # Mock des métriques
        mock_metrics = Mock()
        mock_metrics.to_dict.return_value = {
            'quality_score': 0.8,
            'silhouette_score': 0.2,
            'davies_bouldin_score': 0.5,
            'temporal_smoothness': 0.7,
            'regime_persistence': 10.0
        }
        mock_assessor_instance.assess_hmm_regime_quality.return_value = mock_metrics
        
        # 1. Feature Engineering
        feature_config = FeatureEngineeringConfig(
            ewma_configs=[EWMAConfig(short_window=8, long_window=16, name="8+16")],
            enable_vectorbt_optimization=False,
            enable_hardware_optimization=False,
            enable_numba_jit=False
        )
        feature_engineer = RollingHMMFeatureEngineer(feature_config)
        features = feature_engineer.generate_features(self.market_data)
        
        # Vérifier les caractéristiques
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        
        # 2. PCA
        features_pca, pca_model, explained_var = feature_engineer.apply_pca(
            features, n_components=3
        )
        
        # Vérifier l'ACP
        self.assertEqual(features_pca.shape[1], 3)
        self.assertGreater(explained_var, 0)
        
        # 3. HMM Model
        hmm_config = StickyHMMConfig(
            n_components=3,
            n_iter=50,  # Réduit pour les tests
            covariance_type='diag',
            min_covar=1e-3,
            kappa=10.0,
            use_sticky_priors=True,
            kmeans_init=True,
            random_state=42
        )
        hmm_model = StickyHMMModel(hmm_config)
        hmm_model.fit(features_pca.values)
        
        # Vérifier le modèle
        self.assertTrue(hmm_model.is_fitted)
        
        # 4. Prédiction
        regime_labels = hmm_model.predict(features_pca.values)
        regime_probs = hmm_model.predict_proba(features_pca.values)
        
        # Vérifier les prédictions
        self.assertEqual(len(regime_labels), len(features_pca))
        self.assertEqual(regime_probs.shape, (len(features_pca), 3))
        
        # 5. Qualité
        forward_returns = self.market_data['close'].pct_change().shift(-1)
        forward_returns = forward_returns.loc[features_pca.index]
        
        metrics = mock_assessor_instance.assess_hmm_regime_quality(
            regime_labels=regime_labels,
            feature_data=features_pca,
            transition_matrix=hmm_model.get_transition_matrix(),
            hmm_model=None,
            forward_returns=forward_returns,
            timestamps=features_pca.index,
            timeframe='1h',
            min_regime_size=10,
            run_validators=True,
            temporal_sensitivity_mode="standard"
        )
        
        # Vérifier les métriques
        self.assertIsNotNone(metrics)


if __name__ == '__main__':
    # Exécuter tous les tests
    unittest.main(verbosity=2)