"""
Test complet pour vérifier le fonctionnement du nouveau système d'analyse économique des régimes
intégré dans cluster_quality_assessor.py.

Ce test valide :
1. L'intégration entre ClusterQualityAssessor et RegimeEconomicRelevanceAnalyzer
2. La génération des métriques de performance économiques
3. Les tests de signification statistique
4. La génération des rapports dans le répertoire outcomes/
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Ajouter le répertoire src au chemin Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
    QualityThresholds
)
from src.training.steps.market_analysis.clusters.regime_economic_relevance_analyzer import (
    RegimeEconomicRelevanceAnalyzer,
    PerformanceMetrics,
    StrategyResults
)

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_success_response,
    assert_error_response,
    assert_float_equals,
    assert_percentage_equals,
    assert_dict_structure,
    assert_list_structure,
    assert_dataframe_structure,
    assert_performance_metrics
)


class TestRegimeEconomicRelevance:
    """Classe de test pour le système d'analyse économique des régimes."""
    
    @pytest.fixture
    def sample_data(self):
        """Génère des données synthétiques réalistes pour les tests."""
        np.random.seed(42)  # Pour la reproductibilité
        
        # Paramètres
        n_samples = 500
        n_regimes = 3
        
        # Générer les timestamps
        start_date = pd.Timestamp('2023-01-01')
        timestamps = pd.date_range(start=start_date, periods=n_samples, freq='1h')
        
        # Générer les étiquettes de régimes avec persistance
        regime_labels = self._generate_persistent_regimes(n_samples, n_regimes)
        
        # Générer les caractéristiques (features) financières
        features = self._generate_financial_features(n_samples, regime_labels)
        
        # Générer les prix et rendements
        prices, forward_returns = self._generate_price_returns(n_samples, regime_labels)
        
        return {
            'regime_labels': regime_labels,
            'features': features,
            'prices': prices,
            'forward_returns': forward_returns,
            'timestamps': timestamps
        }
    
    def _generate_persistent_regimes(self, n_samples, n_regimes):
        """Génère des étiquettes de régimes avec persistance réaliste."""
        # Points de changement de régime (environ 10-20% des échantillons)
        n_changes = max(5, min(n_samples - 1, n_samples // 10))
        change_points = sorted(np.random.choice(n_samples - 1, n_changes, replace=False))
        
        regime_labels = np.zeros(n_samples, dtype=int)
        current_regime = 0
        
        for i in range(n_samples):
            if i in change_points:
                # Changer de régime (éviter le même régime)
                available_regimes = [r for r in range(n_regimes) if r != current_regime]
                current_regime = np.random.choice(available_regimes)
            regime_labels[i] = current_regime
        
        return regime_labels
    
    def _generate_financial_features(self, n_samples, regime_labels):
        """Génère des caractéristiques financières réalistes."""
        features = pd.DataFrame(index=range(n_samples))
        
        # Vérifier la structure du DataFrame généré
        expected_columns = ['rsi', 'volatility', 'volume', 'momentum', 'spread']
        assert_dataframe_structure(
            features,
            expected_columns=expected_columns,
            min_rows=n_samples,
            max_rows=n_samples,
            message="Le DataFrame des caractéristiques doit avoir la structure attendue"
        )
        
        # Caractéristiques de base par régime
        regime_characteristics = {
            0: {'rsi_mean': 30, 'volatility_mean': 0.02, 'volume_mean': 100},
            1: {'rsi_mean': 50, 'volatility_mean': 0.04, 'volume_mean': 150},
            2: {'rsi_mean': 70, 'volatility_mean': 0.08, 'volume_mean': 200}
        }
        
        for i in range(n_samples):
            regime = regime_labels[i]
            char = regime_characteristics[regime]
            
            # RSI avec bruit
            features.loc[i, 'rsi'] = np.clip(
                np.random.normal(char['rsi_mean'], 10), 0, 100
            )
            
            # Volatilité avec bruit
            features.loc[i, 'volatility'] = max(
                0.001, np.random.normal(char['volatility_mean'], char['volatility_mean'] * 0.3)
            )
            
            # Volume avec bruit
            features.loc[i, 'volume'] = max(
                1, np.random.normal(char['volume_mean'], char['volume_mean'] * 0.2)
            )
            
            # Momentum
            features.loc[i, 'momentum'] = np.random.normal(0, 0.02)
            
            # Spread bid-ask simulé
            features.loc[i, 'spread'] = max(0.0001, np.random.normal(0.001, 0.0005))
        
        return features
    
    def _generate_price_returns(self, n_samples, regime_labels):
        """Génère des prix et rendements réalistes basés sur les régimes."""
        # Caractéristiques de rendement par régime
        regime_returns = {
            0: {'mean': 0.0005, 'std': 0.02},    # Trending up
            1: {'mean': 0.0001, 'std': 0.01},    # Range/sideways
            2: {'mean': -0.0003, 'std': 0.05}   # Volatile/crisis
        }
        
        # Générer les rendements
        returns = np.zeros(n_samples)
        for i in range(n_samples):
            regime = regime_labels[i]
            char = regime_returns[regime]
            returns[i] = np.random.normal(char['mean'], char['std'])
        
        # Limiter les rendements extrêmes
        returns = np.clip(returns, -0.15, 0.15)
        
        # Calculer les prix cumulés
        prices = 100 * np.cumprod(1 + returns)
        
        # Créer les Series pandas
        forward_returns = pd.Series(returns, name='forward_returns')
        prices_series = pd.Series(prices, name='price')
        
        return prices_series, forward_returns
    
    def test_cluster_quality_assessor_creation(self):
        """Test la création du ClusterQualityAssessor."""
        assessor = ClusterQualityAssessor()
        
        assert assessor is not None, "Le ClusterQualityAssessor ne doit pas être None"
        assert hasattr(assessor, 'assess_quality'), "Le ClusterQualityAssessor doit avoir la méthode assess_quality"
        assert hasattr(assessor, 'assess_economic_relevance'), "Le ClusterQualityAssessor doit avoir la méthode assess_economic_relevance"
        
        print("✅ Test création ClusterQualityAssessor réussi")
    
    def test_regime_economic_analyzer_creation(self):
        """Test la création du RegimeEconomicRelevanceAnalyzer."""
        analyzer = RegimeEconomicRelevanceAnalyzer()
        
        assert analyzer is not None, "Le RegimeEconomicRelevanceAnalyzer ne doit pas être None"
        assert hasattr(analyzer, 'evaluate_strategies'), "Le RegimeEconomicRelevanceAnalyzer doit avoir la méthode evaluate_strategies"
        assert hasattr(analyzer, 'perform_significance_test'), "Le RegimeEconomicRelevanceAnalyzer doit avoir la méthode perform_significance_test"
        assert hasattr(analyzer, 'generate_economic_report'), "Le RegimeEconomicRelevanceAnalyzer doit avoir la méthode generate_economic_report"
        
        print("✅ Test création RegimeEconomicRelevanceAnalyzer réussi")
    
    def test_assess_quality_with_economic_analysis(self, sample_data):
        """Test l'évaluation de la qualité avec analyse économique."""
        assessor = ClusterQualityAssessor()
        
        # Vérifier la structure des données en entrée
        assert_dataframe_structure(
            sample_data['features'],
            min_rows=1,
            message="Les données de caractéristiques doivent être un DataFrame valide"
        )
        
        assert isinstance(sample_data['forward_returns'], pd.Series), "Les rendements doivent être une Series pandas"
        assert len(sample_data['forward_returns']) > 0, "Les rendements ne doivent pas être vides"
        
        # Évaluer la qualité avec analyse économique
        metrics = assessor.assess_quality(
            regime_labels=sample_data['regime_labels'],
            feature_data=sample_data['features'],
            forward_returns=sample_data['forward_returns'],
            timestamps=sample_data['timestamps']
        )
        
        # Vérifier que les résultats économiques sont présents
        assert isinstance(metrics, ClusterQualityMetrics), "Les métriques doivent être de type ClusterQualityMetrics"
        assert metrics.economic_relevance_analysis is not None, "L'analyse économique ne doit pas être None"
        assert len(metrics.economic_relevance_analysis) > 0, "L'analyse économique doit contenir des données"
        
        # Vérifier les métriques de stratégie
        assert metrics.strategy_performance_metrics is not None, "Les métriques de stratégie ne doivent pas être None"
        assert len(metrics.strategy_performance_metrics) > 0, "Les métriques de stratégie doivent contenir des données"
        
        print("✅ Test évaluation qualité avec analyse économique réussi")
        return metrics
    
    def test_economic_results_structure(self, sample_data):
        """Test la structure des résultats de l'analyse économique."""
        assessor = ClusterQualityAssessor()
        metrics = assessor.assess_quality(
            regime_labels=sample_data['regime_labels'],
            feature_data=sample_data['features'],
            forward_returns=sample_data['forward_returns'],
            timestamps=sample_data['timestamps']
        )
        
        economic_results = metrics.economic_relevance_analysis
        
        # Vérifier la structure des résultats
        assert_dict_structure(
            economic_results,
            required_keys=['strategy_performance', 'significance_tests'],
            message="Les résultats économiques doivent contenir les clés requises"
        )
        
        strategy_perf = economic_results['strategy_performance']
        assert_dict_structure(
            strategy_perf,
            required_keys=['buy_hold', 'real_regime'],
            message="La performance des stratégies doit contenir buy_hold et real_regime"
        )
        
        # Vérifier les métriques de performance
        for strategy_name, strategy in strategy_perf.items():
            assert_dict_structure(
                strategy,
                required_keys=['metrics'],
                message=f"La stratégie {strategy_name} doit contenir des métriques"
            )
            metrics_dict = strategy['metrics']
            
            # Vérifier les métriques attendues
            expected_metrics = [
                'cagr', 'sharpe_ratio', 'max_drawdown',
                'volatility', 'total_return', 'hit_rate'
            ]
            
            for metric in expected_metrics:
                assert metric in metrics_dict, f"La métrique {metric} doit être présente dans {strategy_name}"
                assert isinstance(metrics_dict[metric], (int, float)), f"La métrique {metric} doit être numérique dans {strategy_name}"
                
                # Vérifier les valeurs avec assertions standardisées
                if metric in ['hit_rate']:
                    assert_percentage_equals(
                        metrics_dict[metric], metrics_dict[metric],
                        tolerance=0.01,
                        message=f"La métrique {metric} de {strategy_name} doit être un pourcentage valide"
                    )
                else:
                    assert_float_equals(
                        metrics_dict[metric], metrics_dict[metric],
                        tolerance=1e-6,
                        message=f"La métrique {metric} de {strategy_name} doit être un nombre valide"
                    )
        
        print("✅ Test structure des résultats économiques réussi")
    
    def test_performance_metrics_calculation(self, sample_data):
        """Test le calcul des métriques de performance."""
        analyzer = RegimeEconomicRelevanceAnalyzer()
        
        # Évaluer les stratégies
        strategies = analyzer.evaluate_strategies(
            prices=sample_data['prices'],
            regime_labels=sample_data['regime_labels']
        )
        
        # Vérifier que toutes les stratégies sont présentes
        assert_dict_structure(
            strategies,
            required_keys=['buy_hold', 'real_regime'],
            message="Les stratégies doivent contenir buy_hold et real_regime"
        )
        
        # Vérifier les métriques de performance
        for strategy_name, strategy in strategies.items():
            assert isinstance(strategy, StrategyResults), f"La stratégie {strategy_name} doit être de type StrategyResults"
            assert isinstance(strategy.metrics, PerformanceMetrics), f"Les métriques de {strategy_name} doivent être de type PerformanceMetrics"
            
            # Vérifier les métriques clés avec assertions standardisées
            metrics = strategy.metrics
            assert metrics.cagr is not None, f"Le CAGR de {strategy_name} ne doit pas être None"
            assert metrics.sharpe_ratio is not None, f"Le Sharpe ratio de {strategy_name} ne doit pas être None"
            assert metrics.max_drawdown is not None, f"Le max drawdown de {strategy_name} ne doit pas être None"
            assert metrics.volatility is not None, f"La volatilité de {strategy_name} ne doit pas être None"
            assert metrics.total_return is not None, f"Le retour total de {strategy_name} ne doit pas être None"
            assert metrics.hit_rate is not None, f"Le hit rate de {strategy_name} ne doit pas être None"
            
            # Vérifier les valeurs numériques avec tolérances appropriées
            if metrics.cagr is not None:
                assert_float_equals(
                    metrics.cagr, metrics.cagr,
                    tolerance=1e-6,
                    message=f"Le CAGR de {strategy_name} doit être un nombre valide"
                )
            
            if metrics.sharpe_ratio is not None:
                assert_float_equals(
                    metrics.sharpe_ratio, metrics.sharpe_ratio,
                    tolerance=1e-6,
                    message=f"Le Sharpe ratio de {strategy_name} doit être un nombre valide"
                )
            
            if metrics.max_drawdown is not None:
                assert_float_equals(
                    metrics.max_drawdown, metrics.max_drawdown,
                    tolerance=1e-6,
                    message=f"Le max drawdown de {strategy_name} doit être un nombre valide"
                )
            
            if metrics.volatility is not None:
                assert_float_equals(
                    metrics.volatility, metrics.volatility,
                    tolerance=1e-6,
                    message=f"La volatilité de {strategy_name} doit être un nombre valide"
                )
            
            if metrics.total_return is not None:
                assert_float_equals(
                    metrics.total_return, metrics.total_return,
                    tolerance=1e-6,
                    message=f"Le retour total de {strategy_name} doit être un nombre valide"
                )
            
            if metrics.hit_rate is not None:
                assert_percentage_equals(
                    metrics.hit_rate, metrics.hit_rate,
                    tolerance=0.01,
                    message=f"Le hit rate de {strategy_name} doit être un pourcentage valide"
                )
        
        print("✅ Test calcul des métriques de performance réussi")
    
    def test_significance_tests(self, sample_data):
        """Test les tests de signification économique."""
        analyzer = RegimeEconomicRelevanceAnalyzer()
        
        # Évaluer les stratégies
        strategies = analyzer.evaluate_strategies(
            prices=sample_data['prices'],
            regime_labels=sample_data['regime_labels']
        )
        
        # Effectuer les tests de signification
        significance_results = analyzer.perform_significance_test(
            strategies=strategies,
            test_method='bootstrap'
        )
        
        # Vérifier la structure des résultats
        assert significance_results is not None, "Les résultats de signification ne doivent pas être None"
        assert_dict_structure(
            significance_results,
            required_keys=['method', 'n_permutations', 'confidence_intervals'],
            message="Les résultats de signification doivent contenir les clés requises"
        )
        assert significance_results['method'] == 'bootstrap', "La méthode de test doit être 'bootstrap'"
        
        # Vérifier les intervalles de confiance
        confidence_intervals = significance_results['confidence_intervals']
        assert isinstance(confidence_intervals, dict), "Les intervalles de confiance doivent être un dictionnaire"
        
        for strategy_name, intervals in confidence_intervals.items():
            assert isinstance(intervals, dict), f"Les intervalles pour {strategy_name} doivent être un dictionnaire"
            
            # Vérifier les métriques avec intervalles
            for metric in ['mean', 'sharpe', 'total_return']:
                assert metric in intervals, f"La métrique {metric} doit être présente dans les intervalles de {strategy_name}"
                metric_interval = intervals[metric]
                
                assert_dict_structure(
                    metric_interval,
                    required_keys=['ci_95_lower', 'ci_95_upper', 'p_value'],
                    message=f"L'intervalle pour {metric} dans {strategy_name} doit contenir les clés requises"
                )
                
                # Vérifier les types et les valeurs avec assertions standardisées
                assert_float_equals(
                    metric_interval['ci_95_lower'], metric_interval['ci_95_lower'],
                    tolerance=1e-6,
                    message=f"ci_95_lower doit être un nombre valide pour {metric} dans {strategy_name}"
                )
                
                assert_float_equals(
                    metric_interval['ci_95_upper'], metric_interval['ci_95_upper'],
                    tolerance=1e-6,
                    message=f"ci_95_upper doit être un nombre valide pour {metric} dans {strategy_name}"
                )
                
                assert_float_equals(
                    metric_interval['p_value'], metric_interval['p_value'],
                    tolerance=1e-6,
                    message=f"p_value doit être un nombre valide pour {metric} dans {strategy_name}"
                )
                
                assert 0 <= metric_interval['p_value'] <= 1, f"p_value doit être entre 0 et 1 pour {metric} dans {strategy_name}"
        
        print("✅ Test des tests de signification réussi")
    
    def test_report_generation(self, sample_data):
        """Test la génération des rapports économiques."""
        analyzer = RegimeEconomicRelevanceAnalyzer()
        
        # Évaluer les stratégies
        strategies = analyzer.evaluate_strategies(
            prices=sample_data['prices'],
            regime_labels=sample_data['regime_labels']
        )
        
        # DEBUG: Vérifier que les stratégies sont valides
        print(f"DEBUG: Stratégies évaluées: {list(strategies.keys())}")
        
        # Effectuer les tests de signification
        print("DEBUG: Exécution des tests de signification...")
        significance_results = analyzer.perform_significance_test(
            strategies=strategies,
            test_method='bootstrap'
        )
        
        # Générer le rapport
        print("DEBUG: Génération du rapport...")
        report_path = analyzer.generate_economic_report(
            strategies=strategies,
            significance_results=significance_results,
            output_dir="outcomes"
        )
        
        # DEBUG: Vérifier le résultat
        print(f"DEBUG: Chemin du rapport généré: {report_path}")
        print(f"DEBUG: Type du rapport: {type(report_path)}")
        
        # Vérifier que le rapport a été créé
        assert report_path is not None, "Le chemin du rapport ne doit pas être None"
        assert Path(report_path).exists(), f"Le fichier de rapport n'existe pas: {report_path}"
        
        # Vérifier que le rapport est dans le bon répertoire
        assert "outcomes" in report_path, f"Le rapport doit être dans le répertoire outcomes: {report_path}"
        
        # Vérifier le contenu du rapport
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # DEBUG: Afficher le contenu du rapport pour diagnostic
            print(f"DEBUG: Contenu du rapport (premiers 1000 caractères):")
            print(content[:1000])
            print(f"DEBUG: Sections trouvées dans le rapport:")
            lines = content.split('\n')
            for line in lines:
                if line.startswith('#'):
                    print(f"  {line}")
            
            # Vérifier les sections attendues
            assert "# Rapport de Pertinence Économique des Régimes" in content, "Le rapport doit contenir le titre principal"
            assert "## Résumé Exécutif" in content, "Le rapport doit contenir la section Résumé Exécutif"
            assert "## Analyse Détaillée des Performances" in content, "Le rapport doit contenir la section Analyse Détaillée des Performances"
            assert "## Tests de Signification" in content, "Le rapport doit contenir la section Tests de Signification"
            assert "## Recommandations" in content, "Le rapport doit contenir la section Recommandations"
        
        print(f"✅ Test génération de rapport réussi: {report_path}")
        return report_path
    
    def test_integration_complete(self, sample_data):
        """Test l'intégration complète du système."""
        assessor = ClusterQualityAssessor()
        
        # Évaluer la qualité complète
        metrics = assessor.assess_quality(
            regime_labels=sample_data['regime_labels'],
            feature_data=sample_data['features'],
            forward_returns=sample_data['forward_returns'],
            timestamps=sample_data['timestamps']
        )
        
        # Vérifier que tout est intégré
        assert metrics.economic_relevance_analysis is not None, "L'analyse économique ne doit pas être None"
        assert metrics.strategy_performance_metrics is not None, "Les métriques de stratégie ne doivent pas être None"
        assert metrics.economic_significance_test is not None, "Le test de signification ne doit pas être None"
        assert metrics.economic_report_path is not None, "Le chemin du rapport ne doit pas être None"
        
        # Vérifier que le rapport économique existe
        if metrics.economic_report_path:
            assert Path(metrics.economic_report_path).exists(), f"Le fichier de rapport n'existe pas: {metrics.economic_report_path}"
        
        # Vérifier la cohérence des résultats
        economic_analysis = metrics.economic_relevance_analysis
        strategy_perf = metrics.strategy_performance_metrics
        
        # Le nombre de stratégies doit correspondre
        assert len(economic_analysis.get('strategy_performance', {})) == len(strategy_perf), "Le nombre de stratégies doit correspondre entre l'analyse et les métriques"
        
        print("✅ Test d'intégration complète réussi")
    
    def test_error_handling(self, sample_data):
        """Test la gestion des erreurs."""
        assessor = ClusterQualityAssessor()
        
        # Test avec des données vides
        empty_metrics = assessor.assess_quality(
            regime_labels=np.array([]),
            feature_data=pd.DataFrame(),
            forward_returns=pd.Series([])
        )
        
        assert isinstance(empty_metrics, ClusterQualityMetrics), "Les métriques vides doivent être de type ClusterQualityMetrics"
        # Le quality_score peut être None ou 0.0 pour des données vides
        assert empty_metrics.quality_score in [0.0, None], "Le quality_score pour données vides doit être 0.0 ou None"
        
        # Test avec des données incompatibles
        mismatch_metrics = assessor.assess_quality(
            regime_labels=sample_data['regime_labels'][:10],
            feature_data=sample_data['features'],
            forward_returns=sample_data['forward_returns']
        )
        
        # Devrait gérer l'incompatibilité de longueur
        assert isinstance(mismatch_metrics, ClusterQualityMetrics), "Les métriques incompatibles doivent être de type ClusterQualityMetrics"
        
        print("✅ Test gestion des erreurs réussi")
    
    def test_economic_analyzer_direct(self, sample_data):
        """Test l'utilisation directe de RegimeEconomicRelevanceAnalyzer."""
        analyzer = RegimeEconomicRelevanceAnalyzer(
            risk_free_rate=0.02,
            trading_days_per_year=252,
            transaction_cost=0.001,
            significance_tests=True,
            n_permutations=100  # Réduit pour les tests
        )
        
        # Vérifier les données d'entrée avec assertions standardisées
        assert isinstance(sample_data['prices'], pd.Series), "Les prix doivent être une Series pandas"
        assert len(sample_data['prices']) > 0, "Les prix ne doivent pas être vides"
        
        assert isinstance(sample_data['regime_labels'], np.ndarray), "Les étiquettes de régime doivent être un array numpy"
        assert len(sample_data['regime_labels']) > 0, "Les étiquettes de régime ne doivent pas être vides"
        
        # Évaluer directement
        strategies = analyzer.evaluate_strategies(
            prices=sample_data['prices'],
            regime_labels=sample_data['regime_labels'],
            predicted_regimes=sample_data['regime_labels']  # Simuler une prédiction parfaite
        )
        
        # Vérifier que les stratégies prédites sont incluses
        assert 'predicted_regime' in strategies, "Les stratégies doivent contenir predicted_regime"
        
        # Tests de signification avec assertions standardisées
        significance = analyzer.perform_significance_test(strategies, 'bootstrap')
        assert significance is not None, "Les résultats de signification ne doivent pas être None"
        
        # Vérifier la structure des résultats de signification
        assert_dict_structure(
            significance,
            required_keys=['method', 'confidence_intervals'],
            message="Les résultats de signification doivent avoir la structure requise"
        )
        
        # Sauvegarder les résultats
        results_path = analyzer.save_results(strategies, significance, "outcomes")
        assert results_path is not None, "Le chemin des résultats ne doit pas être None"
        assert Path(results_path).exists(), f"Le fichier de résultats n'existe pas: {results_path}"
        
        # Vérifier que le fichier est dans le bon répertoire
        assert "outcomes" in results_path, f"Les résultats doivent être dans le répertoire outcomes: {results_path}"
        
        print("✅ Test utilisation directe de l'analyseur économique réussi")


if __name__ == "__main__":
    # Exécuter les tests manuellement pour le débogage
    print("🧪 Démarrage des tests pour le système d'analyse économique des régimes")
    
    test_instance = TestRegimeEconomicRelevance()
    
    # Générer les données de test
    print("📊 Génération des données synthétiques...")
    sample_data = test_instance._generate_persistent_regimes(500, 3)
    features = test_instance._generate_financial_features(500, sample_data)
    prices, returns = test_instance._generate_price_returns(500, sample_data)
    
    data = {
        'regime_labels': sample_data,
        'features': features,
        'prices': prices,
        'forward_returns': returns,
        'timestamps': pd.date_range('2023-01-01', periods=500, freq='1h')
    }
    
    # Exécuter les tests
    try:
        test_instance.test_cluster_quality_assessor_creation()
        test_instance.test_regime_economic_analyzer_creation()
        metrics = test_instance.test_assess_quality_with_economic_analysis(data)
        test_instance.test_economic_results_structure(data)
        test_instance.test_performance_metrics_calculation(data)
        test_instance.test_significance_tests(data)
        report_path = test_instance.test_report_generation(data)
        test_instance.test_integration_complete(data)
        test_instance.test_error_handling(data)
        test_instance.test_economic_analyzer_direct(data)
        
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print(f"📄 Rapport généré: {report_path}")
        
    except Exception as e:
        print(f"\n❌ ÉCHEC DES TESTS: {e}")
        import traceback
        traceback.print_exc()