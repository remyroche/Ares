#!/usr/bin/env python3
"""
Script d'exécution pour les tests du module Rolling HMM Clustering

Ce script permet d'exécuter les tests de différentes manières :
- Tous les tests ensemble
- Par classe de test spécifique
- Avec rapport de couverture de code
- Avec sortie détaillée ou concise
"""

import sys
import os
import unittest
import argparse
from pathlib import Path

# Ajouter le répertoire racine au chemin Python
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Importer les modules de test
from src.training.steps.market_analysis.rolling_hmm_clustering.test_rolling_hmm_clustering import (
    TestFeatureEngineering,
    TestStickyHMMModel,
    TestHPOConfig,
    TestRollingHMMRegimeDiscoveryStep,
    TestEndToEnd
)


def run_all_tests(verbosity=2):
    """Exécuter tous les tests."""
    print("🧪 Exécution de tous les tests pour Rolling HMM Clustering")
    print("=" * 70)
    
    # Créer une suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter toutes les classes de test
    test_classes = [
        TestFeatureEngineering,
        TestStickyHMMModel,
        TestHPOConfig,
        TestRollingHMMRegimeDiscoveryStep,
        TestEndToEnd
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_feature_engineering_tests(verbosity=2):
    """Exécuter uniquement les tests de feature engineering."""
    print("🧪 Exécution des tests de Feature Engineering")
    print("=" * 70)
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatureEngineering)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_sticky_hmm_tests(verbosity=2):
    """Exécuter uniquement les tests du modèle HMM sticky."""
    print("🧪 Exécution des tests du modèle HMM Sticky")
    print("=" * 70)
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestStickyHMMModel)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_hpo_config_tests(verbosity=2):
    """Exécuter uniquement les tests de configuration HPO."""
    print("🧪 Exécution des tests de configuration HPO")
    print("=" * 70)
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestHPOConfig)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_regime_discovery_tests(verbosity=2):
    """Exécuter uniquement les tests de découverte de régimes."""
    print("🧪 Exécution des tests de découverte de régimes")
    print("=" * 70)
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRollingHMMRegimeDiscoveryStep)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_end_to_end_tests(verbosity=2):
    """Exécuter uniquement les tests de bout en bout."""
    print("🧪 Exécution des tests de bout en bout")
    print("=" * 70)
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEnd)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_with_coverage():
    """Exécuter les tests avec rapport de couverture de code."""
    try:
        import coverage
        
        print("🧪 Exécution des tests avec rapport de couverture de code")
        print("=" * 70)
        
        # Créer une instance de coverage
        cov = coverage.Coverage(source=[
            'src.training.steps.market_analysis.rolling_hmm_clustering.feature_engineering',
            'src.training.steps.market_analysis.rolling_hmm_clustering.sticky_hmm_model',
            'src.training.steps.market_analysis.rolling_hmm_clustering.hpo_config',
            'src.training.steps.market_analysis.rolling_hmm_clustering.rolling_hmm_regime_discovery_step'
        ])
        
        # Démarrer la collecte de couverture
        cov.start()
        
        # Exécuter tous les tests
        success = run_all_tests(verbosity=1)
        
        # Arrêter la collecte de couverture
        cov.stop()
        cov.save()
        
        # Générer le rapport
        print("\n📊 Rapport de couverture de code:")
        print("-" * 70)
        cov.report()
        
        # Générer le rapport HTML
        cov.html_report(directory='test_output/rolling_hmm_coverage')
        print(f"\n📄 Rapport HTML généré dans: test_output/rolling_hmm_coverage/index.html")
        
        return success
        
    except ImportError:
        print("❌ Le module 'coverage' n'est pas installé. Installez-le avec: pip install coverage")
        return False


def check_syntax():
    """Vérifier la syntaxe des fichiers Python."""
    print("🔍 Vérification de la syntaxe des fichiers Python")
    print("=" * 70)
    
    # Fichiers à vérifier
    files_to_check = [
        'feature_engineering.py',
        'sticky_hmm_model.py',
        'hpo_config.py',
        'rolling_hmm_regime_discovery_step.py',
        'test_rolling_hmm_clustering.py'
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✅ {file_path}: Syntaxe OK")
        except SyntaxError as e:
            print(f"❌ {file_path}: Erreur de syntaxe à la ligne {e.lineno}: {e.msg}")
            all_good = False
        except Exception as e:
            print(f"❌ {file_path}: Erreur: {e}")
            all_good = False
    
    return all_good


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Exécuter les tests pour Rolling HMM Clustering')
    parser.add_argument(
        '--module', '-m',
        choices=['all', 'feature', 'hmm', 'hpo', 'discovery', 'e2e'],
        default='all',
        help='Module à tester (défaut: all)'
    )
    parser.add_argument(
        '--verbosity', '-v',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Niveau de verbosité (0=concis, 1=normal, 2=détaillé)'
    )
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Générer un rapport de couverture de code'
    )
    parser.add_argument(
        '--syntax', '-s',
        action='store_true',
        help='Vérifier uniquement la syntaxe des fichiers'
    )
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie si nécessaire
    os.makedirs('test_output', exist_ok=True)
    
    # Vérifier la syntaxe si demandé
    if args.syntax:
        success = check_syntax()
        sys.exit(0 if success else 1)
    
    # Exécuter avec couverture si demandé
    if args.coverage:
        success = run_with_coverage()
        sys.exit(0 if success else 1)
    
    # Exécuter les tests selon le module choisi
    if args.module == 'all':
        success = run_all_tests(args.verbosity)
    elif args.module == 'feature':
        success = run_feature_engineering_tests(args.verbosity)
    elif args.module == 'hmm':
        success = run_sticky_hmm_tests(args.verbosity)
    elif args.module == 'hpo':
        success = run_hpo_config_tests(args.verbosity)
    elif args.module == 'discovery':
        success = run_regime_discovery_tests(args.verbosity)
    elif args.module == 'e2e':
        success = run_end_to_end_tests(args.verbosity)
    
    # Afficher le résultat final
    print("\n" + "=" * 70)
    if success:
        print("✅ Tous les tests ont réussi!")
        sys.exit(0)
    else:
        print("❌ Certains tests ont échoué!")
        sys.exit(1)


if __name__ == '__main__':
    main()