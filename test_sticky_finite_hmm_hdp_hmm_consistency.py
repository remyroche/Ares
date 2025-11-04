#!/usr/bin/env python3
"""
Test de Cohérence sticky_finite_hmm vs HDP-HMM

Ce script valide que sticky_finite_hmm utilise les mêmes données et fonctionnalités
que HDP-HMM après les corrections apportées.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Crée des données de test cohérentes pour les deux algorithmes."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Créer des données avec des régimes distincts
    data = []
    labels = []
    
    # Régime 1: 300 échantillons
    regime1_data = np.random.randn(300, n_features) + 2
    regime1_labels = np.zeros(300)
    
    # Régime 2: 400 échantillons
    regime2_data = np.random.randn(400, n_features) - 1
    regime2_labels = np.ones(400)
    
    # Régime 3: 300 échantillons
    regime3_data = np.random.randn(300, n_features) + 0.5
    regime3_labels = np.ones(300) * 2
    
    # Combiner toutes les données
    all_data = np.vstack([regime1_data, regime2_data, regime3_data])
    all_labels = np.concatenate([regime1_labels, regime2_labels, regime3_labels])
    
    # Créer un DataFrame avec index temporel
    index = pd.date_range(start='2024-01-01', periods=n_samples, freq='1H')
    df = pd.DataFrame(all_data, index=index, columns=[f'feature_{i}' for i in range(n_features)])
    
    return df, all_labels

def test_data_validation_consistency():
    """Teste que les deux algorithmes valident les données de manière cohérente."""
    print("🔍 Test de cohérence de validation des données...")
    
    try:
        from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import StickyFiniteHMMClusterer
        from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import HDPHMMClusterer
        
        # Créer les clusterers
        sticky_clusterer = StickyFiniteHMMClusterer()
        hdp_clusterer = HDPHMMClusterer()
        
        # Créer des données de test
        test_data, _ = create_test_data()
        
        print(f"   📊 Test avec {len(test_data)} échantillons, {test_data.shape[1]} features")
        
        # Tester la validation des données avec sticky_finite_hmm
        try:
            sticky_validated = sticky_clusterer._validate_input(test_data.values)
            sticky_validation_success = True
            print("   ✅ Sticky Finite HMM validation: PASS")
        except Exception as e:
            sticky_validation_success = False
            print(f"   ❌ Sticky Finite HMM validation: FAIL - {e}")
        
        # Tester la validation des données avec HDP-HMM
        try:
            hdp_validated = hdp_clusterer._validate_input(test_data.values)
            hdp_validation_success = True
            print("   ✅ HDP-HMM validation: PASS")
        except Exception as e:
            hdp_validation_success = False
            print(f"   ❌ HDP-HMM validation: FAIL - {e}")
        
        # Vérifier que les validations sont cohérentes
        if sticky_validation_success == hdp_validation_success:
            print("   ✅ Cohérence de validation: PASS")
            return True
        else:
            print("   ❌ Cohérence de validation: FAIL")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur lors du test de validation: {e}")
        return False

def test_preprocessing_consistency():
    """Teste que les deux algorithmes utilisent les mêmes paramètres de prétraitement."""
    print("🔧 Test de cohérence du prétraitement...")
    
    try:
        from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import StickyFiniteHMMConfig
        from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import HDPHMMConfig
        
        # Comparer les configurations
        sticky_config = StickyFiniteHMMConfig()
        hdp_config = HDPHMMConfig()
        
        # Vérifier les paramètres de PCA
        pca_consistency = (
            sticky_config.enable_pca == hdp_config.enable_pca and
            sticky_config.pca_components == hdp_config.pca_components
        )
        
        # Vérifier les paramètres de validation
        validation_consistency = (
            sticky_config.min_samples_required == hdp_config.min_samples_required and
            sticky_config.min_features_required == hdp_config.min_features_required and
            sticky_config.max_nan_ratio == hdp_config.max_nan_ratio
        )
        
        print(f"   📊 PCA: sticky={sticky_config.enable_pca}, hdp={hdp_config.enable_pca}")
        print(f"   📊 PCA Components: sticky={sticky_config.pca_components}, hdp={hdp_config.pca_components}")
        print(f"   📊 Min Samples: sticky={sticky_config.min_samples_required}, hdp={hdp_config.min_samples_required}")
        print(f"   📊 Min Features: sticky={sticky_config.min_features_required}, hdp={hdp_config.min_features_required}")
        
        if pca_consistency and validation_consistency:
            print("   ✅ Cohérence du prétraitement: PASS")
            return True
        else:
            print("   ❌ Cohérence du prétraitement: FAIL")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur lors du test de prétraitement: {e}")
        return False

def test_artifact_naming_consistency():
    """Teste que les artefacts sont nommés de manière cohérente."""
    print("📦 Test de cohérence du nommage des artefacts...")
    
    try:
        # Vérifier les artefacts principaux que les deux algorithmes doivent créer
        expected_artifacts = [
            "hdp_hmm_regime_labels",
            "hdp_hmm_regime_probabilities", 
            "hdp_hmm_cluster_statistics",
            "hdp_hmm_transition_matrix"
        ]
        
        # Vérifier que sticky_finite_hmm sauvegarde bien avec les noms compatibles
        sticky_step_path = "Ares/src/training/steps/market_analysis/sticky_finite_hmm_clustering/sticky_finite_hmm_regime_discovery_step.py"
        
        # Lire le fichier pour vérifier les artefacts
        with open(sticky_step_path, 'r') as f:
            content = f.read()
        
        # Vérifier la présence des artefacts compatibles
        artifact_checks = []
        for artifact in expected_artifacts:
            if artifact in content:
                artifact_checks.append(True)
                print(f"   ✅ Trouvé artefact compatible: {artifact}")
            else:
                artifact_checks.append(False)
                print(f"   ❌ Artefact compatible manquant: {artifact}")
        
        # Vérifier aussi les artefacts de compatibilité
        compatibility_artifacts = [
            "sticky_finite_hmm_regime_labels",
            "sticky_finite_hmm_regime_probabilities"
        ]
        
        compatibility_checks = []
        for artifact in compatibility_artifacts:
            if artifact in content:
                compatibility_checks.append(True)
                print(f"   ✅ Trouvé artefact de compatibilité: {artifact}")
            else:
                compatibility_checks.append(False)
                print(f"   ❌ Artefact de compatibilité manquant: {artifact}")
        
        if all(artifact_checks) and all(compatibility_checks):
            print("   ✅ Cohérence du nommage des artefacts: PASS")
            return True
        else:
            print("   ❌ Cohérence du nommage des artefacts: FAIL")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur lors du test du nommage des artefacts: {e}")
        return False

def test_data_source_consistency():
    """Teste que les deux algorithmes utilisent les mêmes sources de données."""
    print("📥 Test de cohérence des sources de données...")
    
    try:
        # Lire les deux fichiers de step pour vérifier les sources de données
        sticky_step_path = "Ares/src/training/steps/market_analysis/sticky_finite_hmm_clustering/sticky_finite_hmm_regime_discovery_step.py"
        hdp_step_path = "Ares/src/training/steps/market_analysis/hdp_hmm_clustering/hdp_hmm_regime_discovery_step.py"
        
        # Vérifier sticky_finite_hmm
        with open(sticky_step_path, 'r') as f:
            sticky_content = f.read()
        
        # Vérifier HDP-HMM  
        with open(hdp_step_path, 'r') as f:
            hdp_content = f.read()
        
        # Sources de données attendues
        expected_sources = [
            "klines_downloading_processing",
            "data_collection", 
            "data_reading"
        ]
        
        # Vérifier dans sticky_finite_hmm
        sticky_sources = []
        for source in expected_sources:
            if source in sticky_content:
                sticky_sources.append(True)
                print(f"   ✅ Sticky_Finite_HMM utilise source: {source}")
            else:
                sticky_sources.append(False)
                print(f"   ❌ Sticky_Finite_HMM ne trouve pas source: {source}")
        
        # Vérifier dans HDP-HMM
        hdp_sources = []
        for source in expected_sources:
            if source in hdp_content:
                hdp_sources.append(True)
                print(f"   ✅ HDP-HMM utilise source: {source}")
            else:
                hdp_sources.append(False)
                print(f"   ❌ HDP-HMM ne trouve pas source: {source}")
        
        if all(sticky_sources) and all(hdp_sources):
            print("   ✅ Cohérence des sources de données: PASS")
            return True
        else:
            print("   ❌ Cohérence des sources de données: FAIL")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur lors du test des sources de données: {e}")
        return False

def run_comprehensive_test():
    """Lance tous les tests de cohérence."""
    print("🚀 Démarrage du test de cohérence sticky_finite_hmm vs HDP-HMM")
    print("=" * 70)
    
    tests = [
        ("Validation des données", test_data_validation_consistency),
        ("Prétraitement", test_preprocessing_consistency),
        ("Nommage des artefacts", test_artifact_naming_consistency),
        ("Sources de données", test_data_source_consistency),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 50)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Échec du test {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés! Les deux implémentations sont cohérentes.")
        return True
    else:
        print(f"\n⚠️ {total-passed} test(s) ont échoué. Des corrections supplémentaires sont nécessaires.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)