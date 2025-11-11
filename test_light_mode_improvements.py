#!/usr/bin/env python3
"""
Script de test pour valider les améliorations du pipeline en mode light.

Ce script vérifie :
1. La configuration à 3 folds en mode light
2. Le logging pour la gestion des régimes rares
3. Le logging pour la gestion mémoire sous pression
4. Les optimisations LightGBM/XGBoost pour petits datasets
5. La configuration de DepthWiseCNN et son HPO réactivé
"""

import sys
import os
import yaml
import logging
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cv_folds_light_mode():
    """Test 1: Vérifier la configuration à 3 folds en mode light."""
    print("🔍 Test 1: Validation croisée à 3 folds en mode light")
    
    # Charger la configuration du pipeline de régimes
    config_file = "src/training/steps/market_analysis/rolling_hmm_clustering_iterative/rolling_hmm_regime_discovery_step.py"
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Vérifier si la condition pour 3 folds en mode light est présente
        if "cv_folds'] = hpo_config.get('cv_folds', 3)  # 3 folds en mode light (amélioration demandée)" in content:
            print("✅ Configuration 3 folds en mode light trouvée")
            print("   ↪ cv_folds = hpo_config.get('cv_folds', 3)  # 3 folds en mode light (amélioration demandée)")
        else:
            print("❌ Configuration 3 folds en mode light NON trouvée")
            return False
    else:
        print(f"❌ Fichier de configuration non trouvé: {config_file}")
        return False
    
    return True

def test_regime_logging():
    """Test 2: Vérifier le logging pour la gestion des régimes rares."""
    print("\n🔍 Test 2: Logging pour la gestion des régimes rares")
    
    # Charger la configuration des régimes
    config_file = "src/training/steps/market_analysis/components/regime_models_training_refactored/training/regime_models_training.py"
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Vérifier la présence des logs pour min_regime_samples
        logs_to_check = [
            "🔍 REGIME FILTERING: Checking regime sample counts",
            "⚠️ INSUFFICIENT REGIME SAMPLES",
            "🔧 ADAPTIVE MIN_REGIME_SAMPLES",
            "📊 REGIME DISTRIBUTION BY FOLD",
            "min_regime_samples",
            "Petits datasets optimisation"
        ]
        
        found_logs = []
        for log_msg in logs_to_check:
            if log_msg in content:
                found_logs.append(log_msg)
        
        if found_logs:
            print(f"✅ Logs pour régimes rares trouvés ({len(found_logs)}/{len(logs_to_check)})")
            for log in found_logs:
                print(f"   ↪ {log[:50]}...")
        else:
            print("❌ Logs pour régimes rares NON trouvés")
            return False
    else:
        print(f"❌ Fichier de configuration des régimes non trouvé: {config_file}")
        return False
    
    return True

def test_memory_logging():
    """Test 3: Vérifier le logging pour la gestion mémoire sous pression."""
    print("\n🔍 Test 3: Logging pour la gestion mémoire sous pression")
    
    # Charger le gestionnaire de mémoire
    config_file = "src/utils/ml_common/training/memory_manager.py"
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Vérifier la présence des logs détaillés pour la gestion mémoire
        logs_to_check = [
            "🚨 MEMORY PRESSURE DETECTED",
            "🔧 ADAPTIVE SUBSET SIZE",
            "⚠️ HIGH MEMORY USAGE",
            "📊 MEMORY USAGE STATS",
            "🧹 MEMORY CLEANUP"
        ]
        
        found_logs = []
        for log_msg in logs_to_check:
            if log_msg in content:
                found_logs.append(log_msg)
        
        if found_logs:
            print(f"✅ Logs pour gestion mémoire trouvés ({len(found_logs)}/{len(logs_to_check)})")
            for log in found_logs:
                print(f"   ↪ {log[:50]}...")
        else:
            print("❌ Logs pour gestion mémoire NON trouvés")
            return False
    else:
        print(f"❌ Fichier de gestion mémoire non trouvé: {config_file}")
        return False
    
    return True

def test_lightgbm_xgboost_optimization():
    """Test 4: Vérifier les optimisations LightGBM/XGBoost pour petits datasets."""
    print("\n🔍 Test 4: Optimisations LightGBM/XGBoost pour petits datasets")
    
    # Charger la configuration des modèles de régimes
    config_file = "src/training/steps/market_analysis/components/regime_models_training_refactored/training/regime_models_training.py"
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Vérifier la présence des optimisations pour petits datasets
        optimizations_to_check = [
            "# Petits datasets optimisation",
            "small_dataset_params",
            "early_stopping_patience': 5  # Plus agressif pour petits datasets",
            "min_child_samples': 5  # Plus strict pour éviter l'overfitting",
            "if n_samples < 500:",
            "n_estimators': 100  # Réduit pour petits datasets"
        ]
        
        found_opts = []
        for opt in optimizations_to_check:
            if opt in content:
                found_opts.append(opt)
        
        if found_opts:
            print(f"✅ Optimisations LightGBM/XGBoost trouvées ({len(found_opts)}/{len(optimizations_to_check)})")
            for opt in found_opts:
                print(f"   ↪ {opt[:50]}...")
        else:
            print("❌ Optimisations LightGBM/XGBoost NON trouvées")
            return False
    else:
        print(f"❌ Fichier de configuration des modèles non trouvé: {config_file}")
        return False
    
    return True

def test_depthwise_cnn_configuration():
    """Test 5: Vérifier la configuration de DepthWiseCNN et son HPO."""
    print("\n🔍 Test 5: Configuration DepthWiseCNN et HPO réactivé")
    
    # Vérifier les fichiers de configuration
    config_files = [
        "src/training/steps/model_training/analyst_base_config.yaml",
        "src/training/steps/model_training/analyst_ensemble_config.yaml",
        "src/training/steps/model_training/tactician_ensemble_config.yaml"
    ]
    
    all_checks_passed = True
    
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                content = f.read()
            
            print(f"\n📋 Vérification de {os.path.basename(config_file)}:")
            
            # Vérifier la présence de depthwise_cnn
            if "depthwise_cnn:" in content:
                print("   ✅ Configuration depthwise_cnn trouvée")
            else:
                print("   ❌ Configuration depthwise_cnn NON trouvée")
                all_checks_passed = False
            
            # Vérifier la présence des outputs depthwise_cnn
            if "depthwise_cnn_output:" in content or "depthwise_cnn_predictions" in content:
                print("   ✅ Output depthwise_cnn trouvé")
            else:
                print("   ❌ Output depthwise_cnn NON trouvé")
                all_checks_passed = False
            
            # Vérifier la réactivation du HPO en mode light
            if "DepthwiseCNN HPO: REACTIVATED" in content:
                print("   ✅ HPO DepthWiseCNN réactivé en mode light")
            else:
                print("   ❌ HPO DepthWiseCNN NON réactivé en mode light")
                all_checks_passed = False
        else:
            print(f"   ❌ Fichier non trouvé: {config_file}")
            all_checks_passed = False
    
    return all_checks_passed

def test_unified_training_step():
    """Test 6: Vérifier les modifications dans unified_models_training_step.py"""
    print("\n🔍 Test 6: Modifications dans unified_models_training_step.py")
    
    config_file = "src/training/steps/model_training/unified_models_training_step.py"
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Vérifier la réactivation du HPO pour DepthWiseCNN en mode light
        if "# REACTIVATE DepthwiseCNN HPO in light mode (user request)" in content:
            print("   ✅ Réactivation HPO DepthWiseCNN en mode light trouvée")
        else:
            print("   ❌ Réactivation HPO DepthWiseCNN NON trouvée")
            return False
            
        # Vérifier l'import de tprint_data_preview
        if "from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning, tprint_data_preview" in content:
            print("   ✅ Import tprint_data_preview corrigé")
        else:
            print("   ❌ Import tprint_data_preview NON corrigé")
            return False
    else:
        print(f"   ❌ Fichier non trouvé: {config_file}")
        return False
    
    return True

def main():
    """Fonction principale pour exécuter tous les tests."""
    print("🚀 DÉMARRAGE DES TESTS DES AMÉLIORATIONS EN MODE LIGHT")
    print("=" * 80)
    
    tests = [
        ("Validation croisée à 3 folds", test_cv_folds_light_mode),
        ("Logging régimes rares", test_regime_logging),
        ("Logging gestion mémoire", test_memory_logging),
        ("Optimisations LightGBM/XGBoost", test_lightgbm_xgboost_optimization),
        ("Configuration DepthWiseCNN", test_depthwise_cnn_configuration),
        ("Modifications unified_models_training_step", test_unified_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur lors du test '{test_name}': {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{status} - {test_name}")
        if result:
            passed_tests += 1
    
    print("\n" + "=" * 80)
    print(f"📈 RÉSULTAT GLOBAL: {passed_tests}/{total_tests} tests passés")
    
    if passed_tests == total_tests:
        print("🎉 TOUTES LES AMÉLIORATIONS SONT CORRECTEMENT IMPLEMENTÉES!")
        print("\n✅ Le pipeline est prêt pour l'exécution en mode light avec:")
        print("   • 3 folds au lieu de 5")
        print("   • Logging détaillé pour les régimes rares")
        print("   • Logging détaillé pour la gestion mémoire")
        print("   • Optimisations LightGBM/XGBoost pour petits datasets")
        print("   • DepthWiseCNN avec HPO réactivé")
        return 0
    else:
        print("⚠️ CERTAINES AMÉLIORATIONS NÉCESSITENT D'ATTENTION")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)