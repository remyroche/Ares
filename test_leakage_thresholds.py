#!/usr/bin/env python3
"""
Script de test pour valider les nouveaux seuils de détection de fuite de données
dans le pipeline regime_ensemble_training.

Ce script vérifie que les seuils critiques (5%) et d'avertissement (2%) sont
correctement appliqués et que le multiplicateur pour petits datasets est désactivé.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import du module de détection de fuite
try:
    from src.utils.ml_common.validation.data_leakage_prevention import (
        DataLeakagePrevention, 
        DataLeakageConfig
    )
    logger.info("✅ Import réussi du module data_leakage_prevention")
except ImportError as e:
    logger.error(f"❌ Erreur d'import: {e}")
    sys.exit(1)

def test_threshold_values():
    """Test que les nouveaux seuils sont correctement définis."""
    logger.info("\n🔍 Test 1: Vérification des valeurs de seuils")
    
    config = DataLeakageConfig()
    
    # Vérification des seuils critiques
    assert config.critical_leakage_threshold == 0.05, f"Seuil critique incorrect: {config.critical_leakage_threshold} (attendu: 0.05)"
    logger.info(f"✅ Seuil critique correct: {config.critical_leakage_threshold:.1%}")
    
    # Vérification des seuils d'avertissement
    assert config.warning_leakage_threshold == 0.02, f"Seuil d'avertissement incorrect: {config.warning_leakage_threshold} (attendu: 0.02)"
    logger.info(f"✅ Seuil d'avertissement correct: {config.warning_leakage_threshold:.1%}")
    
    # Vérification du multiplicateur pour petits datasets
    assert config.small_dataset_leakage_multiplier == 1.0, f"Multiplicateur incorrect: {config.small_dataset_leakage_multiplier} (attendu: 1.0)"
    logger.info(f"✅ Multiplicateur pour petits datasets correct: {config.small_dataset_leakage_multiplier}")
    
    logger.info("✅ Test 1 réussi: Tous les seuils sont correctement définis")

def test_leakage_detection():
    """Test la détection de fuite avec différents niveaux."""
    logger.info("\n🔍 Test 2: Détection de fuite avec différents niveaux")
    
    prevention = DataLeakagePrevention()
    
    # Création de données de test avec timestamps
    n_samples = 1000
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Test 1: Pas de fuite (0%)
    data_clean = pd.DataFrame({
        'timestamp': timestamps,
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    report_clean = prevention.detect_temporal_leakage(data_clean, 'timestamp', 'target', 'test_clean')
    assert report_clean.overall_leakage_rate == 0.0, f"Taux de fuite incorrect pour données propres: {report_clean.overall_leakage_rate}"
    logger.info(f"✅ Données propres: taux de fuite = {report_clean.overall_leakage_rate:.2%}")
    
    # Test 2: Fuite faible (1.5%) - devrait être en dessous du seuil d'avertissement
    n_leak_samples = int(0.015 * n_samples)  # 1.5%
    data_low_leak = data_clean.copy()
    # Introduire une fuite temporelle en mélangeant quelques échantillons
    leak_indices = np.random.choice(n_samples, n_leak_samples, replace=False)
    data_low_leak.loc[leak_indices, 'timestamp'] = data_low_leak.loc[leak_indices, 'timestamp'] + timedelta(hours=24)
    
    report_low_leak = prevention.detect_temporal_leakage(data_low_leak, 'timestamp', 'target', 'test_low_leak')
    logger.info(f"✅ Fuite faible (1.5%): taux = {report_low_leak.overall_leakage_rate:.2%}, sévérité = {report_low_leak.severity_level}")
    assert report_low_leak.severity_level in ["none", "low", "medium"], f"Sévérité incorrecte pour fuite faible: {report_low_leak.severity_level}"
    
    # Test 3: Fuite modérée (3%) - devrait déclencher un avertissement
    n_leak_samples = int(0.03 * n_samples)  # 3%
    data_medium_leak = data_clean.copy()
    leak_indices = np.random.choice(n_samples, n_leak_samples, replace=False)
    data_medium_leak.loc[leak_indices, 'timestamp'] = data_medium_leak.loc[leak_indices, 'timestamp'] + timedelta(hours=24)
    
    report_medium_leak = prevention.detect_temporal_leakage(data_medium_leak, 'timestamp', 'target', 'test_medium_leak')
    logger.info(f"✅ Fuite modérée (3%): taux = {report_medium_leak.overall_leakage_rate:.2%}, sévérité = {report_medium_leak.severity_level}")
    assert report_medium_leak.severity_level == "high", f"Sévérité incorrecte pour fuite modérée: {report_medium_leak.severity_level}"
    
    # Test 4: Fuite élevée (7%) - devrait être critique
    n_leak_samples = int(0.07 * n_samples)  # 7%
    data_high_leak = data_clean.copy()
    leak_indices = np.random.choice(n_samples, n_leak_samples, replace=False)
    data_high_leak.loc[leak_indices, 'timestamp'] = data_high_leak.loc[leak_indices, 'timestamp'] + timedelta(hours=24)
    
    report_high_leak = prevention.detect_temporal_leakage(data_high_leak, 'timestamp', 'target', 'test_high_leak')
    logger.info(f"✅ Fuite élevée (7%): taux = {report_high_leak.overall_leakage_rate:.2%}, sévérité = {report_high_leak.severity_level}")
    assert report_high_leak.severity_level == "critical", f"Sévérité incorrecte pour fuite élevée: {report_high_leak.severity_level}"
    
    logger.info("✅ Test 2 réussi: Détection de fuite fonctionne correctement avec les nouveaux seuils")

def test_small_dataset_handling():
    """Test que les petits datasets n'ont plus de multiplicateur permissif."""
    logger.info("\n🔍 Test 3: Gestion des petits datasets")
    
    prevention = DataLeakagePrevention()
    
    # Création d'un petit dataset (500 échantillons < 1000)
    n_samples = 500
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Test avec fuite de 3% (serait en dessous de 4.5% avec l'ancien multiplicateur)
    n_leak_samples = int(0.03 * n_samples)  # 3%
    data_small = pd.DataFrame({
        'timestamp': timestamps,
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Introduire la fuite
    leak_indices = np.random.choice(n_samples, n_leak_samples, replace=False)
    data_small.loc[leak_indices, 'timestamp'] = data_small.loc[leak_indices, 'timestamp'] + timedelta(hours=24)
    
    report_small = prevention.detect_temporal_leakage(data_small, 'timestamp', 'target', 'test_small')
    logger.info(f"✅ Petit dataset (500 samples, 3% fuite): taux = {report_small.overall_leakage_rate:.2%}, sévérité = {report_small.severity_level}")
    
    # Avec le nouveau seuil, 3% devrait déclencher un avertissement (pas de multiplicateur)
    assert report_small.severity_level == "high", f"Sévérité incorrecte pour petit dataset avec 3% de fuite: {report_small.severity_level}"
    
    logger.info("✅ Test 3 réussi: Les petits datasets sont traités avec les mêmes seuils stricts")

def main():
    """Fonction principale de test."""
    logger.info("🚀 Démarrage des tests de validation des seuils de détection de fuite")
    logger.info("=" * 70)
    
    try:
        test_threshold_values()
        test_leakage_detection()
        test_small_dataset_handling()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 TOUS LES TESTS RÉUSSIS!")
        logger.info("✅ Les nouveaux seuils de détection de fuite sont correctement appliqués:")
        logger.info("   • Seuil critique: 5% (au lieu de 25%)")
        logger.info("   • Seuil d'avertissement: 2% (au lieu de 10%)")
        logger.info("   • Multiplicateur pour petits datasets: 1.0 (au lieu de 1.5)")
        logger.info("   • Logs explicatifs ajoutés pour les données financières")
        
    except AssertionError as e:
        logger.error(f"❌ Test échoué: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()