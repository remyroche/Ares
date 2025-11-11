# Tests pour le module Rolling HMM Clustering

Ce document décrit la suite de tests complète pour le module `rolling_hmm_clustering`, qui couvre tous les composants avec des tests unitaires, d'intégration et de bout en bout.

## Structure des tests

La suite de tests est organisée en 5 classes principales :

### 1. TestFeatureEngineering
Tests unitaires pour la classe `RollingHMMFeatureEngineer` :
- Initialisation avec différentes configurations
- Validation des configurations EWMA
- Génération de caractéristiques (returns, volatilité, tendance, volume)
- Mise en cache et récupération des caractéristiques
- Application de l'ACP et mise en cache

### 2. TestStickyHMMModel
Tests unitaires pour la classe `StickyHMMModel` :
- Initialisation avec différentes configurations
- Ajustement du modèle avec données synthétiques
- Prédiction et prédiction de probabilités
- Calcul de log-vraisemblance
- Récupération de la matrice de transition et de la distribution stationnaire
- Calcul des durées attendues des états

### 3. TestHPOConfig
Tests unitaires pour la classe `RollingHMMOptimizer` :
- Initialisation avec différentes configurations
- Création et validation des groupes de paramètres
- Création et exécution de la fonction objectif
- Processus d'optimisation avec dépendances mockées

### 4. TestRollingHMMRegimeDiscoveryStep
Tests d'intégration pour la classe `RollingHMMRegimeDiscoveryStep` :
- Initialisation et validation de la configuration
- Extraction des configurations de caractéristiques et HPO
- Initialisation de l'optimisation matérielle
- Filtrage des données selon le mode d'exécution
- Exécution du clustering avec dépendances mockées

### 5. TestEndToEnd
Test de bout en bout pour l'ensemble du pipeline :
- Feature Engineering → PCA → HMM → Prédiction → Qualité
- Validation de l'interaction entre tous les composants

## Exécution des tests

### Via le script d'exécution

Le script `run_rolling_hmm_tests.py` offre plusieurs options d'exécution :

```bash
# Exécuter tous les tests
python run_rolling_hmm_tests.py

# Exécuter uniquement les tests de feature engineering
python run_rolling_hmm_tests.py --module feature

# Exécuter uniquement les tests du modèle HMM
python run_rolling_hmm_tests.py --module hmm

# Exécuter uniquement les tests de configuration HPO
python run_rolling_hmm_tests.py --module hpo

# Exécuter uniquement les tests de découverte de régimes
python run_rolling_hmm_tests.py --module discovery

# Exécuter uniquement les tests de bout en bout
python run_rolling_hmm_tests.py --module e2e

# Exécuter avec rapport de couverture de code
python run_rolling_hmm_tests.py --coverage

# Vérifier uniquement la syntaxe des fichiers
python run_rolling_hmm_tests.py --syntax

# Exécuter avec verbosité réduite
python run_rolling_hmm_tests.py --verbosity 1
```

### Via unittest

```bash
# Exécuter tous les tests
python -m unittest test_rolling_hmm_clustering.py -v

# Exécuter une classe de test spécifique
python -m unittest test_rolling_hmm_clustering.TestFeatureEngineering -v
```

## Stratégie de mocking

Pour isoler les tests et éviter les dépendances externes, les approches de mocking suivantes sont utilisées :

1. **ClusterQualityAssessor** : Mock pour éviter les dépendances complexes d'évaluation de qualité
2. **ArtifactManager** : Mock pour éviter les dépendances du système de fichiers
3. **HardwareManager** : Désactivé dans les configurations de test
4. **VectorBT Optimizers** : Désactivés dans les configurations de test
5. **Numba JIT** : Désactivé dans les configurations de test

## Données de test

Les tests utilisent des données synthétiques générées avec une graine fixe (np.random.seed(42)) pour assurer la reproductibilité :

```python
# Données de marché synthétiques
n_samples = 100-200
market_data = pd.DataFrame({
    'open': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
    'high': 100 + np.cumsum(np.random.normal(0.1, 1, n_samples)),
    'low': 100 + np.cumsum(np.random.normal(-0.1, 1, n_samples)),
    'close': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
    'volume': np.random.uniform(1000, 5000, n_samples)
})
```

## Couverture de code cible

La suite de tests vise une couverture de code d'au moins 80% pour chaque composant :

- **feature_engineering.py** : 85%+ de couverture
- **sticky_hmm_model.py** : 90%+ de couverture
- **hpo_config.py** : 80%+ de couverture
- **rolling_hmm_regime_discovery_step.py** : 75%+ de couverture

## Dépendances de test

Les tests nécessitent les dépendances suivantes :

```bash
# Dépendances de base
numpy
pandas
unittest
mock

# Pour la couverture de code (optionnel)
coverage

# Dépendances du module (mockées dans les tests)
hmmlearn
sklearn
numba
vectorbt
```

## Résolution des problèmes courants

### Problèmes d'import

Si vous rencontrez des erreurs d'import, assurez-vous que le répertoire racine du projet est dans le PYTHONPATH :

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
```

### Problèmes de dépendances manquantes

Certaines dépendances sont mockées dans les tests mais doivent être installées pour le développement :

```bash
pip install hmmlearn sklearn numba vectorbt
```

### Tests asynchrones

Les tests asynchrones utilisent `unittest.TestCase` avec des méthodes `async def`. Pour exécuter ces tests, assurez-vous d'utiliser Python 3.7+.

## Rapports de test

### Rapport de couverture

Pour générer un rapport de couverture HTML détaillé :

```bash
python run_rolling_hmm_tests.py --coverage
```

Le rapport sera généré dans `test_output/rolling_hmm_coverage/index.html`.

### Rapport de synthèse

Après exécution, un rapport de synthèse est affiché avec :
- Nombre de tests exécutés
- Nombre de tests réussis/échoués
- Temps d'exécution total
- Taux de couverture par module

## Maintenance des tests

### Ajout de nouveaux tests

Pour ajouter de nouveaux tests :

1. Identifier la classe de test appropriée
2. Ajouter une méthode `test_*` à la classe
3. Utiliser des données synthétiques avec graine fixe
4. Mock des dépendances externes si nécessaire
5. Documenter le cas de test dans le docstring

### Mise à jour des mocks

Lors de l'évolution du code, vérifiez que les mocks sont toujours valides :

- Interfaces des classes mockées
- Signatures des méthodes mockées
- Valeurs de retour des méthodes mockées

### Tests de régression

Pour ajouter des tests de régression :

1. Identifier un bug ou une régression récente
2. Créer un test qui reproduit le problème
3. Vérifier que le test échoue avec le code actuel
4. Corriger le code et vérifier que le test réussit

## Intégration CI/CD

Cette suite de tests est conçue pour s'intégrer dans un pipeline CI/CD :

```yaml
# Exemple de configuration GitHub Actions
- name: Run Rolling HMM Tests
  run: |
    cd src/training/steps/market_analysis/rolling_hmm_clustering
    python run_rolling_hmm_tests.py --syntax
    python run_rolling_hmm_tests.py --module all --verbosity 1
    python run_rolling_hmm_tests.py --coverage
```

## Conclusion

Cette suite de tests fournit une couverture complète du module `rolling_hmm_clustering` avec :
- Tests unitaires isolés pour chaque composant
- Tests d'intégration pour valider les interactions
- Test de bout en bout pour valider le pipeline complet
- Mock des dépendances externes pour une exécution fiable
- Documentation complète pour la maintenance

Elle constitue une base solide pour le développement et la maintenance continue du module.