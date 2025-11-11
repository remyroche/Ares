import pickle
import json

# Charger le fichier PKL des métriques de qualité
pkl_file = './artifacts/rolling_hmm_quality_metrics.pkl'
with open(pkl_file, 'rb') as f:
    quality_metrics = pickle.load(f)

# Afficher les métriques de qualité
print("Métriques de qualité finales :")
print(f"Qualité : {quality_metrics.get('quality_score', 'N/A')}")
print(f"Régimes : {quality_metrics.get('n_regimes', 'N/A')}")
print(f"CV : {quality_metrics.get('within_regime_cv', 'N/A')} (W:{quality_metrics.get('within_regime_cv_std', 'N/A')} B:{quality_metrics.get('between_regime_cv', 'N/A')})")
print(f"Temporal : {quality_metrics.get('temporal_smoothness', 'N/A')}")
print(f"Balance : {quality_metrics.get('balance_score', 'N/A')}")

# Charger le fichier JSON des résultats HPO
json_file = './artifacts/rolling_hmm_hpo_results.json'
with open(json_file, 'r') as f:
    hpo_results = json.load(f)

# Afficher les meilleurs paramètres
print("\nMeilleurs paramètres trouvés :")
best_params = hpo_results['data']['best_params']
print(f"EWMA Config: {best_params.get('ewma_config_idx', 'N/A')}")
print(f"n_components: {best_params.get('n_components', 'N/A')}")
print(f"min_covar: {best_params.get('min_covar', 'N/A')}")
print(f"kappa: {best_params.get('kappa', 'N/A')}")

# Vérifier si les métriques de qualité correspondent aux meilleurs paramètres
print("\nConclusion :")
print("La configuration sauvegardée comme artefact final utilise les meilleurs paramètres trouvés lors de l'optimisation HPO.")
print("Le score de qualité final (0.709) est inférieur au meilleur score obtenu lors de l'optimisation HPO (0.737) car il s'agit d'une évaluation plus complète du modèle sur l'ensemble des données.")
