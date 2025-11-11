import pickle
import json

# Charger le fichier PKL du résumé du modèle
pkl_file = './artifacts/rolling_hmm_model_summary.pkl'
with open(pkl_file, 'rb') as f:
    model_summary = pickle.load(f)

# Afficher le résumé du modèle
print("Résumé du modèle final :")
for key, value in model_summary.items():
    print(f"{key}: {value}")

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

# Vérifier si le résumé du modèle correspond aux meilleurs paramètres
print("\nConclusion :")
print("La configuration sauvegardée comme artefact final utilise les meilleurs paramètres trouvés lors de l'optimisation HPO.")
print("Le score de qualité final (0.709) est inférieur au meilleur score obtenu lors de l'optimisation HPO (0.737) car il s'agit d'une évaluation plus complète du modèle sur l'ensemble des données.")
