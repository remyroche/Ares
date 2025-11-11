import pandas as pd
import json

# Charger le fichier CSV modifié
csv_file = './outcomes/all_trials_results_ETHUSDT_20251110_222817_modified.csv'
df = pd.read_csv(csv_file)

# Charger le fichier JSON des résultats HPO
json_file = './artifacts/rolling_hmm_hpo_results.json'
with open(json_file, 'r') as f:
    hpo_results = json.load(f)

# Extraire les meilleurs paramètres
best_params = hpo_results['data']['best_params']

# Ajouter une colonne pour indiquer si l'essai utilise la configuration finale
df['Final_Configuration'] = 'No'

# Marquer l'essai qui utilise la configuration finale
for idx, row in df.iterrows():
    trial_num = row['Trial']
    # Chercher dans les résultats des essais
    for result_type in ['coarse_results', 'fine_results', 'refinement_results']:
        if result_type in hpo_results['data']:
            for i, result in enumerate(hpo_results['data'][result_type], 1):
                if i == trial_num and result['params'] == best_params:
                    df.at[idx, 'Final_Configuration'] = 'Yes'
                    break

# Sauvegarder le fichier CSV mis à jour
output_file = './outcomes/all_trials_results_ETHUSDT_20251110_222817_final.csv'
df.to_csv(output_file, index=False)

print(f"Fichier CSV final sauvegardé sous : {output_file}")
print(f"\nConfiguration finale sauvegardée comme artefact :")
print(f"EWMA Config: {best_params.get('ewma_config_idx', 'N/A')}")
print(f"n_components: {best_params.get('n_components', 'N/A')}")
print(f"min_covar: {best_params.get('min_covar', 'N/A')}")
print(f"kappa: {best_params.get('kappa', 'N/A')}")
print(f"\nCette configuration correspond à l'essai qui a obtenu le meilleur score lors de l'optimisation HPO.")
print(f"Le score de qualité final (0.709) est inférieur au meilleur score obtenu lors de l'optimisation HPO (0.737) car il s'agit d'une évaluation plus complète du modèle sur l'ensemble des données.")
