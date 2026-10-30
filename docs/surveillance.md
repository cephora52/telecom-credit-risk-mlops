# Surveillance du modèle : indicateurs et actions

## Deux risques surveillés
1. **Dérive des données (data drift)** : les clients de demain ne ressemblent plus à ceux de l'entraînement
   (par exemple, inflation qui change les dépenses). Mesurable **sans** connaître la vraie classe.
2. **Baisse de performance** (qui peut venir d'une dérive de concept) : le modèle se trompe plus qu'avant.
   Mesurable seulement quand la **vraie classe** est connue, donc avec du retard.

## Indicateurs et actions

| Indicateur | Mesure | Avertissement | Critique | Action |
|---|---|---|---|---|
| Dérive de chaque variable | PSI (Population Stability Index) | ≥ 0,10 | ≥ 0,20 | Voir « Actions » ci-dessous |
| Dérive des variables numériques | Test de Kolmogorov-Smirnov | p < 0,01 (indicatif) | idem | Confirme le PSI ; très sensible sur de grands échantillons, ne sert pas seul |
| Performance globale | F1 macro actuel comparé au F1 macro de la version validée | baisse ≥ 0,03 | baisse ≥ 0,06 | Voir « Actions » ci-dessous |
| Rappel par classe | Rappel de `high`, `low`, `medium` | suivi (voir l'issue #2) | — | Rappelle la limite connue sur `medium` |

Les seuils sont dans `mlflow_settings.py`. Le **statut global** est le plus grave des niveaux observés.

## Actions
- **Statut OK** : rien à faire, continuer au rythme prévu.
- **Avertissement** : relancer la surveillance sous 7 jours, repérer les variables concernées, vérifier la qualité des données entrantes (colonne mal remplie, changement de format).
- **Critique** :
  1. Analyser d'abord la cause : erreur de collecte, ou vrai changement de comportement ?
  2. Si le changement est réel : réentraîner (`train_mlflow.py`) sur des données récentes, comparer au modèle validé, puis promouvoir **seulement** si les critères du registre sont respectés.
  3. Si la performance a fortement baissé : revenir à la version précédente (`registry.py rollback`) en attendant.
  4. Ne **jamais** remplacer le modèle automatiquement dès qu'une dérive apparaît : une dérive n'entraîne pas toujours une baisse de performance, et une fausse alerte (donnée corrompue) donnerait un modèle pire.
  5. Garder une trace de la décision (qui, quand, pourquoi) dans la description de la Pull Request ou du ticket.

## Exécution

```bash
# Sur un fichier de données récentes (mêmes colonnes que le modèle ; la colonne default_risk est facultative)
python src/telecom_mlops/monitoring.py --current donnees_recentes.csv

# Démonstration avec une dérive simulée (nécessite le fichier de données du projet)
python src/telecom_mlops/monitoring.py --simulate-drift
python src/telecom_mlops/monitoring.py --simulate-drift --simulate-concept-drift
```

Sorties dans `reports/` : `monitoring_report.md`, `monitoring_report.json` et `monitoring_psi.png`.
Code de retour : **0** = OK, **1** = avertissement, **2** = critique. Le pipeline CI/CD/CT (issue #5)
pourra s'en servir pour décider de lancer un réentraînement.

## Limites
- La référence est un échantillon de 5 000 lignes de l'entraînement (`models/reference_sample.csv`, créé par `train_mlflow.py`).
- La performance n'est mesurée que si la vraie classe est connue ; en production, elle arrive avec retard.
- L'API ne journalise pas encore les requêtes : sans ce journal, il n'y a pas de « données récentes » réelles à surveiller.
- La latence et les erreurs de l'API relèvent du déploiement (issue #5) et ne sont pas mesurées ici.
- Les seuils sont des points de départ, à ajuster avec l'expérience.
