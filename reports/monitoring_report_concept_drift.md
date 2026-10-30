# Rapport de surveillance du modèle

**Statut global : CRITICAL**

## Dérive des données

| Variable | PSI | p-value KS | Niveau |
|---|---|---|---|
| operator | 0.0009 | — | ok |
| credit_score | 2.5807 | 2.2e-163 | critical |
| credit_rating | 0.0002 | — | ok |
| tenure_months | 0.002 | 0.383 | ok |
| avg_monthly_spend_ngn | 0.6645 | 0.0 | critical |
| recharge_frequency_per_month | 3.6452 | 0.0 | critical |
| payment_consistency_score | 0.0029 | 0.634 | ok |
| late_payment_count | 0.0006 | 0.962 | ok |
| account_type | 0.0019 | — | ok |

## Performance

- Version évaluée : 1 (alias validated)
- F1 macro de référence : 0.4556
- F1 macro actuel : 0.4259 (baisse : 0.0297)
- Rappel par classe : {'high': 0.5613, 'low': 0.7418, 'medium': 0.0723}
- Niveau : ok

## Action recommandée

Analyser d'abord la cause (erreur de collecte ou vrai changement de comportement). Si le changement est réel : réentraîner avec train_mlflow.py sur des données récentes, comparer au modèle validé, puis promouvoir seulement si les critères sont respectés. En cas de baisse de performance sévère, revenir à la version précédente (registry.py rollback). Ne jamais remplacer le modèle automatiquement.
