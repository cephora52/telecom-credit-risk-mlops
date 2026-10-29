# Limites du modèle XGBoost (issue #2)

- Séparation reproductible : test_size=0.2, random_state=42, stratifiée. Choix des réglages sur une validation extraite du train ; le test n'est utilisé qu'une fois.
- Résultat final sur le test : F1 macro = 0.483 (avant : 0.471), AUC macro = 0.650 (avant : 0.658).
- Classe medium : rappel 0.037 -> 0.111, F1 0.065 -> 0.162.
- Le gain sur medium se paie par une baisse de la précision ou du rappel des autres classes (voir le tableau avant/après) : c'est un compromis, pas une amélioration gratuite.
- L'AUC macro reste modeste : les variables disponibles séparent mal la classe medium des classes low et high. Une classe « du milieu » chevauche les deux autres.
- Aucun score de 90 % n'est garanti : il ne serait pas justifié par ces données.
- Variables à vérifier : credit_score et credit_rating (voir l'audit de fuite). Elles ne sont pas retirées automatiquement ; la décision dépend de la façon dont le jeu de données a été généré.
- Les poids de classes [high, low, medium] = [0.6, 1.0, 1.0] sont réglés sur la validation ; ils devront être revalidés si les données dérivent.
- Le modèle final est entraîné sur 80 % du train (les 20 % restants servent à la validation), alors que la référence utilise tout le train : la comparaison reste à mettre en regard de cette différence.
- Données synthétiques : les résultats ne se transposent pas tels quels à des données réelles.
