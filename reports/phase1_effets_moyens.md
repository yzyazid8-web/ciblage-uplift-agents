# Phase 1 : équilibre des groupes et effets moyens

Généré le 2026-11-07 par `python -m ciblage.causal.report`.

## Hillstrom (64 000 clients, essai randomisé 1/3 - 1/3 - 1/3)

Effectifs : temoin = 21 306, email_homme = 21 307, email_femme = 21 387

### Équilibre des groupes avant la campagne

- Effectifs (test du chi² vs protocole) : p = 0.90
- Tests par variable : indicatifs ; le critère retenu est la SMD
- |SMD| maximale : 0.014 (seuil 0.1) → **✅ équilibré**

| variable | SMD email_homme | SMD email_femme | test | p-valeur |
|---|---|---|---|---|
| recency | +0.007 | +0.005 | kruskal | 0.79 |
| history | +0.008 | +0.007 | kruskal | 0.67 |
| mens | -0.005 | -0.009 | chi2 | 0.67 |
| womens | +0.008 | +0.005 | chi2 | 0.73 |
| newbie | -0.001 | +0.003 | chi2 | 0.93 |
| zip_code_Surburban | -0.012 | -0.001 | chi2 | 0.41 |
| zip_code_Rural | +0.014 | +0.004 | chi2 | 0.35 |
| channel_Web | +0.011 | -0.005 | chi2 | 0.23 |
| channel_Multichannel | -0.004 | -0.005 | chi2 | 0.85 |

### Effet moyen de chaque e-mail

| action | résultat | témoin | traité | effet moyen | effet relatif | IC 95 % (Welch) | IC 95 % (bootstrap) | p ajustée (Holm) |
|---|---|---|---|---|---|---|---|---|
| email_homme | visit | 0.1062 | 0.1828 | +7.66 pts | +72 % | [+7.00 pts ; +8.32 pts] | [+6.99 pts ; +8.34 pts] | 8.2e-112 |
| email_homme | conversion | 0.0057 | 0.0125 | +0.68 pts | +119 % | [+0.50 pts ; +0.86 pts] | [+0.50 pts ; +0.87 pts] | 6.0e-13 |
| email_homme | spend | 0.6528 | 1.4226 | +0.770 $ | +118 % | [+0.485 $ ; +1.055 $] | [+0.484 $ ; +1.060 $] | 3.5e-07 |
| email_femme | visit | 0.1062 | 0.1514 | +4.52 pts | +43 % | [+3.89 pts ; +5.16 pts] | [+3.90 pts ; +5.15 pts] | 1.2e-43 |
| email_femme | conversion | 0.0057 | 0.0088 | +0.31 pts | +54 % | [+0.15 pts ; +0.47 pts] | [+0.15 pts ; +0.48 pts] | 3.1e-04 |
| email_femme | spend | 0.6528 | 1.0772 | +0.424 $ | +65 % | [+0.169 $ ; +0.680 $] | [+0.167 $ ; +0.691 $] | 1.1e-03 |

## Criteo (échantillon aléatoire de 2 096 975 lignes, 85.0 % traités)

### Équilibre des groupes

- Effectifs : protocole d'affectation non publié, test non applicable
- Tests par variable (sur un sous-échantillon aléatoire de 200 000 lignes) : indicatifs ; le critère retenu est la SMD
- |SMD| maximale : 0.050 (seuil 0.1) → **✅ équilibré**

| variable | SMD publicité | test | p-valeur |
|---|---|---|---|
| f0 | -0.009 | kruskal | 0.45 |
| f1 | +0.028 | kruskal | 0.02 |
| f2 | -0.005 | kruskal | 0.21 |
| f3 | -0.050 | kruskal | 0.00 |
| f4 | +0.011 | kruskal | 0.22 |
| f5 | -0.031 | kruskal | 0.00 |
| f6 | -0.041 | kruskal | 0.00 |
| f7 | +0.020 | kruskal | 0.00 |
| f8 | -0.022 | kruskal | 0.70 |
| f9 | +0.024 | kruskal | 0.32 |
| f10 | +0.012 | kruskal | 0.22 |
| f11 | -0.009 | kruskal | 0.09 |

### Effet moyen de la publicité

| action | résultat | témoin | traité | effet moyen | effet relatif | IC 95 % (Welch) | IC 95 % (bootstrap) | p ajustée (Holm) |
|---|---|---|---|---|---|---|---|---|
| publicité | visit | 0.0382 | 0.0484 | +1.02 pts | +27 % | [+0.95 pts ; +1.10 pts] | [+0.95 pts ; +1.10 pts] | 2.9e-161 |
| publicité | conversion | 0.0019 | 0.0030 | +0.12 pts | +64 % | [+0.10 pts ; +0.14 pts] | [+0.10 pts ; +0.13 pts] | 4.9e-42 |
