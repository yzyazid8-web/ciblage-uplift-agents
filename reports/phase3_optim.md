# Phase 3 : optimisation de l'allocation sous budget

Généré le 2026-11-09 par `python -m ciblage.optim.benchmark`. Uplift : S-learner réglé en phase 2 ({'max_depth': 2, 'n_estimators': 300, 'min_child_weight': 5}), entraîné sur les 44 800 clients d'entraînement. Coûts supposés : e-mail homme 0.12 €, e-mail femme 0.10 €.

## 1. Solveurs

Instances tirées de l'uplift prédit sur Hillstrom ; budget = coût de 30 % des clients en e-mail homme ; scénario « capacité » : au plus 15 % de clients en e-mail femme. « Écart à la meilleure borne » = perte maximale possible par rapport à l'optimum (la borne est prouvée : relaxation linéaire, dual lagrangien ou MILP). Toutes les solutions ont été vérifiées faisables.

| scénario | clients | méthode | objectif (visites en plus) | écart à la meilleure borne | temps (s) |
|---|---|---|---|---|---|
| budget seul | 1 000 | glouton | 32.3 | 0.128 % | 0.00 |
| budget seul | 1 000 | lagrangien (maison) | 32.3 | 0.128 % | 0.02 |
| budget seul | 1 000 | relaxation linéaire (HiGHS) | 32.3 | 0.128 % | 0.06 |
| budget seul | 1 000 | MILP exact (HiGHS) | 32.3 | 0.000 % | 2.11 |
| budget seul | 5 000 | glouton | 161.2 | 0.020 % | 0.00 |
| budget seul | 5 000 | lagrangien (maison) | 161.2 | 0.027 % | 0.03 |
| budget seul | 5 000 | relaxation linéaire (HiGHS) | 161.2 | 0.027 % | 0.26 |
| budget seul | 5 000 | MILP exact (HiGHS) | 161.3 | 0.000 % | 20.39 |
| budget seul | 20 000 | glouton | 642.6 | 0.004 % | 0.03 |
| budget seul | 20 000 | lagrangien (maison) | 642.6 | 0.005 % | 0.19 |
| budget seul | 20 000 | relaxation linéaire (HiGHS) | 642.6 | 0.005 % | 2.56 |
| budget seul | 64 000 | glouton | 2 057.1 | 0.004 % | 0.10 |
| budget seul | 64 000 | lagrangien (maison) | 2 057.1 | 0.001 % | 0.58 |
| budget seul | 64 000 | relaxation linéaire (HiGHS) | 2 057.1 | 0.001 % | 19.32 |
| budget seul | 1 000 000 | glouton | 32 200.8 | 0.003 % | 1.27 |
| budget seul | 1 000 000 | lagrangien (maison) | 32 201.7 | 0.000 % | 5.91 |
| budget + capacité | 1 000 | glouton | 32.3 | 0.128 % | 0.00 |
| budget + capacité | 1 000 | lagrangien (maison) | 32.3 | 0.128 % | 0.04 |
| budget + capacité | 1 000 | relaxation linéaire (HiGHS) | 32.3 | 0.128 % | 0.02 |
| budget + capacité | 1 000 | MILP exact (HiGHS) | 32.3 | 0.000 % | 1.32 |
| budget + capacité | 5 000 | glouton | 161.2 | 0.020 % | 0.00 |
| budget + capacité | 5 000 | lagrangien (maison) | 161.2 | 0.028 % | 0.11 |
| budget + capacité | 5 000 | relaxation linéaire (HiGHS) | 161.2 | 0.027 % | 0.12 |
| budget + capacité | 5 000 | MILP exact (HiGHS) | 161.3 | 0.000 % | 12.45 |
| budget + capacité | 20 000 | glouton | 642.6 | 0.004 % | 0.02 |
| budget + capacité | 20 000 | lagrangien (maison) | 642.6 | 0.007 % | 0.54 |
| budget + capacité | 20 000 | relaxation linéaire (HiGHS) | 642.6 | 0.005 % | 1.53 |
| budget + capacité | 64 000 | glouton | 2 057.1 | 0.004 % | 0.04 |
| budget + capacité | 64 000 | lagrangien (maison) | 2 057.1 | 0.003 % | 1.77 |
| budget + capacité | 64 000 | relaxation linéaire (HiGHS) | 2 057.1 | 0.001 % | 16.60 |
| budget + capacité | 1 000 000 | glouton | 32 200.8 | 0.040 % | 1.03 |
| budget + capacité | 1 000 000 | lagrangien (maison) | 32 201.6 | 0.038 % | 42.36 |

## 2. Valeur métier sur le TEST de Hillstrom (IPW, IC 95 % bootstrap apparié)

19 200 clients jamais vus par les modèles. Pour chaque budget, les trois politiques dépensent au plus la même somme. « classique » = les clients les plus susceptibles de visiter reçoivent l'e-mail homme (le meilleur en moyenne) ; « optimisé » = relaxation lagrangienne sur l'uplift prédit, qui choisit QUI contacter ET avec QUEL e-mail, en tenant compte de son coût.

| budget | politique | contactés | dépense | taux de visite | gain vs classique (pts) | visites en plus / 1 000 clients |
|---|---|---|---|---|---|---|
| 230 € | classique (e-mail homme) | 1 919 (homme 1 919, femme 0) | 230 € | 11.45 % | +0.00 [+0.00 ; +0.00] | +0.0 |
| 230 € | uplift, e-mail homme seul | 1 919 (homme 1 919, femme 0) | 230 € | 11.71 % | +0.26 [-0.14 ; +0.69] | +2.6 |
| 230 € | optimisé (client + e-mail) | 1 925 (homme 1 893, femme 32) | 230 € | 11.68 % | +0.24 [-0.16 ; +0.65] | +2.4 |
| 461 € | classique (e-mail homme) | 3 839 (homme 3 839, femme 0) | 461 € | 12.33 % | +0.00 [+0.00 ; +0.00] | +0.0 |
| 461 € | uplift, e-mail homme seul | 3 839 (homme 3 839, femme 0) | 461 € | 12.29 % | -0.04 [-0.59 ; +0.51] | -0.4 |
| 461 € | optimisé (client + e-mail) | 3 874 (homme 3 668, femme 206) | 461 € | 12.22 % | -0.11 [-0.64 ; +0.44] | -1.1 |
| 691 € | classique (e-mail homme) | 5 759 (homme 5 759, femme 0) | 691 € | 12.99 % | +0.00 [+0.00 ; +0.00] | +0.0 |
| 691 € | uplift, e-mail homme seul | 5 759 (homme 5 759, femme 0) | 691 € | 12.84 % | -0.15 [-0.78 ; +0.48] | -1.5 |
| 691 € | optimisé (client + e-mail) | 5 873 (homme 5 193, femme 680) | 691 € | 12.76 % | -0.23 [-0.88 ; +0.38] | -2.3 |
| 1 152 € | classique (e-mail homme) | 9 600 (homme 9 600, femme 0) | 1 152 € | 14.51 % | +0.00 [+0.00 ; +0.00] | +0.0 |
| 1 152 € | uplift, e-mail homme seul | 9 600 (homme 9 600, femme 0) | 1 152 € | 14.73 % | +0.22 [-0.45 ; +0.85] | +2.2 |
| 1 152 € | optimisé (client + e-mail) | 9 971 (homme 7 745, femme 2 226) | 1 152 € | 14.74 % | +0.24 [-0.57 ; +0.98] | +2.4 |

## 3. Optimisation prudente (budget 30 %, TEST)

Uplift = moyenne de 5 modèles entraînés sur des sous-échantillons ; score optimisé = moyenne - z x écart-type entre modèles (z = 0 : aucune prudence).

Dispersion moyenne entre modèles : 0.45 pts (e-mail homme 0.60, femme 0.29), pour un uplift prédit moyen de 7.15 et 4.09 pts.

| prudence | contactés | taux de visite | gain vs sans prudence (pts) |
|---|---|---|---|
| z = 0 | 5 892 (homme 5 099, femme 793) | 12.95 % | +0.00 [+0.00 ; +0.00] |
| z = 0.5 | 5 904 (homme 5 037, femme 867) | 12.94 % | -0.00 [-0.18 ; +0.15] |
| z = 1 | 5 921 (homme 4 951, femme 970) | 12.88 % | -0.07 [-0.32 ; +0.18] |
| z = 2 | 5 969 (homme 4 711, femme 1 258) | 13.01 % | +0.06 [-0.30 ; +0.42] |

## 4. Contrôle sur le simulateur (64 000 clients, valeurs VRAIES)

Mêmes modèles, mêmes coûts, même protocole (test de 30 %), mais l'uplift varie fortement d'un client à l'autre (persuadables, « ne pas déranger », sensibilité propre à chaque e-mail). Sans aucun envoi, le taux de visite vrai est de 30.22 %.

| budget | politique | taux de visite (vrai) | visites en plus vs personne (pts) | part du gain de l'oracle |
|---|---|---|---|---|
| 10 % | classique (e-mail homme) | 30.77 % | +0.55 | 16 % |
| 10 % | optimisé (uplift prédit) | 33.29 % | +3.07 | 89 % |
| 10 % | oracle (uplift VRAI, inatteignable) | 33.66 % | +3.44 | 100 % |
| 30 % | classique (e-mail homme) | 33.52 % | +3.30 | 39 % |
| 30 % | optimisé (uplift prédit) | 37.94 % | +7.72 | 91 % |
| 30 % | oracle (uplift VRAI, inatteignable) | 38.66 % | +8.44 | 100 % |
| 50 % | classique (e-mail homme) | 36.74 % | +6.52 | 56 % |
| 50 % | optimisé (uplift prédit) | 41.17 % | +10.95 | 94 % |
| 50 % | oracle (uplift VRAI, inatteignable) | 41.90 % | +11.68 | 100 % |

*Durée totale : 2.5 min.*
