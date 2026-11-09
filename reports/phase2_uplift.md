# Phase 2 : moteur d'uplift

Généré le 2026-11-09 par `python -m ciblage.uplift.select`. Modèle de base : XGBoost 300 arbres, profondeur 4. Toutes les prédictions sont hors échantillon.

## 1. Simulateur (40 000 clients  uplift vrai connu)

PEHE = erreur quadratique moyenne entre uplift prédit et uplift vrai (plus bas = mieux). Prédictions hors échantillon (validation croisée à 5 plis).

| learner | PEHE homme (pts) | PEHE femme (pts) | corr. homme | corr. femme | Qini homme | Qini femme | temps (s) |
|---|---|---|---|---|---|---|---|
| S-learner | 3.24 | 3.72 | 0.94 | 0.91 | 0.094 | 0.082 | 8 |
| T-learner | 7.77 | 7.67 | 0.78 | 0.75 | 0.072 | 0.065 | 15 |
| X-learner | 6.14 | 5.98 | 0.83 | 0.82 | 0.081 | 0.071 | 32 |
| DR-learner | 8.31 | 8.18 | 0.74 | 0.72 | 0.071 | 0.063 | 79 |
| *référence : effet moyen pour tous* | 9.68 | 9.09 | - | - | - | - | - |
| *référence : uplift VRAI (inatteignable)* | 0.00 | 0.00 | 1.00 | 1.00 | 0.100 | 0.087 | - |

## 2. Hillstrom : visite (vraie campagne, 64 000 clients)

### Qualité du classement (hors échantillon)

Repère : envoyer l'e-mail à TOUT le monde donne +7.66 pts (homme) et +4.52 pts (femme). Un bon modèle fait mieux sur ses 30 % préférés.

| learner | Qini homme | AUUC homme | uplift@30% homme (pts) | Qini femme | AUUC femme | uplift@30% femme (pts) |
|---|---|---|---|---|---|---|
| S-learner | 0.012 | 0.006 | +8.20 | 0.059 | 0.027 | +7.02 |
| T-learner | 0.001 | 0.000 | +8.09 | 0.047 | 0.022 | +6.96 |
| X-learner | 0.002 | 0.001 | +8.07 | 0.049 | 0.022 | +7.34 |
| DR-learner | 0.001 | 0.001 | +8.04 | 0.044 | 0.020 | +6.53 |
| *ciblage classique (P(visite))* | 0.025 | 0.013 | +9.08 | 0.021 | 0.010 | +5.44 |

### Valeur des politiques de ciblage (IPW, IC 95 % bootstrap apparié)

Budget : 30 % des clients contactés. Le ciblage par uplift choisit AUSSI le meilleur e-mail pour chaque client.

| politique | clients contactés | taux de visite | gain vs ciblage classique (pts) |
|---|---|---|---|
| personne | 0 % | 10.62 % [10.22 ; 11.02] | -2.75 [-3.18 ; -2.30] |
| hasard 30 % (e-mail homme) | 30 % | 12.97 % [12.54 ; 13.40] | -0.40 [-0.85 ; +0.04] |
| ciblage classique 30 % (e-mail homme) | 30 % | 13.37 % [12.91 ; 13.81] | +0.00 [+0.00 ; +0.00] |
| uplift 30 % (S-learner, meilleur e-mail) | 30 % | 12.95 % [12.51 ; 13.41] | -0.41 [-0.78 ; -0.05] |
| tout le monde (e-mail homme) | 100 % | 18.28 % [17.78 ; 18.78] | +4.91 [+4.38 ; +5.40] |

### Calibration du S-learner : l'uplift prédit correspond-il à l'uplift observé ?

| e-mail | tranche (1 = plus fort uplift prédit) | prédit (pts) | observé (pts) |
|---|---|---|---|
| homme | 1 | +11.78 | +8.50 [+6.80 ; +10.21] |
| homme | 2 | +8.29 | +7.64 [+6.12 ; +9.17] |
| homme | 3 | +7.07 | +8.11 [+6.64 ; +9.58] |
| homme | 4 | +5.98 | +6.71 [+5.28 ; +8.14] |
| homme | 5 | +4.46 | +7.24 [+6.03 ; +8.45] |
| femme | 1 | +8.98 | +7.66 [+6.09 ; +9.24] |
| femme | 2 | +6.67 | +7.37 [+5.92 ; +8.82] |
| femme | 3 | +4.36 | +4.97 [+3.57 ; +6.37] |
| femme | 4 | +1.78 | +0.55 [-0.67 ; +1.78] |
| femme | 5 | +0.38 | +2.03 [+0.64 ; +3.43] |

## 3. Hillstrom : dépense (achats rares, environ 1 %, donc estimations bruitées)

| politique | clients contactés | dépense moyenne par client ($) | gain vs ciblage classique ($) |
|---|---|---|---|
| personne | 0 % | 0.653 [0.501 ; 0.814] | -0.416 [-0.603 ; -0.225] |
| ciblage classique 30 % (e-mail homme) | 30 % | 1.068 [0.866 ; 1.281] | +0.000 [+0.000 ; +0.000] |
| uplift 30 % (S-learner, meilleur e-mail) | 30 % | 0.881 [0.711 ; 1.073] | -0.187 [-0.380 ; +0.003] |
| tout le monde (e-mail homme) | 100 % | 1.423 [1.172 ; 1.655] | +0.354 [+0.150 ; +0.558] |

## 4. Criteo : visite (400 000 lignes tirées de l'échantillon  85 % traités)

Repère : effet moyen de la publicité = +1.06 pts. Validation croisée à 3 plis.

| learner | Qini | AUUC | uplift@30% (pts) | temps (s) |
|---|---|---|---|---|
| S-learner | 0.088 | 0.035 | +3.06 | 20 |
| T-learner | 0.073 | 0.029 | +2.80 | 25 |
| X-learner | 0.078 | 0.031 | +2.92 | 52 |
| DR-learner | 0.077 | 0.030 | +2.81 | 129 |
| *ciblage classique (P(visite))* | 0.086 | 0.034 | +2.99 | - |

*Learner retenu sur Hillstrom (meilleur Qini moyen) : **S-learner**. Durée totale : 9 min.*
