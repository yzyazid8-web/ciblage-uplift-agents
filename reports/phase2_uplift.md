# Phase 2 : moteur d'uplift

Généré le 2026-11-09 par `python -m ciblage.uplift.select`. Modèle de base : XGBoost. Réglages par défaut (300 arbres, profondeur 4) pour le simulateur et Criteo ; réglés par validation croisée sur Hillstrom. Toutes les prédictions sont hors échantillon.

## 1. Simulateur (40 000 clients, uplift vrai connu)

PEHE = erreur quadratique moyenne entre uplift prédit et uplift vrai (plus bas = mieux). Prédictions hors échantillon (validation croisée à 5 plis).

| learner | PEHE homme (pts) | PEHE femme (pts) | corr. homme | corr. femme | Qini homme | Qini femme | temps (s) |
|---|---|---|---|---|---|---|---|
| S-learner | 3.24 | 3.72 | 0.94 | 0.91 | 0.094 | 0.082 | 6 |
| T-learner | 7.77 | 7.67 | 0.78 | 0.75 | 0.072 | 0.065 | 11 |
| X-learner | 6.14 | 5.98 | 0.83 | 0.82 | 0.081 | 0.071 | 27 |
| DR-learner | 8.31 | 8.18 | 0.74 | 0.72 | 0.071 | 0.063 | 61 |
| *référence : effet moyen pour tous* | 9.68 | 9.09 | - | - | - | - | - |
| *référence : uplift VRAI (inatteignable)* | 0.00 | 0.00 | 1.00 | 1.00 | 0.100 | 0.087 | - |

## 2. Hillstrom : visite (vraie campagne, 64 000 clients)

**Protocole** : 19 200 clients (30 %) mis de côté AVANT tout réglage ; réglage par validation croisée à 5 plis sur les 44 800 autres, critère fixé à l'avance = Qini moyen des deux e-mails ; le test ne sert qu'une fois, pour tous les chiffres ci-dessous.

### Réglage (entraînement uniquement)

| learner | configurations testées | Qini CV défaut | Qini CV réglé | réponse retenue | effet retenu | durée (min) |
|---|---|---|---|---|---|---|
| S-learner | 13 | 0.0361 | 0.0388 | {'max_depth': 2, 'n_estimators': 300, 'min_child_weight': 5} | - | 0.9 |
| T-learner | 13 | 0.0246 | 0.0332 | {'max_depth': 2, 'n_estimators': 100, 'min_child_weight': 100} | - | 1.9 |
| X-learner | 25 | 0.0216 | 0.0342 | {'max_depth': 2, 'n_estimators': 100, 'min_child_weight': 5} | {'max_depth': 2, 'n_estimators': 50, 'min_child_weight': 50} | 6.3 |
| DR-learner | 25 | 0.0218 | 0.0330 | {'max_depth': 2, 'n_estimators': 100, 'min_child_weight': 100} | {'max_depth': 2, 'n_estimators': 50, 'min_child_weight': 300} | 16.9 |

### Qualité du classement sur le TEST

Repère sur le test : e-mail à tout le monde = +7.65 pts (homme), +4.51 pts (femme).

| learner | Qini homme | uplift@30% homme (pts) | Qini femme | uplift@30% femme (pts) |
|---|---|---|---|---|
| S-learner | 0.011 | +7.74 | 0.062 | +8.41 |
| T-learner | 0.006 | +8.92 | 0.085 | +8.90 |
| X-learner | 0.007 | +8.47 | 0.082 | +7.85 |
| DR-learner | 0.002 | +8.92 | 0.080 | +8.09 |
| *ciblage classique (P(visite))* | 0.009 | +7.88 | 0.024 | +5.39 |

### Valeur des politiques sur le TEST (IPW, IC 95 % bootstrap apparié)

Budget : 30 % des clients. « classique homme » = les 30 % les plus susceptibles de visiter reçoivent l'e-mail homme (pratique courante). Les lignes « e-mail homme » et « e-mail femme » isolent l'effet du CLASSEMENT ; la ligne « meilleur e-mail » ajoute le CHOIX de l'e-mail client par client.

| politique | contactés | résultat moyen par client | gain vs classique homme (pts) |
|---|---|---|---|
| personne | 0 % | 10.62 % [9.86 % ; 11.40 %] | -2.37 [-3.18 ; -1.58] |
| hasard 30 %, e-mail homme | 30 % | 12.62 % [11.81 % ; 13.46 %] | -0.37 [-1.17 ; +0.47] |
| classique homme | 30 % | 12.99 % [12.16 % ; 13.77 %] | +0.00 [+0.00 ; +0.00] |
| uplift (S-learner) 30 %, e-mail homme | 30 % | 12.84 % [11.99 % ; 13.66 %] | -0.15 [-0.79 ; +0.47] |
| classique 30 %, e-mail femme | 30 % | 12.13 % [11.36 % ; 12.93 %] | -0.86 [-1.67 ; -0.05] |
| uplift (S-learner) 30 %, e-mail femme | 30 % | 13.16 % [12.41 % ; 14.00 %] | +0.17 [-0.74 ; +1.04] |
| uplift (S-learner) 30 %, meilleur e-mail par client | 30 % | 12.83 % [11.99 % ; 13.64 %] | -0.16 [-0.82 ; +0.46] |
| tout le monde, e-mail homme | 100 % | 18.27 % [17.32 % ; 19.19 %] | +5.28 [+4.32 ; +6.17] |

### Calibration du S-learner sur le TEST

| e-mail | tranche (1 = plus fort uplift prédit) | prédit (pts) | observé (pts) |
|---|---|---|---|
| homme | 1 | +11.55 | +8.54 [+5.29 ; +11.78] |
| homme | 2 | +8.52 | +7.00 [+4.14 ; +9.86] |
| homme | 3 | +6.90 | +8.73 [+5.99 ; +11.47] |
| homme | 4 | +5.43 | +7.31 [+4.88 ; +9.74] |
| homme | 5 | +3.57 | +7.24 [+5.29 ; +9.18] |
| femme | 1 | +7.14 | +7.57 [+4.45 ; +10.69] |
| femme | 2 | +5.63 | +7.95 [+5.29 ; +10.61] |
| femme | 3 | +3.73 | +4.87 [+2.35 ; +7.38] |
| femme | 4 | +2.54 | +0.04 [-2.47 ; +2.55] |
| femme | 5 | +1.64 | +2.67 [+0.83 ; +4.51] |

## 3. Hillstrom : dépense sur le TEST (achats rares, environ 1 % : estimations bruitées)

Réglages repris de la visite (pas de réglage spécifique : trop peu d'achats).

| politique | contactés | résultat moyen par client | gain vs classique homme ($) |
|---|---|---|---|
| personne | 0 % | 0.793 $ [0.523 $ ; 1.123 $] | -0.420 [-0.781 ; -0.075] |
| classique homme | 30 % | 1.213 $ [0.856 $ ; 1.603 $] | +0.000 [+0.000 ; +0.000] |
| uplift (S-learner) 30 %, meilleur e-mail par client | 30 % | 1.068 $ [0.730 $ ; 1.455 $] | -0.145 [-0.522 ; +0.213] |
| tout le monde, e-mail homme | 100 % | 1.511 $ [1.076 $ ; 1.955 $] | +0.298 [-0.103 ; +0.701] |

## 4. Criteo : visite (400 000 lignes tirées de l'échantillon, 85 % traités ; réglages par défaut)

Repère : effet moyen de la publicité = +1.06 pts. Validation croisée à 3 plis.

| learner | Qini | AUUC | uplift@30% (pts) | temps (s) |
|---|---|---|---|---|
| S-learner | 0.088 | 0.035 | +3.06 | 22 |
| T-learner | 0.073 | 0.029 | +2.80 | 27 |
| X-learner | 0.078 | 0.031 | +2.92 | 55 |
| DR-learner | 0.077 | 0.030 | +2.81 | 86 |
| *ciblage classique (P(visite))* | 0.086 | 0.034 | +2.99 | - |

*Learner retenu sur Hillstrom (meilleur Qini en validation croisée sur l'entraînement) : **S-learner**. Durée totale : 32 min.*
