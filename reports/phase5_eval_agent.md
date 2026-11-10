# Phase 5 : évaluation automatique de l'agent

Généré le 2026-11-09 par `python -m ciblage.eval.run --rapport` ; code : eval @ fc7067b1 ; modèle évalué : openai/gpt-oss-120b (modèle de secours désactivé).

**Méthode** : les cas sont générés à partir de scénarios connus (la bonne réponse est connue d'avance, aucune annotation manuelle), puis formulés en français avec des gabarits variés. Niveau 1 = compréhension seule (1 appel au LLM par cas) ; niveau 2 = conversations complètes (système entier). Taux donnés avec le nombre de cas évalués et un IC de Wilson à 95 %.

## Niveau 1 : compréhension (97 cas évalués, 10 en erreur technique, exclus)

| métrique | réussis / évalués | taux | IC 95 % |
|---|---|---|---|
| Intention reconnue (LLM seul) | 94 / 97 | 97 % | [91 ; 99] % |
| Intention reconnue (après règles déterministes) | 94 / 97 | 97 % | [91 ; 99] % |
| Paramètres extraits exactement | 83 / 83 | 100 % | [96 ; 100] % |
| Aucun paramètre inventé (demandes à vérifier) | 34 / 34 | 100 % | [90 ; 100] % |
| Demande ambiguë -> question posée | 6 / 8 | 75 % | [41 ; 93] % |
| Demande claire -> pas de fausse ambiguïté | 30 / 30 | 100 % | [89 ; 100] % |
| Hors sujet et injections refusés | 0 / 0 | - | - |

### Par catégorie (intention après règles, paramètres exacts)

| categorie | cas | intention | parametres |
|---|---|---|---|
| ambigu | 8 | 100 % | 100 % |
| comparer | 1 | 100 % | 100 % |
| conseil | 14 | 100 % | 100 % |
| donnees | 4 | 100 % | 100 % |
| expliquer | 11 | 100 % | 100 % |
| exporter | 2 | 100 % | 100 % |
| modifier | 17 | 100 % | 100 % |
| plan | 24 | 88 % | 100 % |
| plan_incomplet | 10 | 100 % | 100 % |
| plan_profit | 6 | 100 % | 100 % |

Latence de la compréhension : médiane 1.3 s, maximum 24.1 s.

### Analyse des échecs

- `n1-003` (plan) « Je dispose de 1600 € pour ma campagne ; le mail homme revient à 13 centimes et le mail fem » → intention exporter
- `n1-007` (plan) « Je dispose de 680 € pour ma campagne ; le mail homme revient à 29 centimes et le mail femm » → intention conseil
- `n1-019` (plan) « Je dispose de 1450€ pour ma campagne ; l'e-mail homme revient à 5 cts et le mail femme à 0 » → intention conseil
- `n1-042` (ambigu) « Fais un plan avec 400, l'e-mail homme à 12 et l'e-mail femme à 10. » → intention plan, ambiguïté NON signalée
- `n1-047` (ambigu) « Budget de 900 € par mois, e-mail homme 0,12 €, e-mail femme 0,09 €. » → intention plan, ambiguïté NON signalée

## Niveau 2 : conversations complètes (2 conversations, 2 tours)

| vérification | réussis / évalués | taux | IC 95 % |
|---|---|---|---|
| Bon type de réponse finale (réponse / question) | 2 / 2 | 100 % | [34 ; 100] % |
| Scénario final correct | 2 / 2 | 100 % | [34 ; 100] % |
| Bonne analyse effectuée (plan, conseil, explication...) | 2 / 2 | 100 % | [34 ; 100] % |
| Conversation entièrement réussie | 2 / 2 | 100 % | [34 ; 100] % |
| Chiffres vérifiés dès la 1re rédaction | 2 / 2 | 100 % | [34 ; 100] % |
| Repli sur le texte standard (moins = mieux) | 0 / 2 | 0 % | [0 ; 66] % |

Par tour (hors tours sans LLM) : 2 appels au LLM en médiane  3 254 tokens en médiane  latence médiane 4.2 s (maximum 4.4 s).

### Analyse des échecs

Aucun échec.

## Limites

- Gabarits rédigés à la main : ils ne couvrent pas toute la variété des formulations réelles.
- Un seul passage par cas : la variabilité du LLM d'un appel à l'autre n'est pas mesurée.
- Échantillons modestes (quota gratuit) : les intervalles de confiance restent larges.
- La qualité rédactionnelle n'est pas notée ; seuls la fidélité des chiffres et le comportement le sont.
