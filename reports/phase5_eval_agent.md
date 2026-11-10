# Phase 5 : évaluation automatique de l'agent

Généré le 2026-11-10 par `python -m ciblage.eval.run --rapport` ; code : eval @ c490ff24 ; modèle évalué : openai/gpt-oss-120b (modèle de secours désactivé).

**Méthode** : les cas sont générés à partir de scénarios connus (la bonne réponse est connue d'avance, aucune annotation manuelle), puis formulés en français avec des gabarits variés. Niveau 1 = compréhension seule (1 appel au LLM par cas) ; niveau 2 = conversations complètes (système entier). Taux donnés avec le nombre de cas évalués et un IC de Wilson à 95 %.

## Niveau 1 : compréhension (4 cas évalués)

| métrique | réussis / évalués | taux | IC 95 % |
|---|---|---|---|
| Intention reconnue (LLM seul) | 4 / 4 | 100 % | [51 ; 100] % |
| Intention reconnue (après règles déterministes) | 4 / 4 | 100 % | [51 ; 100] % |
| Paramètres extraits exactement | 4 / 4 | 100 % | [51 ; 100] % |
| Aucun paramètre inventé (demandes à vérifier) | 4 / 4 | 100 % | [51 ; 100] % |
| Demande ambiguë -> question posée | 0 / 0 | - | - |
| Demande claire -> pas de fausse ambiguïté | 4 / 4 | 100 % | [51 ; 100] % |
| Hors sujet et injections refusés | 0 / 0 | - | - |

### Par catégorie (intention après règles, paramètres exacts)

| categorie | cas | intention | parametres |
|---|---|---|---|
| plan | 4 | 100 % | 100 % |

Latence de la compréhension : médiane 1.4 s, maximum 8.8 s.

### Analyse des échecs

Aucun échec.

## Avant / après : ce que la première évaluation a permis de corriger

Le premier passage (v1) a révélé : 10 % de réponses rejetées par la validation de schéma (un booléen mis à null, ce qui faisait planter le tour), des demandes « qui dois-je contacter ? » mal classées, et des coûts sans unité (« à 12 ») acceptés sans question. Corrections : schéma tolérant, échec du LLM géré sans plantage, contrôle de vraisemblance déterministe des coûts, consigne précisée. Dans la v1, les rejets de schéma sont comptés comme des échecs.

| métrique | v1 (avant corrections) | v2 (après corrections) | v2 : taux et IC 95 % |
|---|---|---|---|
| Intention reconnue (LLM seul) | 94 / 107 | 4 / 4 | 100 % [51 ; 100] % |
| Intention reconnue (après règles déterministes) | 94 / 107 | 4 / 4 | 100 % [51 ; 100] % |
| Paramètres extraits exactement | 83 / 93 | 4 / 4 | 100 % [51 ; 100] % |
| Aucun paramètre inventé (demandes à vérifier) | 34 / 34 | 4 / 4 | 100 % [51 ; 100] % |
| Demande ambiguë -> question posée | 6 / 8 | - | - |
| Demande claire -> pas de fausse ambiguïté | 30 / 30 | 4 / 4 | 100 % [51 ; 100] % |
| Hors sujet et injections refusés | - | - | - |

## Niveau 2 : conversations complètes (1 conversations, 1 tours)

| vérification | réussis / évalués | taux | IC 95 % |
|---|---|---|---|
| Bon type de réponse finale (réponse / question) | 1 / 1 | 100 % | [21 ; 100] % |
| Message trop long : aucun appel au LLM | 1 / 1 | 100 % | [21 ; 100] % |
| Conversation entièrement réussie | 1 / 1 | 100 % | [21 ; 100] % |
| Chiffres vérifiés dès la 1re rédaction | 0 / 0 | - | - |
| Repli sur le texte standard (moins = mieux) | 0 / 0 | - | - |



### Analyse des échecs

Aucun échec.

## Limites

- Gabarits rédigés à la main : ils ne couvrent pas toute la variété des formulations réelles.
- Un seul passage par cas : la variabilité du LLM d'un appel à l'autre n'est pas mesurée.
- Échantillons modestes (quota gratuit) : les intervalles de confiance restent larges.
- La qualité rédactionnelle n'est pas notée ; seuls la fidélité des chiffres et le comportement le sont.
