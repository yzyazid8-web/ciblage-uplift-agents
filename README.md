# Copilote de ciblage causal

**Qui contacter, avec quel e-mail, pour un budget donné ?** Un responsable marketing pose la question
en français ; le système estime pour chaque client ce que chaque e-mail change *vraiment* (uplift),
optimise l'allocation sous budget, vérifie le plan et l'explique — chaque chiffre contrôlé.

> **Démo en ligne** (Azure Container Apps, réveil ~1 min) : https://ciblage-copilote.politebeach-befb1bc3.francecentral.azurecontainerapps.io
> **Vitrine** (instantanée, résultats pré-calculés) : https://yzyazid8-web.github.io/ciblage-uplift-agents/

<!-- Démo animée à enregistrer : ![Démo](docs/demo.gif) -->

## Le problème

La plupart des campagnes ciblent les clients **les plus susceptibles d'acheter**. Beaucoup auraient
acheté **sans** l'e-mail : le budget est dépensé pour rien. La bonne question est causale :
*sur qui l'action change-t-elle le résultat ?*

| Type de client | Sans e-mail | Avec e-mail | Faut-il le contacter ? |
|---|---|---|---|
| Fidèle | visite | visite | non, inutile |
| **Persuadable** | ne visite pas | **visite** | **oui** |
| Perdu | ne visite pas | ne visite pas | non |
| « Ne pas déranger » | visite | ne visite plus | surtout pas |

## Ce que fait le système

```
Demande simple : « J'ai 700 €, l'e-mail homme coûte 0,12 €… »
  → compréhension (LLM, sortie structurée) → routeur déterministe → clarification si ambigu
  → uplift client par client (S-learner, retenu parmi S/T/X/DR) → optimisation sous budget
    (relaxation lagrangienne) → vérification indépendante → rédaction (LLM)
  → contrôle des chiffres et du verdict, puis ligne « Source »

Analyse en plusieurs étapes : « Compare 500, 700 et 1 000 €, lequel est le plus rentable ? »
  → agent analyste : le LLM choisit lui-même ses outils (plans, conseils, explications),
    au plus 5 étapes, outils en lecture seule → même contrôle des chiffres et du verdict
```

Le LLM **comprend, choisit des outils et rédige ; il ne calcule jamais** : tous les chiffres viennent
du code, et chaque nombre d'une réponse est vérifié avant affichage, ainsi que le verdict (une réponse
ne peut pas dire « prouvé » quand l'intervalle de confiance contient 0). Chaque réponse se termine par
une **ligne « Source »** qui dit de quel calcul elle provient — auditable.

**Pourquoi un workflow, plus un seul agent ?** Les demandes courantes suivent un chemin fixé par le
code : prévisible, testable, 2 appels au LLM par tour. Seules les questions qui demandent plusieurs
calculs sont confiées à un agent autonome, borné (5 étapes, outils sans effet). Je n'ai pas fait de
système « multi-agents » : un second LLM critique ferait moins bien que le vérificateur déterministe,
pour plus de coût et plus de pannes.

## Les effets sont-ils causaux ?

Oui, parce que les données viennent d'un **essai randomisé** : chaque client a reçu au hasard l'e-mail
homme, l'e-mail femme ou rien. La différence de visite mesure donc l'**effet** de l'e-mail, pas une
corrélation. L'effet d'un client précis reste **estimé** ; seul l'effet moyen est garanti (randomisation
+ pondération inverse IPW).

**Et sans A/B test ?** J'ai aussi traité le cas observationnel. En cassant volontairement la
randomisation de Hillstrom (biais connu), la simple différence de moyennes passe de la vérité
**+7,7 points** à **+10,4** (biaisée) ; un estimateur **doublement robuste** (AIPW : score de propension
+ modèle de réponse, codés à la main) la ramène à **+7,8**. Mais si on retire un facteur clé de
l'ajustement, le biais revient (**+9,0**) : aucune méthode ne corrige un confondant qu'elle ne voit pas.
L'agent sait expliquer tout cela (`python -m ciblage.causal.observational` → `reports/phase1b_observationnel.md`).

## Résultats (tous générés par script, voir `reports/`)

**Données** : Hillstrom, vraie campagne e-mail randomisée (64 000 clients) ; Criteo Uplift (14 M de
lignes) pour l'échelle ; un simulateur à uplift connu pour la vérité terrain.

- **Effet moyen** (phase 1) : e-mail homme +7,7 points de visite, e-mail femme +4,5 (base 10,6 %) ;
  l'effet de l'e-mail femme varie fortement selon ce que le client a déjà acheté.
- **Uplift** (phase 2, 30 % de test mis de côté avant tout réglage) : à e-mail égal, cibler par uplift
  rapporte **+1,0 point de visite [+0,4 ; +1,6]** de plus que le ciblage classique pour l'e-mail femme
  (S-learner ; T-learner +1,2 ; X et DR non significatifs) ; **aucun gain** pour l'e-mail homme, dont
  l'effet est presque uniforme.
- **Optimisation** (phase 3) : relaxation lagrangienne codée à la main, **écart prouvé à l'optimum
  ≤ 0,04 % dès 5 000 clients** ; **1 M de clients en ~6 s** avec le budget seul (~42 s avec une limite
  d'envois) ; à 64 000 clients, 0,6 s contre 19 s pour la relaxation linéaire HiGHS. Sur données
  simulées (uplift hétérogène), le plan optimisé capte **89–94 % du gain maximal**, contre 16–56 % pour
  le ciblage classique. Sur Hillstrom, il fait jeu égal avec le ciblage classique : je l'écris tel quel.
- **Agent** (phase 5 : 119 messages et 30 conversations générés à partir de scénarios connus, sans
  annotation ; IC de Wilson à 95 %) :
  - compréhension : intention reconnue **116/119 (97 %)**, paramètres exacts **82/83**, **aucun
    paramètre inventé (34/34)** ;
  - conversations complètes : **29/30 réussies**, export seulement après « oui » 5/5, aucune fuite des
    consignes sous injection (3/3) ; chiffres vérifiés dès la 1ʳᵉ rédaction dans 91 % des cas ;
  - coût : 2 appels au LLM et ~3 000 tokens par tour en médiane, latence médiane 5 s ;
  - points faibles assumés : ambiguïtés signalées 5/8 seulement, et 3 injections sur 6 font échouer la
    compréhension (le système répond alors par une excuse, sans rien divulguer ni exécuter).

## Ce que j'ai codé moi-même (et validé)

| Brique | Validation |
|---|---|
| S-, T-, X-, DR-learners multi-traitements | uplift retrouvé sur le simulateur |
| Qini, AUUC, courbes d'uplift | identiques à scikit-uplift |
| Valeur de politique par IPW + bootstrap apparié | vraie valeur retrouvée sur le simulateur |
| Relaxation lagrangienne + borne d'optimalité | ≥ 98 % de l'optimum MILP exact, borne = relaxation linéaire |
| Inférence causale hors A/B test (propension, IPW, AIPW doublement robuste) | effet vrai retrouvé après biais injecté sur Hillstrom |
| Workflow LangGraph (routeur, clarification, export confirmé) + agent analyste à outils | tests avec faux LLM + tests réels |
| Contrôle de fidélité des réponses (chiffres et verdict) | cas trouvés par un audit externe |
| Évaluation automatique de l'agent | IC de Wilson, échecs du modèle comptés, pannes exclues |

## Lancer le projet

```bash
# Application complète (tableau de bord + agent). Sans clé, le formulaire et l'API /plan fonctionnent.
docker build -t ciblage .
docker run -p 8080:8080 -e GROQ_API_KEY=gsk_... ciblage        # http://localhost:8080
docker run -p 8000:8000 ciblage uvicorn ciblage.serve.api:app --host 0.0.0.0 --port 8000  # /docs
```

En local : `pip install -r requirements-dev.txt && pip install -e . && marimo run app.py`.
Rapports : `python -m ciblage.causal.report`, `python -m ciblage.causal.observational`,
`python -m ciblage.uplift.select`, `python -m ciblage.optim.benchmark`, `python -m ciblage.eval.run --rapport`.

## Limites

- Une seule campagne (Hillstrom, 2008, deux semaines) ; les **coûts des e-mails sont des hypothèses**.
- L'uplift d'un client est **estimé**, jamais observé ; la garantie de mesure (IPW) repose sur la
  randomisation de la campagne.
- Sur Hillstrom, l'e-mail principal a un effet presque uniforme : peu à optimiser. Le gain de la méthode
  se voit quand l'effet varie (e-mail femme, simulateur).
- Le **choix entre les deux e-mails** n'est pas validé par les données : le modèle prédit des écarts
  d'effet que la mesure ne confirme pas. Les raisons données par l'agent sont fidèles à l'optimiseur,
  pas une preuve causale individuelle.
- Agent : gabarits de test écrits à la main, un seul passage par cas, quota gratuit (IC larges).

## Stack

Python 3.12 · pandas · scikit-learn · XGBoost · SciPy/HiGHS · LangGraph · LangChain · Groq
(gpt-oss-120b) · marimo · Plotly · FastAPI · MLflow · Docker · GitHub Actions · pytest · ruff · mypy
