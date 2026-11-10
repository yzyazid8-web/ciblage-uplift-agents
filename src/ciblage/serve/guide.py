"""Guide de l'application : ce que l'agent sait de l'outil lui-même (but, écran, méthode, limites)."""

from __future__ import annotations

from typing import Any

from ciblage.serve.planner import Planner
from ciblage.uplift import chosen

HILLSTROM_CLIENTS = 64_000  # campagne e-mail Hillstrom (MineThatData, 2008)


def app_guide(planner: Planner) -> dict[str, Any]:
    return {
        "role": (
            "Copilote de ciblage : il dit qui contacter, avec quel e-mail (homme ou femme) et pour "
            "quel budget, puis mesure ce que le plan aurait rapporté face au ciblage classique."
        ),
        "idee_cle": (
            "Il vise les clients sur qui l'e-mail change vraiment le comportement (uplift), pas "
            "ceux qui auraient visité le site de toute façon."
        ),
        "donnees": {
            "source": "vraie campagne d'e-mails randomisée Hillstrom (MineThatData)",
            "clients_de_la_campagne": HILLSTROM_CLIENTS,
            "clients_du_portefeuille_affiche": planner.n_clients,
            "part_mise_de_cote_jamais_vue_par_les_modeles_pct": round(100 * chosen.TEST_SIZE),
            "tirage_au_sort": (
                "chaque client a reçu au hasard l'e-mail homme, l'e-mail femme ou rien, à parts "
                "égales : on peut donc mesurer ce qu'un plan aurait rapporté, pas seulement le prédire"
            ),
            "suivi_apres_envoi_semaines": 2,
            "resultat_mesure": "visite du site dans les semaines suivant l'envoi",
            "couts_des_emails": "absents des données : ce sont des hypothèses saisies par l'utilisateur",
        },
        "methode": {
            "etape_1_uplift": (
                "un modèle (S-learner sur XGBoost) prédit, pour chaque client et chaque e-mail, de "
                "combien l'e-mail augmente sa probabilité de visite"
            ),
            "etape_2_optimisation": (
                "un optimiseur (relaxation lagrangienne) choisit qui reçoit quel e-mail en respectant "
                "le budget et les limites d'envois ; il fournit le seuil de rentabilité et une preuve "
                "de l'écart à la meilleure allocation possible"
            ),
            "etape_3_mesure": (
                "grâce au tirage au sort, le plan est évalué sur les vrais comportements (pondération "
                "inverse, IPW) et comparé au ciblage classique, avec des intervalles de confiance "
                "(bootstrap)"
            ),
            "etape_4_agent": (
                "un agent (LangGraph et un LLM) comprend les demandes en français, lance les calculs "
                "et rédige les réponses ; chaque chiffre est vérifié avant affichage"
            ),
            "niveau_de_confiance_pct": 95,
        },
        "ecran": {
            "zone_de_gauche_de_haut_en_bas": [
                "le formulaire, pour calculer un plan sans l'agent : budget, coût de chaque e-mail, "
                "limites d'envois (zéro = sans limite), objectif visites ou profit, valeur d'une "
                "visite, bouton « Calculer le plan »",
                "le bandeau « Plan affiché », qui rappelle le scénario et sa source (formulaire, "
                "agent ou valeurs par défaut)",
                "les cinq indicateurs : clients contactés, dépense, visites en plus prédites, gain "
                "mesuré face au ciblage classique, visites par euro en plus",
                "le graphique « Taux de visite mesuré sur le portefeuille » : le taux de visite "
                "mesuré de quatre politiques (personne, ciblage classique, plan optimisé, tout le "
                "monde avec l'e-mail homme), avec leur intervalle de confiance",
                "le graphique « Qui reçoit quoi, selon ce que le client a déjà acheté » : pour "
                "chaque segment, les clients qui reçoivent l'e-mail homme, l'e-mail femme ou rien ; "
                "le tableau dessous ajoute l'uplift prédit moyen",
                "trois volets dépliables : lire les chiffres, ce que fait l'agent, ce qu'il faut "
                "savoir sur ces données",
            ],
            "panneau_de_droite": (
                "uniquement la conversation avec l'agent (aucun résultat ni graphique) : on lui "
                "décrit sa campagne en français ou on lui pose une question ; ses étapes s'affichent "
                "en direct, et quand il calcule un plan, les résultats de la zone de gauche se "
                "mettent à jour"
            ),
            "defilement": "la zone de gauche et le panneau de droite défilent séparément",
        },
        "agent": {
            "sait_faire": [
                "calculer un plan à partir d'un budget et des coûts des e-mails",
                "le recalculer après un changement (« et avec la moitié du budget ? »)",
                "comparer les scénarios calculés",
                "dire si un budget plus élevé vaut le coup, ou à partir de quel coût un e-mail ne sert plus",
                "expliquer chaque résultat affiché : indicateurs, graphiques, décision d'un client, clients écartés",
                "décrire le portefeuille de clients",
                "préparer le fichier de campagne, après confirmation explicite",
            ],
            "garde_fous": [
                "le LLM comprend et rédige, il ne calcule jamais : les chiffres viennent du code",
                "chaque nombre de la réponse est comparé aux calculs ; sinon la réponse est réécrite, puis remplacée par un texte standard",
                "s'il manque une information ou si la demande est ambiguë, il pose une question",
            ],
            "exemples_de_questions": [
                "J'ai 700 €, l'e-mail homme coûte 0,12 € et l'e-mail femme 0,10 €.",
                "Interprète le graphique 2.",
                "Explique-moi les indicateurs.",
                "Est-ce que 100 € de plus valent le coup ?",
                "Pourquoi le client 12 n'est-il pas contacté ?",
            ],
        },
        "resultats_du_projet": {
            "effet_moyen_email_homme_points": 7.7,
            "effet_moyen_email_femme_points": 4.5,
            "taux_de_visite_sans_email_pct": 10.6,
            "simulation_part_du_gain_maximal_plan_optimise_pct_min": 89,
            "simulation_part_du_gain_maximal_plan_optimise_pct_max": 94,
            "simulation_part_du_gain_maximal_ciblage_classique_pct_min": 16,
            "simulation_part_du_gain_maximal_ciblage_classique_pct_max": 56,
            "lecture": (
                "Sur ces données, l'e-mail homme a un effet presque uniforme : le ciblage classique "
                "est alors un bon substitut et le plan fait jeu égal avec lui. L'uplift fait la "
                "différence quand l'effet varie d'un client à l'autre, comme sur les données simulées."
            ),
        },
    }
