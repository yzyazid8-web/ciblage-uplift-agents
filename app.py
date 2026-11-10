import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Copilote de ciblage", css_file="app.css")


@app.cell
def _():
    import uuid
    from dataclasses import asdict

    import marimo as mo
    import plotly.graph_objects as go

    from ciblage.serve.planner import Planner, Scenario

    # Palette de référence validée (daltonisme) : slot 1 bleu, slot 2 orange ; neutres pour le reste
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b4b2aa"
    DEFAULT = Scenario(budget=700.0, cost_homme=0.12, cost_femme=0.10)
    return (
        BLUE,
        DEFAULT,
        GRID,
        INK,
        MUTED,
        NEUTRAL,
        ORANGE,
        Planner,
        SURFACE,
        Scenario,
        asdict,
        go,
        mo,
        uuid,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Copilote de ciblage

        **Qui contacter, avec quel e-mail, pour un budget donné ?** Le plan vise les clients sur qui
        l'e-mail **change vraiment** le résultat (uplift), pas seulement ceux qui auraient visité
        de toute façon.

        Portefeuille : **19 200 clients** d'une vraie campagne randomisée (Hillstrom), jamais vus par
        les modèles. Comme les e-mails y ont été envoyés au hasard, on peut **mesurer** ce que chaque
        plan aurait rapporté, et pas seulement le prédire.

        Décris ta campagne à **l'agent, dans le panneau de droite**, ou utilise le formulaire
        ci-dessous : les résultats se mettent à jour dans les deux cas.
        """
    )
    return


@app.cell
def _(Planner, mo):
    with mo.status.spinner(title="Entraînement des modèles (environ 5 s)…"):
        planner = Planner()
    return (planner,)


@app.cell
def _(DEFAULT, mo):
    # état partagé : le scénario affiché par les graphiques, piloté par le formulaire OU l'agent
    get_scenario, set_scenario = mo.state(DEFAULT)
    get_source, set_source = mo.state("valeurs par défaut")
    return get_scenario, get_source, set_scenario, set_source


@app.cell
def _(mo, planner, uuid):
    from ciblage.agents.copilot import SESSIONS_DB, Copilot
    from ciblage.agents.llm import LLMUnavailable

    try:
        copilot = Copilot.from_env(planner=planner, db_path=SESSIONS_DB)
        agent_error = None
    except LLMUnavailable as _err:
        copilot, agent_error = None, str(_err)
    session_id = f"marimo-{uuid.uuid4().hex[:8]}"
    return agent_error, copilot, session_id


@app.cell
def _(
    Scenario,
    agent_error,
    asdict,
    copilot,
    get_scenario,
    mo,
    session_id,
    set_scenario,
    set_source,
):
    def agent_model(messages, config):
        """Un tour de conversation, diffusé en direct ; met à jour les graphiques si le plan change."""
        text = messages[-1].content
        # synchronisation dans l'autre sens : l'agent part du plan que l'utilisateur a sous les yeux
        copilot.sync_scenario(session_id, asdict(get_scenario()))
        steps = []
        for event in copilot.stream(session_id, text):
            if event["type"] == "etape":
                steps.append(event["texte"])
                yield f"› {event['texte']}  \n"
                continue
            yield "\n---\n\n" + event["texte"]
            scenario = event.get("scenario")
            if event["type"] == "reponse" and scenario and Scenario(**scenario) != get_scenario():
                set_scenario(Scenario(**scenario))
                set_source("conversation avec l'agent")

    if copilot is None:
        chat = mo.callout(
            mo.md(f"**Agent indisponible** : {agent_error}. Le formulaire reste utilisable."),
            kind="warn",
        )
    else:
        chat = mo.ui.chat(
            agent_model,
            prompts=[
                "J'ai 700 €, l'e-mail homme coûte 0,12 € et l'e-mail femme 0,10 €.",
                "Et avec 1 000 € ?",
                "Est-ce que 100 € de plus valent le coup ?",
                "Pourquoi le client 12 est-il contacté ou non ?",
                "Qui sont les clients écartés ?",
                "Compare les scénarios.",
            ],
            max_height=520,
        )
    # panneau isolé à droite (voir app.css), qui défile indépendamment des résultats
    mo.sidebar(
        [
            mo.md("### Agent de ciblage"),
            chat,
            mo.md(f"*Session `{session_id}` : sauvegardée, reprenable via l'API.*"),
        ],
        width="460px",
    )
    return


@app.cell
def _(mo):
    # grille de 3 colonnes : chaque libellé au-dessus de son champ
    def _field(label: str, key: str) -> str:
        return (
            f"<div><div style='font-weight:600;margin-bottom:.25rem'>{label}</div>{{{key}}}</div>"
        )

    _grid = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.9rem 1.25rem"
    form = (
        mo.Html(
            f"<div style='{_grid}'>"
            + _field("Budget (€)", "budget")
            + _field("Coût e-mail homme (€)", "cost_h")
            + _field("Coût e-mail femme (€)", "cost_f")
            + _field("Envois max. e-mail homme", "max_h")
            + _field("Envois max. e-mail femme", "max_f")
            + _field("Valeur d'une visite (€)", "value")
            + "</div>"
            + "<div style='margin-top:.9rem'><b>Objectif</b> {objective}</div>"
            + "<div style='color:var(--muted-foreground);font-size:.85em;margin-top:.4rem'>"
            + "Envois max. : 0 = sans limite. La valeur d'une visite ne sert qu'à l'objectif profit."
            + "</div>"
        )
        .batch(
            budget=mo.ui.number(start=0, stop=10_000, step=10, value=700, full_width=True),
            cost_h=mo.ui.number(start=0, stop=10, step=0.01, value=0.12, full_width=True),
            cost_f=mo.ui.number(start=0, stop=10, step=0.01, value=0.10, full_width=True),
            max_h=mo.ui.number(start=0, stop=20_000, step=100, value=0, full_width=True),
            max_f=mo.ui.number(start=0, stop=20_000, step=100, value=0, full_width=True),
            objective=mo.ui.radio(
                options={"Maximiser les visites": "visites", "Maximiser le profit": "profit"},
                value="Maximiser les visites",
                inline=True,
            ),
            value=mo.ui.number(start=0, stop=100, step=0.5, value=5.0, full_width=True),
        )
        .form(submit_button_label="Calculer le plan", bordered=True)
    )
    return (form,)


@app.cell
def _(form):
    form
    return


@app.cell
def _(Scenario, form, set_scenario, set_source):
    # le formulaire pilote le scénario affiché quand on le soumet
    if form.value is not None:
        _v = form.value
        set_scenario(
            Scenario(
                budget=float(_v["budget"]),
                cost_homme=float(_v["cost_h"]),
                cost_femme=float(_v["cost_f"]),
                max_homme=int(_v["max_h"]) or None,
                max_femme=int(_v["max_f"]) or None,
                objective=_v["objective"],
                value_per_visit=float(_v["value"]),
            )
        )
        set_source("formulaire")
    return


@app.cell
def _(get_scenario, get_source, mo, planner):
    scenario = get_scenario()
    _errors = scenario.errors()
    if _errors:
        plan = None
        _out = mo.callout(mo.md("\n".join(f"- {e}" for e in _errors)), kind="danger")
    else:
        with mo.status.spinner(title="Optimisation et mesure du plan…"):
            plan = planner.plan(scenario, n_boot=300)
        _limits = ", ".join(
            f"max {n} e-mails {k}"
            for k, n in (("homme", scenario.max_homme), ("femme", scenario.max_femme))
            if n
        )
        _budget = f"{scenario.budget:,.0f}".replace(",", " ")
        _out = mo.callout(
            mo.md(
                f"**Plan affiché** (source : {get_source()}) : budget {_budget} €, "
                f"e-mail homme {scenario.cost_homme:.2f} €, e-mail femme {scenario.cost_femme:.2f} €, "
                f"objectif {scenario.objective}{' ; ' + _limits if _limits else ''}."
            ),
            kind="info",
        )
    _out
    return (plan,)


@app.cell
def _(mo, plan):
    mo.stop(plan is None)
    _s = plan.summary
    _c = plan.comparison.set_index("politique")
    _g = _c.loc["plan optimisé"]
    _gain = f"{_g['gain vs classique (pts)']:+.2f} pts"
    _ci = f"IC 95 % [{_g['gain_bas']:+.2f} ; {_g['gain_haut']:+.2f}]"
    # 3 + 2 : le plan (qui, combien, quoi), puis les mesures pour décider
    _row1 = [
        mo.stat(
            f"{_s['clients_contactes']:,}".replace(",", " "),
            label="Clients contactés",
            caption=f"homme {_s['email_homme']:,} · femme {_s['email_femme']:,}".replace(",", " "),
            bordered=True,
        ),
        mo.stat(
            f"{_s['depense']:,.0f} €".replace(",", " "),
            label="Dépense",
            caption=f"budget non utilisé : {_s['budget_inutilise']:.0f} €",
            bordered=True,
        ),
        mo.stat(
            f"{_s['visites_en_plus_predites']:,.0f}".replace(",", " "),
            label="Visites en plus (prédites)",
            caption=f"{_s['cout_par_visite_en_plus']:.2f} € par visite gagnée",
            bordered=True,
        ),
    ]
    _row2 = [
        mo.stat(_gain, label="Gain mesuré vs ciblage classique", caption=_ci, bordered=True),
        mo.stat(
            f"{_s['prix_marginal_budget']:.2f}",
            label="Visites par euro en plus",
            caption=f"écart prouvé à l'optimum ≤ {100 * plan.bound_gap:.2f} %",
            bordered=True,
        ),
    ]
    mo.vstack(
        [
            mo.hstack(_row1, widths="equal", gap=0.5),
            mo.hstack(_row2, widths="equal", gap=0.5),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(BLUE, GRID, INK, MUTED, NEUTRAL, SURFACE, go, mo, plan):
    mo.stop(plan is None)
    _c = plan.comparison.iloc[::-1]  # « personne » en bas
    _colors = [BLUE if p == "plan optimisé" else NEUTRAL for p in _c["politique"]]
    _fig = go.Figure(
        go.Scatter(
            x=100 * _c["taux de visite"],
            y=_c["politique"],
            mode="markers",
            marker={"size": 12, "color": _colors, "line": {"color": SURFACE, "width": 2}},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": 100 * (_c["ic_haut"] - _c["taux de visite"]),
                "arrayminus": 100 * (_c["taux de visite"] - _c["ic_bas"]),
                "color": MUTED,
                "thickness": 2,
                "width": 0,
            },
            customdata=_c[["contactés", "dépense (€)", "gain vs classique (pts)"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Taux de visite : %{x:.2f} %<br>Contactés : %{customdata[0]:,}"
                "<br>Dépense : %{customdata[1]:,.0f} €<br>Gain vs classique : %{customdata[2]:+.2f} pts"
                "<extra></extra>"
            ),
        )
    )
    _fig.update_layout(
        title={
            "text": "Taux de visite mesuré sur le portefeuille (IC 95 %)",
            "x": 0,
            "font": {"color": INK},
        },
        xaxis={"title": "Taux de visite (%)", "gridcolor": GRID, "zeroline": False, "color": MUTED},
        yaxis={"color": INK},
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        height=300,
        margin={"l": 10, "r": 20, "t": 50, "b": 40},
        showlegend=False,
    )
    _fig
    return


@app.cell
def _(BLUE, GRID, INK, MUTED, NEUTRAL, ORANGE, SURFACE, go, mo, plan):
    mo.stop(plan is None)
    _seg = plan.by_segment
    _fig = go.Figure()
    for _col, _color in (("e-mail homme", BLUE), ("e-mail femme", ORANGE), ("rien", NEUTRAL)):
        _fig.add_bar(
            y=_seg["segment"],
            x=_seg[_col],
            name=_col,
            orientation="h",
            marker={"color": _color, "line": {"color": SURFACE, "width": 2}},
            hovertemplate=f"<b>%{{y}}</b><br>{_col} : %{{x:,}} clients<extra></extra>",
        )
    _fig.update_layout(
        barmode="stack",
        title={
            "text": "Qui reçoit quoi, selon ce que le client a déjà acheté",
            "x": 0,
            "font": {"color": INK},
        },
        xaxis={"title": "Clients", "gridcolor": GRID, "color": MUTED},
        yaxis={"color": INK},
        legend={"orientation": "h", "y": -0.25, "x": 0},
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        height=300,
        margin={"l": 10, "r": 20, "t": 50, "b": 70},
    )
    mo.vstack(
        [
            _fig,
            mo.ui.table(
                _seg, selection=None, label="Détail par segment (uplift prédit moyen en points)"
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Comment lire ces chiffres ?": mo.md(
                """
                - **Visites en plus (prédites)** : ce que le modèle d'uplift attend du plan.
                - **Gain mesuré** : ce que le plan aurait réellement rapporté, estimé sur la campagne
                  randomisée (pondération inverse, IPW), comparé au **ciblage classique** qui envoie
                  l'e-mail homme aux clients les plus susceptibles de visiter, au même budget.
                - **Visites par euro en plus** : ce que rapporterait un euro de budget supplémentaire
                  (prix marginal issu de la relaxation lagrangienne). C'est aussi le **seuil de
                  rentabilité** : un client est contacté si son e-mail rapporte plus que ce seuil.
                - **Écart prouvé à l'optimum** : le plan est au plus à cet écart de la meilleure
                  allocation possible.
                - **Coûts** : ce sont des hypothèses saisies ici (les données n'en contiennent pas).
                """
            ),
            "Ce que fait l'agent (et ce qu'il ne fait pas)": mo.md(
                """
                - Le LLM **comprend** ta demande et **rédige** la réponse ; il ne calcule **jamais** :
                  tous les chiffres viennent du code (uplift, optimisation, mesures).
                - Chaque nombre de la réponse est **vérifié** contre les calculs avant affichage ; en
                  cas d'écart, la réponse est réécrite, puis remplacée par un texte standard sûr.
                - S'il manque une information ou si la demande est ambiguë, il **pose une question**.
                - L'export du fichier de campagne n'est fait qu'après ta **confirmation** explicite.
                - Les étapes (› …) s'affichent en direct au-dessus de chaque réponse.
                """
            ),
            "Ce qu'il faut savoir sur ce jeu de données": mo.md(
                """
                Sur Hillstrom, l'e-mail homme a un effet **presque uniforme** (+7 à +9 points de visite
                pour tous les profils) : le ciblage classique est alors un bon substitut, et le plan
                optimisé fait jeu égal avec lui. L'uplift fait la différence quand l'effet **varie**
                d'un client à l'autre : c'est le cas de l'e-mail femme (+1 point de visite à e-mail
                égal) et, sur données simulées, le plan optimisé capte environ 90 % du gain maximal
                contre 16 à 56 % pour le ciblage classique (voir `reports/phase3_optim.md`).
                """
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
