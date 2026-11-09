import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Copilote de ciblage")


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go

    from ciblage.serve.planner import Planner, Scenario

    # Palette de référence validée (daltonisme) : slot 1 bleu, slot 2 orange ; neutres pour le reste
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b4b2aa"
    return BLUE, GRID, INK, MUTED, NEUTRAL, ORANGE, Planner, SURFACE, Scenario, go, mo


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
        """
    )
    return


@app.cell
def _(Planner, mo):
    with mo.status.spinner(title="Entraînement des modèles (environ 5 s)…"):
        planner = Planner()
    return (planner,)


@app.cell
def _(mo):
    form = (
        mo.md(
            """
            **Budget (€)** {budget} **Coût e-mail homme (€)** {cost_h} **Coût e-mail femme (€)** {cost_f}

            **Envois max. e-mail homme** {max_h} **Envois max. e-mail femme** {max_f} *(0 = sans limite)*

            **Objectif** {objective} **Valeur d'une visite (€, objectif profit)** {value}
            """
        )
        .batch(
            budget=mo.ui.number(start=0, stop=10_000, step=10, value=700),
            cost_h=mo.ui.number(start=0, stop=10, step=0.01, value=0.12),
            cost_f=mo.ui.number(start=0, stop=10, step=0.01, value=0.10),
            max_h=mo.ui.number(start=0, stop=20_000, step=100, value=0),
            max_f=mo.ui.number(start=0, stop=20_000, step=100, value=0),
            objective=mo.ui.radio(
                options={"Maximiser les visites": "visites", "Maximiser le profit": "profit"},
                value="Maximiser les visites",
                inline=True,
            ),
            value=mo.ui.number(start=0, stop=100, step=0.5, value=5.0),
        )
        .form(submit_button_label="Calculer le plan", bordered=True)
    )
    form
    return (form,)


@app.cell
def _(Scenario, form, mo, planner):
    _v = form.value or {
        "budget": 700,
        "cost_h": 0.12,
        "cost_f": 0.10,
        "max_h": 0,
        "max_f": 0,
        "objective": "visites",
        "value": 5.0,
    }
    scenario = Scenario(
        budget=float(_v["budget"]),
        cost_homme=float(_v["cost_h"]),
        cost_femme=float(_v["cost_f"]),
        max_homme=int(_v["max_h"]) or None,
        max_femme=int(_v["max_f"]) or None,
        objective=_v["objective"],
        value_per_visit=float(_v["value"]),
    )
    _errors = scenario.errors()
    if _errors:
        plan = None
        _out = mo.callout(mo.md("\n".join(f"- {e}" for e in _errors)), kind="danger")
    else:
        with mo.status.spinner(title="Optimisation et mesure du plan…"):
            plan = planner.plan(scenario, n_boot=300)
        _out = None
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
    mo.hstack(
        [
            mo.stat(
                f"{_s['clients_contactes']:,}".replace(",", " "),
                label="Clients contactés",
                caption=f"homme {_s['email_homme']:,} · femme {_s['email_femme']:,}".replace(
                    ",", " "
                ),
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
            mo.stat(_gain, label="Gain mesuré vs ciblage classique", caption=_ci, bordered=True),
            mo.stat(
                f"≤ {100 * plan.bound_gap:.2f} %",
                label="Écart prouvé à l'optimum",
                caption=f"calculé en {plan.seconds:.2f} s",
                bordered=True,
            ),
        ],
        widths="equal",
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
                - **Écart prouvé à l'optimum** : la relaxation lagrangienne fournit une borne ; le plan
                  est au plus à cet écart de la meilleure allocation possible.
                - **Coûts** : ce sont des hypothèses saisies ici (les données n'en contiennent pas).
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
