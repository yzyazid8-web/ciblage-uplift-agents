"""Vitrine statique du projet (GitHub Pages, gratuite) : docs/index.html, générée par script.

Usage :
    python -m ciblage.serve.showcase --enregistrer-conversation   # vraie conversation avec l'agent (Groq)
    python -m ciblage.serve.showcase                              # (re)génère docs/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from ciblage import config
from ciblage.serve.planner import Planner, Scenario

DOCS = config.ROOT / "docs"
CONVERSATION_FILE = DOCS / "conversation.json"
REPO_URL = "https://github.com/yzyazid8-web/ciblage-uplift-agents"
BUDGETS = (200.0, 400.0, 700.0, 1000.0, 1500.0)
CONVERSATION = [
    "J'ai 700 €, l'e-mail homme coûte 0,12 € et l'e-mail femme 0,10 €. Maximise les visites.",
    "Est-ce que 100 € de plus valent le coup ?",
    "Pourquoi le client 12 est-il contacté ou non ?",
    "Qui sont les clients écartés ?",
    "Et avec 1 000 € ?",
    "Compare les scénarios.",
]


# données
def record_conversation(path: Path = CONVERSATION_FILE) -> list[dict[str, Any]]:
    """Enregistre une VRAIE conversation avec l'agent (nécessite la clé Groq)."""
    from ciblage.agents.copilot import Copilot

    copilot = Copilot.from_env(db_path=None)
    turns = []
    for message in CONVERSATION:
        r = copilot.send("vitrine", message)
        turns.append(
            {
                "message": message,
                "type": r.kind,
                "reponse": r.text,
                "etapes": r.trace[1:],
                "appels_llm": r.llm_calls,
            }
        )
    payload = {"date": date.today().isoformat(), "tours": turns}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return turns


def precompute(planner: Planner) -> list[dict[str, Any]]:
    """Plans pré-calculés (et mesurés) pour chaque budget du curseur."""
    out = []
    for budget in BUDGETS:
        plan = planner.plan(Scenario(budget=budget), n_boot=300)
        s = plan.summary
        out.append(
            {
                "budget": budget,
                "contactes": s["clients_contactes"],
                "homme": s["email_homme"],
                "femme": s["email_femme"],
                "depense": round(s["depense"], 2),
                "visites": round(s["visites_en_plus_predites"]),
                "cout_visite": round(s["cout_par_visite_en_plus"], 2),
                "prix_marginal": round(s["prix_marginal_budget"], 3),
                "ecart_optimum_pct": round(100 * plan.bound_gap, 3),
                "politiques": [
                    {
                        "nom": r["politique"],
                        "taux": round(100 * r["taux de visite"], 2),
                        "bas": round(100 * r["ic_bas"], 2),
                        "haut": round(100 * r["ic_haut"], 2),
                        "gain": round(r["gain vs classique (pts)"], 2),
                        "gain_bas": round(r["gain_bas"], 2),
                        "gain_haut": round(r["gain_haut"], 2),
                        "contactes": int(r["contactés"]),
                    }
                    for r in plan.comparison.to_dict(orient="records")
                ],
                "segments": [
                    {
                        "segment": r["segment"],
                        "homme": int(r["e-mail homme"]),
                        "femme": int(r["e-mail femme"]),
                        "rien": int(r["rien"]),
                    }
                    for r in plan.by_segment.to_dict(orient="records")
                ],
            }
        )
    return out


# extraits des rapports (markdown -> HTML minimal, sans dépendance)
def _inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def md_to_html(md: str) -> str:
    """Paragraphes, listes et tableaux markdown -> HTML (suffisant pour nos rapports)."""
    blocks, lines, i = [], md.strip().splitlines(), 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= {"-", ":"} for c in cells):
                    rows.append(cells)
                i += 1
            head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]
            )
            blocks.append(
                f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
            )
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        if line.startswith("*Durée totale"):  # métadonnée de fin de rapport, sans intérêt ici
            i += 1
            continue
        if line and not line.startswith("#"):
            blocks.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "\n".join(blocks)


def report_section(report: str, heading: str) -> str:
    """Contenu d'une section d'un rapport, du titre donné jusqu'au titre suivant de même niveau."""
    path = config.REPORTS_DIR / report
    if not path.exists():
        return f"<p><em>Rapport {html.escape(report)} non disponible.</em></p>"
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^(#+) {re.escape(heading)}.*$", text, flags=re.MULTILINE)
    if not m:
        return f"<p><em>Section « {html.escape(heading)} » introuvable dans {html.escape(report)}.</em></p>"
    level = len(m.group(1))
    rest = text[m.end() :]
    nxt = re.search(rf"^#{{1,{level}}} ", rest, flags=re.MULTILINE)
    return md_to_html(rest[: nxt.start()] if nxt else rest)


# page
CSS = """
:root{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--blue:#2a78d6;--orange:#eb6834;--neutral:#b4b2aa;--accent-bg:#eef4fc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#222220;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);
--blue:#3987e5;--orange:#d95926;--neutral:#6b6a64;--accent-bg:#1f2a38}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:2rem;line-height:1.2;margin:0 0 8px}h2{font-size:1.35rem;margin:48px 0 12px}
h3{font-size:1.05rem;margin:24px 0 8px}p,li{color:var(--ink2)}strong{color:var(--ink)}
a{color:var(--blue)}code{background:var(--panel);border:1px solid var(--border);padding:1px 5px;border-radius:4px}
.lead{font-size:1.12rem}.banner{background:var(--accent-bg);border-radius:8px;padding:12px 16px;margin:16px 0}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
@media (max-width:640px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.kpi .v{font-size:1.5rem;font-weight:650;color:var(--ink)}.kpi .l{font-size:.85rem;color:var(--ink2)}
.kpi .c{font-size:.8rem;color:var(--muted)}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:12px 0}
.controls input{flex:1;min-width:200px}.budget{font-weight:650;font-size:1.1rem;min-width:90px}
.chart{background:var(--panel);border:1px solid var(--border);border-radius:8px;margin:12px 0;min-height:320px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem;margin:8px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}th{color:var(--ink)}td{color:var(--ink2)}
.turn{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:12px 0}
.q{font-weight:600;color:var(--ink)}.steps{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.step{font-size:.75rem;color:var(--ink2);border:1px solid var(--border);border-radius:999px;padding:2px 8px}
.a{white-space:pre-wrap;color:var(--ink2)}footer{margin-top:48px;color:var(--muted);font-size:.85rem}
details{margin:8px 0}summary{cursor:pointer;color:var(--ink)}
"""

JS = """
const D = JSON.parse(document.getElementById('donnees').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fr = (x, d=0) => x.toLocaleString('fr-FR', {minimumFractionDigits:d, maximumFractionDigits:d});
function layout(title){ return {title:{text:title,x:0,font:{color:css('--ink'),size:15}},
  paper_bgcolor:css('--panel'), plot_bgcolor:css('--panel'), font:{color:css('--ink2')},
  margin:{l:10,r:20,t:48,b:48}, height:320, xaxis:{gridcolor:css('--grid'),zeroline:false,automargin:true},
  yaxis:{automargin:true}}; }
function render(i){
  const p = D.plans[i];
  document.getElementById('budget').textContent = fr(p.budget) + ' €';
  const k = [
    [fr(p.contactes), 'clients contactés', 'homme ' + fr(p.homme) + ' · femme ' + fr(p.femme)],
    [fr(p.depense, 2) + ' €', 'dépense', ''],
    [fr(p.visites), 'visites en plus (prédites)', fr(p.cout_visite, 2) + ' € par visite gagnée'],
    [fr(p.prix_marginal, 2), 'visites par euro en plus', 'seuil de rentabilité'],
    ['≤ ' + fr(p.ecart_optimum_pct, 2) + ' %', 'écart prouvé à l’optimum', 'relaxation lagrangienne']];
  const opt = p.politiques.find(x => x.nom === 'plan optimisé');
  k.push([(opt.gain>=0?'+':'') + fr(opt.gain, 2) + ' pt', 'gain mesuré vs ciblage classique',
          'IC 95 % [' + fr(opt.gain_bas, 2) + ' ; ' + fr(opt.gain_haut, 2) + ']']);
  document.getElementById('kpis').innerHTML = k.map(([v,l,c]) =>
    `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div><div class="c">${c}</div></div>`).join('');
  const pol = [...p.politiques].reverse();
  Plotly.react('graph-politiques', [{type:'scatter', mode:'markers', x:pol.map(r=>r.taux), y:pol.map(r=>r.nom),
    marker:{size:12, color:pol.map(r=>r.nom==='plan optimisé'?css('--blue'):css('--neutral')), line:{color:css('--panel'),width:2}},
    error_x:{type:'data', symmetric:false, array:pol.map(r=>r.haut-r.taux), arrayminus:pol.map(r=>r.taux-r.bas), color:css('--muted'), thickness:2, width:0},
    customdata:pol.map(r=>[r.contactes, r.gain]),
    hovertemplate:'<b>%{y}</b><br>Taux de visite : %{x:.2f} %<br>Contactés : %{customdata[0]:,}<br>Gain vs classique : %{customdata[1]:+.2f} pt<extra></extra>'}],
    Object.assign(layout('Taux de visite mesuré sur la campagne (IC 95 %)'), {xaxis:{title:'Taux de visite (%)', gridcolor:css('--grid'), zeroline:false}}),
    {displayModeBar:false, responsive:true});
  const seg = p.segments, series = [['homme','e-mail homme','--blue'],['femme','e-mail femme','--orange'],['rien','rien','--neutral']];
  Plotly.react('graph-segments', series.map(([key,name,col]) => ({type:'bar', orientation:'h', name,
    y:seg.map(s=>s.segment), x:seg.map(s=>s[key]), marker:{color:css(col), line:{color:css('--panel'),width:2}},
    hovertemplate:'<b>%{y}</b><br>'+name+' : %{x:,} clients<extra></extra>'})),
    Object.assign(layout('Qui reçoit quoi, selon ce que le client a déjà acheté'), {barmode:'stack',
      legend:{orientation:'h', y:-0.25, traceorder:'normal'}, xaxis:{title:'Clients', gridcolor:css('--grid')}}),
    {displayModeBar:false, responsive:true});
}
const slider = document.getElementById('curseur');
slider.addEventListener('input', () => render(+slider.value));
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render(+slider.value));
render(+slider.value);
"""


def _conversation_html() -> str:
    if not CONVERSATION_FILE.exists():
        return "<p><em>Conversation pas encore enregistrée.</em></p>"
    data = json.loads(CONVERSATION_FILE.read_text(encoding="utf-8"))
    parts = [
        f"<p>Enregistrée le {html.escape(data['date'])} avec l'agent réel (Groq, gpt-oss-120b).</p>"
    ]
    for t in data["tours"]:
        steps = "".join(f'<span class="step">{html.escape(s)}</span>' for s in t["etapes"])
        parts.append(
            f'<div class="turn"><div class="q">› {html.escape(t["message"])}</div>'
            f'<div class="steps">{steps}</div><div class="a">{html.escape(t["reponse"])}</div></div>'
        )
    return "\n".join(parts)


def build(planner: Planner | None = None, output: Path = DOCS / "index.html") -> Path:
    plans = precompute(planner or Planner())
    start = BUDGETS.index(700.0)
    data = json.dumps({"plans": plans}, ensure_ascii=False).replace(
        "</", r"<\/"
    )  # pas de « </ » dans un <script>
    sections = {
        "uplift": report_section("phase2_uplift.md", "À e-mail égal"),
        "simulateur": report_section("phase3_optim.md", "4. Contrôle sur le simulateur"),
        "eval1": report_section("phase5_eval_agent.md", "Niveau 1"),
        "avant_apres": report_section("phase5_eval_agent.md", "Avant / après"),
        "eval2": report_section("phase5_eval_agent.md", "Niveau 2"),
    }
    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Copilote de ciblage</title><meta name="description" content="Uplift, optimisation sous budget et agent LLM vérifié : qui contacter, avec quel e-mail, pour un budget donné.">
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script></head>
<body><main>
<h1>Copilote de ciblage causal multi-agents</h1>
<p class="lead"><strong>Qui contacter, avec quel e-mail, pour un budget donné ?</strong> Le système estime
pour chaque client ce que chaque e-mail change <em>vraiment</em> (uplift), optimise l'allocation sous budget,
vérifie le plan et l'explique, avec des chiffres contrôlés.</p>
<div class="banner">Démo <strong>pré-calculée</strong> sur une vraie campagne randomisée (Hillstrom, 19 200 clients
jamais vus par les modèles). L'application complète, avec l'agent en direct, se lance avec Docker :
<a href="{REPO_URL}">code et instructions sur GitHub</a>.</div>

<h2>Le plan selon le budget</h2>
<p>Coûts supposés : e-mail homme 0,12 €, e-mail femme 0,10 €. Les taux de visite sont <strong>mesurés</strong> sur
la campagne randomisée (pondération inverse), pas seulement prédits.</p>
<div class="controls"><label for="curseur">Budget</label>
<input id="curseur" type="range" min="0" max="{len(BUDGETS) - 1}" step="1" value="{start}">
<span class="budget" id="budget"></span></div>
<div class="kpis" id="kpis"></div>
<div class="chart" id="graph-politiques"></div>
<div class="chart" id="graph-segments"></div>

<h2>Une vraie conversation avec l'agent</h2>
<p>Le LLM <strong>comprend et rédige</strong> ; il ne calcule jamais. Chaque nombre de ses réponses est vérifié
contre les calculs avant affichage. Les étapes du graphe apparaissent au-dessus de chaque réponse.</p>
{_conversation_html()}

<h2>Résultats clés (repris des rapports générés)</h2>
<h3>L'uplift vise-t-il mieux que le ciblage classique, à e-mail égal ?</h3>{sections["uplift"]}
<h3>Quand l'effet varie d'un client à l'autre : contrôle sur données simulées</h3>{sections["simulateur"]}
<h3>Évaluation automatique de l'agent : compréhension</h3>{sections["eval1"]}
<details><summary>Avant / après : ce que la première évaluation a permis de corriger</summary>{sections["avant_apres"]}</details>
<h3>Évaluation automatique de l'agent : conversations complètes</h3>{sections["eval2"]}

<h2>Limites</h2>
<ul><li>Une seule campagne (2008) ; les coûts des e-mails sont des hypothèses.</li>
<li>L'uplift d'un client est estimé, jamais observé ; la mesure repose sur la randomisation de la campagne.</li>
<li>Sur Hillstrom, l'e-mail homme a un effet presque uniforme : peu à optimiser, et le plan fait jeu égal avec
le ciblage classique. La méthode fait la différence quand l'effet varie (e-mail femme, données simulées).</li></ul>
<footer>Page générée le {date.today().isoformat()} par <code>python -m ciblage.serve.showcase</code>.</footer>
</main>
<script id="donnees" type="application/json">{data}</script>
<script>{JS}</script></body></html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    # GitHub Pages : publier les fichiers tels quels, sans traitement Jekyll
    (output.parent / ".nojekyll").touch()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Vitrine statique du projet")
    parser.add_argument("--enregistrer-conversation", action="store_true")
    args = parser.parse_args()
    if args.enregistrer_conversation:
        record_conversation()
    print(f"Vitrine écrite : {build()}")


if __name__ == "__main__":
    main()
